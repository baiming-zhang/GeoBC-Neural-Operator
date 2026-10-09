"""Run the current GeoBC plan through one recoverable serial GPU queue.

The JSON plan is reloaded between stages and while waiting for inputs, allowing
new jobs to be appended during a running stage. Completed outputs are reused.
Dry runs read local artifacts only and do not acquire locks or create outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geobc.exploration import exclusive_lock, identity_digest, request_identity, parser as exploration_parser
from geobc.train import utc_now, write_json


def load_jobs(plan: Path, only: list[str] | None = None) -> list[dict[str, Any]]:
    record = json.loads(plan.read_text(encoding="utf-8"))
    jobs = record["jobs"]
    names = [job["name"] for job in jobs]
    outputs = [str((ROOT / job["output"]).resolve()) for job in jobs]
    if len(names) != len(set(names)) or len(outputs) != len(set(outputs)):
        raise ValueError("Plan job names and output directories must be unique")
    for job in jobs:
        if (not isinstance(job["command"], list) or not job["command"]
                or any(not isinstance(item, str) for item in job["command"])):
            raise ValueError(f"Invalid argv for {job['name']}")
        if not isinstance(job.get("requires", []), list):
            raise ValueError(f"Invalid requires for {job['name']}")
    if only:
        missing = set(only).difference(names)
        if missing:
            raise ValueError(f"Unknown --only jobs: {sorted(missing)}")
        jobs = [job for job in jobs if job["name"] in only]
    return jobs


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def completed_summary(job: dict[str, Any]) -> dict[str, Any] | None:
    summary = read_json_if_exists(ROOT / job["output"] / "summary.json")
    if not summary or summary.get("status") != "completed":
        return None
    command = job["command"]
    if "geobc.exploration" in command:
        cli_args = command[command.index("geobc.exploration") + 1:]
        parsed = exploration_parser().parse_args(cli_args)
        if parsed.steps is None and parsed.epochs is None:
            parsed.steps = 2000
        if summary["config"].get("request_digest") != identity_digest(request_identity(parsed)):
            raise ValueError(f"Completed output for {job['name']} differs from current plan")
    elif "geobc.conditional_train" in command:
        # The trainer's own strict contract uses only small source/config
        # metadata here; no labels, checkpoints, models or CUDA are loaded.
        from geobc.conditional_train import parser as conditional_parser, _scope
        parsed = conditional_parser().parse_args(command[command.index("geobc.conditional_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from current strict configuration/data contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    return summary


def stage_command(job: dict[str, Any]) -> list[str]:
    command = job["command"].copy()
    output = ROOT / job["output"]
    fitting = ("geobc.exploration" in command and "--mode" in command
               and command[command.index("--mode") + 1] == "fit")
    if ((fitting or "geobc.conditional_train" in command)
            and "--resume" not in command and (output / "last.pt").exists()):
        command.extend(["--resume", str(output / "last.pt")])
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.chdir(ROOT)
    plan = args.plan.resolve()
    state_dir = args.state_dir.resolve()
    if args.dry_run:
        records = []
        for job in load_jobs(plan, args.only):
            summary = completed_summary(job)
            records.append({"name": job["name"], "completed_reuse": summary is not None,
                            "command": stage_command(job), "output": job["output"],
                            "missing_inputs": [path for path in job.get("requires", []) if not (ROOT / path).exists()]})
        result = {"dry_run": True, "plan": str(plan), "serial_GPU_queue": True,
                  "state_directory": str(state_dir), "dynamic_plan_reload": True, "jobs": records}
        print(json.dumps(result, indent=2), flush=True)
        return result
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / ".coordinator.lock"
    lock = exclusive_lock(lock_path)
    try:
        lock.__enter__()
    except RuntimeError:
        print(f"A live coordinator already holds {lock_path}; no duplicate jobs started", flush=True)
        return {"status": "already_running", "lock": str(lock_path)}
    started = time.monotonic()
    state = {"phase": "starting", "plan": str(plan), "pid": os.getpid(),
             "started_at_utc": utc_now(), "completed": [], "failures": [], "child_pid": None}
    def update(**extra):
        state.update(extra, elapsed_seconds=time.monotonic() - started, updated_at_utc=utc_now())
        write_json(state_dir / "coordinator_status.json", state)
    completed_names = set()
    try:
        update()
        while True:
            # Always reload the plan: never retain a stale list after a long stage.
            jobs = load_jobs(plan, args.only)
            state["total_jobs"] = len(jobs)
            pending = []
            for job in jobs:
                if job["name"] in completed_names:
                    continue
                summary = completed_summary(job)
                if summary:
                    completed_names.add(job["name"])
                    state["completed"].append({"name": job["name"], "reused": True,
                                               "elapsed_seconds": summary.get("elapsed_seconds")})
                    print(f"Reuse completed {job['name']}", flush=True)
                else:
                    pending.append(job)
            if not pending:
                update(phase="completed", current=None, child_pid=None, completed_at_utc=utc_now())
                print(f"GeoBC queue completed {len(completed_names)} stages", flush=True)
                return state
            job = pending[0]
            wait_started = time.monotonic()
            while True:
                refreshed = {entry["name"]: entry for entry in load_jobs(plan, args.only)}
                if job["name"] not in refreshed:
                    raise ValueError(f"Waiting job was removed from plan: {job['name']}")
                job = refreshed[job["name"]]
                missing = [path for path in job.get("requires", []) if not (ROOT / path).exists()]
                if not missing:
                    break
                update(phase="waiting_for_inputs", current=job["name"], missing_inputs=missing,
                       waiting_seconds=time.monotonic() - wait_started)
                if time.monotonic() - wait_started > args.input_timeout:
                    raise TimeoutError(f"Missing inputs for {job['name']}: {missing}")
                time.sleep(5)
            command = stage_command(job)
            update(phase="running", current=job["name"], current_stage=job.get("stage"),
                   command=command, missing_inputs=[], current_started_at_utc=utc_now())
            log_path = state_dir / (job["name"] + ".log")
            stage_started = time.monotonic()
            print(f"Start {job['name']}; log {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                update(child_pid=child.pid)
                while child.poll() is None:
                    time.sleep(5)
                    status = read_json_if_exists(ROOT / job["output"] / "status.json")
                    update(current_elapsed_seconds=time.monotonic() - stage_started,
                           child_status=status, eta_seconds=status.get("eta_seconds") if status else None,
                           estimated_completion_utc=status.get("estimated_completion_utc") if status else None)
            if child.returncode:
                failure = {"name": job["name"], "exit_code": child.returncode,
                           "log": str(log_path), "failed_at_utc": utc_now(),
                           "recovery": "rerun coordinator; completed stages reuse, fitting resumes last.pt"}
                state["failures"].append(failure)
                update(phase="failed", child_pid=None)
                raise RuntimeError(f"Stage failed: {job['name']}; see {log_path}")
            summary = completed_summary(job)
            if not summary:
                raise RuntimeError(f"Stage exited successfully without a completed summary: {job['name']}")
            completed_names.add(job["name"])
            state["completed"].append({"name": job["name"], "reused": False,
                                       "elapsed_seconds": time.monotonic() - stage_started})
            update(child_pid=None, child_status=None)
    except BaseException as error:
        update(phase="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("notes/geobc_exploration_plan.json"))
    parser.add_argument("--state-dir", type=Path, default=Path("runs/geobc_exploration"))
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-timeout", type=float, default=3600)
    run(parser.parse_args())
