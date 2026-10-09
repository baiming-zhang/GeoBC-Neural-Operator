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
    elif "geobc.physics_train" in command:
        from geobc.physics_train import parser as physics_parser, _scope
        parsed = physics_parser().parse_args(command[command.index("geobc.physics_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its displacement contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.gradient_train" in command:
        from geobc.gradient_train import parser as gradient_parser, _scope
        parsed = gradient_parser().parse_args(command[command.index("geobc.gradient_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its gradient-supervision contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.gradient_balance_train" in command:
        from geobc.gradient_balance_train import parser as balance_parser, _scope
        parsed = balance_parser().parse_args(command[command.index("geobc.gradient_balance_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its calibrated-gradient contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.bc_role_train" in command:
        from geobc.bc_role_train import parser as role_parser, _scope
        parsed = role_parser().parse_args(command[command.index("geobc.bc_role_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its BC-role displacement contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.specialist_train" in command:
        from geobc.specialist_train import parser as specialist_parser, _scope
        parsed = specialist_parser().parse_args(command[command.index("geobc.specialist_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed TRAIN-CAD contract")
        for filename in ("last.pt", "endpoint_250.pt", "endpoint_10000.pt", "metrics_endpoint_250.json",
                         "metrics_endpoint_10000.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.query_context_train" in command:
        from geobc.query_context_train import parser as query_parser, _scope
        parsed = query_parser().parse_args(command[command.index("geobc.query_context_train") + 1:])
        contract = _scope(parsed)[4]
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its query-context displacement contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "tools/diagnose_query_learning.py" in command:
        # Frozen diagnostics have no optimizer/checkpoint outputs. Compare their
        # small request/source/last-checkpoint metadata before allowing reuse.
        import importlib.util
        spec = importlib.util.spec_from_file_location("query_learning_diagnostic", ROOT / "tools/diagnose_query_learning.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        contract = module.request_contract(requested_plan)
        if summary.get("config", {}).get("contract") != contract:
            raise ValueError(f"Completed output for {job['name']} differs from its frozen diagnostic contract")
        if not (ROOT / job["output"] / "config.json").is_file():
            raise ValueError(f"Completed output for {job['name']} lacks config.json")
    elif "tools/diagnose_hash_modulation_use.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_modulation_use_diagnostic", ROOT / "tools/diagnose_hash_modulation_use.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen hash gate-use diagnostic contract")
        if not (ROOT / job["output"] / "config.json").is_file():
            raise ValueError(f"Completed output for {job['name']} lacks config.json")
    elif "tools/diagnose_field_error_gradient.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("field_error_gradient_diagnostic", ROOT / "tools/diagnose_field_error_gradient.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen field-error contract")
        for path in (ROOT / job["output"] / "config.json", ROOT / "notes/geobc_error_gradient_result.json"):
            if not path.is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {path}")
    elif "tools/validate_hash_modulation_native.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_modulation_native_check", ROOT / "tools/validate_hash_modulation_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its native interface check")
        if not (ROOT / "notes/hash_modulation_native_ready.json").is_file():
            raise ValueError(f"Completed native check for {job['name']} lacks readiness record")
    elif "tools/validate_graph_context_native.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("graph_context_native_check", ROOT / "tools/validate_graph_context_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its graph-context native check")
        if not (ROOT / "notes/graph_context_native_ready.json").is_file():
            raise ValueError(f"Completed native check for {job['name']} lacks readiness record")
    elif "tools/prepare_topology_context_balanced.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("balanced_topology_context_preparation", ROOT / "tools/prepare_topology_context_balanced.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its balanced topology contract")
        result_path = ROOT / "notes/geobc_topology_balanced_feasibility_result.json"
        if not result_path.is_file() or read_json_if_exists(result_path).get("status") != "completed":
            raise ValueError(f"Completed input diagnostic for {job['name']} lacks its formal result")
    elif "tools/prepare_topology_context.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("topology_context_preparation", ROOT / "tools/prepare_topology_context.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its topology input contract")
        result_path = ROOT / "notes/geobc_topology_context_feasibility_result.json"
        if not result_path.is_file() or read_json_if_exists(result_path).get("status") != "completed":
            raise ValueError(f"Completed input diagnostic for {job['name']} lacks its formal result")
    elif "tools/diagnose_hash_condition_paths.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_condition_path_diagnostic", ROOT / "tools/diagnose_hash_condition_paths.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its frozen H/C path contract")
        result_path = ROOT / "notes/geobc_hash_condition_path_result.json"
        if not result_path.is_file() or read_json_if_exists(result_path).get("status") != "completed":
            raise ValueError(f"Completed diagnostic for {job['name']} lacks its formal result")
    elif "tools/validate_hash_bandwidth_native.py" in command:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hash_bandwidth_native_check", ROOT / "tools/validate_hash_bandwidth_native.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        requested_plan = Path(command[command.index("--plan") + 1])
        if not requested_plan.is_absolute(): requested_plan = ROOT / requested_plan
        if summary.get("config", {}).get("contract") != module.request_contract(requested_plan):
            raise ValueError(f"Completed output for {job['name']} differs from its bandwidth native check")
        if not (ROOT / "notes/hash_bandwidth_native_ready.json").is_file():
            raise ValueError(f"Completed native check for {job['name']} lacks readiness record")
    elif "geobc.topology_message_train" in command:
        from geobc.topology_message_train import parser as topology_parser, _scope
        parsed = topology_parser().parse_args(command[command.index("geobc.topology_message_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[3]:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed topology-message contract")
        if summary.get("endpoint_step") != 1000 or summary.get("message_mode") != parsed.message_mode:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed endpoint/mode")
        for filename in ("last.pt", "endpoint_1000.pt", "metrics_step0000.json", "metrics_step1000.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.hash_bandwidth_train" in command:
        from geobc.hash_bandwidth_train import parser as bandwidth_parser, _scope
        parsed = bandwidth_parser().parse_args(command[command.index("geobc.hash_bandwidth_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[4]:
            raise ValueError(f"Completed output for {job['name']} differs from its fixed bandwidth contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.graph_context_train" in command:
        from geobc.graph_context_train import parser as graph_parser, _scope
        parsed = graph_parser().parse_args(command[command.index("geobc.graph_context_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[4]:
            raise ValueError(f"Completed output for {job['name']} differs from its graph-context contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    elif "geobc.hash_modulation_train" in command:
        from geobc.hash_modulation_train import parser as modulation_parser, _scope
        parsed = modulation_parser().parse_args(command[command.index("geobc.hash_modulation_train") + 1:])
        if summary.get("config", {}).get("contract") != _scope(parsed)[4]:
            raise ValueError(f"Completed output for {job['name']} differs from its hash modulation contract")
        for filename in ("best.pt", "last.pt", "metrics_best.json", "metrics_latest.json", "config.json"):
            if not (ROOT / job["output"] / filename).is_file():
                raise ValueError(f"Completed output for {job['name']} lacks {filename}")
    return summary


def stage_command(job: dict[str, Any]) -> list[str]:
    command = job["command"].copy()
    output = ROOT / job["output"]
    fitting = ("geobc.exploration" in command and "--mode" in command
               and command[command.index("--mode") + 1] == "fit")
    if ((fitting or "geobc.conditional_train" in command or "geobc.physics_train" in command
         or "geobc.gradient_train" in command or "geobc.gradient_balance_train" in command
         or "geobc.bc_role_train" in command or "geobc.specialist_train" in command
         or "geobc.query_context_train" in command or "geobc.hash_modulation_train" in command
         or "geobc.graph_context_train" in command or "geobc.hash_bandwidth_train" in command
         or "geobc.topology_message_train" in command)
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
