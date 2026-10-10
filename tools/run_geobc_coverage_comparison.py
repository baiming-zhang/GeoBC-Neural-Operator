"""Independent, recoverable serial queue for the frozen ten-arm coverage study.

No scientific dependencies are imported here. Frozen original TRAIN40 arms
can run during acquisition. The common GPU lock is released while waiting for
new inputs; all GPU preparation/model stages then run one process at a time.
Completed terminal arms are never silently retrained.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geobc.coverage_comparison_train import (KIND, arms, arm_output, digest, lock,
    path, read, release, request_contract, sha, utc, write)

OWN = "tools/run_geobc_coverage_comparison.py"


def existing(filename):
    try:
        return read(filename)
    except FileNotFoundError:
        return None


def prepared(plan):
    value = existing(path(plan["data_cache"]) / "manifest.json")
    return value if value and value.get("status") == "completed" else None


def status(output, phase, **details):
    record = {"kind": KIND, "phase": phase, "pid": os.getpid(), "updated_at_utc": utc(), **details}
    write(output / "coordinator_status.json", record)
    return record


def child(output, argv, expected_summary, phase, *, arm_id=None):
    """Process lifetime and exit are real; never infer scientific completion from PID."""
    tag = str(time.time_ns()); logs = output / "coordinator_logs"; logs.mkdir(parents=True, exist_ok=True)
    stdout = logs / (phase + "_" + tag + ".stdout.log")
    stderr = logs / (phase + "_" + tag + ".stderr.log")
    began = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err,
                                   env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        status(output, phase, child_pid=process.pid, arm_id=arm_id, command=argv,
               stdout=str(stdout), stderr=str(stderr), child_started_at_utc=utc())
        last = time.monotonic()
        while process.poll() is None:
            time.sleep(5)
            if time.monotonic() - last >= 30:
                progress = None
                if arm_id is not None:
                    progress = existing(output / arm_id / ("prediction_status.json" if phase == "predict" else "status.json"))
                status(output, phase, child_pid=process.pid, arm_id=arm_id, command=argv,
                       child_elapsed_seconds=time.monotonic() - began, progress=progress,
                       stdout=str(stdout), stderr=str(stderr))
                last = time.monotonic()
    elapsed = time.monotonic() - began
    receipt = {"command": argv, "returncode": process.returncode, "process_wall_seconds": elapsed,
               "stdout": str(stdout), "stderr": str(stderr), "arm_id": arm_id, "phase": phase}
    filename = output / "coordinator_receipts.json"
    previous = existing(filename) or []; previous.append(receipt); write(filename, previous)
    summary = existing(expected_summary)
    if process.returncode != 0 or not summary or summary.get("status") != "completed":
        raise RuntimeError(f"{phase} child did not commit a terminal scientific summary: {receipt}")
    return summary


def prepare(plan_path, plan, output, ensure_GPU):
    """Only source-authorized acquisition/preparation; targets obey data helper."""
    manifest = prepared(plan)
    if manifest is not None:
        return manifest
    raw_summary = path(plan["acquisition"]["output"]) / "summary.json"
    acquired = existing(raw_summary)
    if not acquired or acquired.get("status") != "completed":
        # Root may start the standalone fetch before the training source is
        # released. This queue waits for that owned binary-only result rather
        # than launching a second downloader or sampling/renaming missing CADs.
        while not acquired or acquired.get("status") != "completed":
            if acquired and acquired.get("status") in ("failed", "completed_negative", "budget_exhausted"):
                reason = acquired.get("terminal_reason", acquired.get("reason"))
                write(output / "acquisition_terminal.json", {"status": "completed", "kind": KIND,
                      "phase": "acquisition_terminal_negative", "acquisition_status": acquired["status"],
                      "terminal_reason": reason, "acquisition_summary_path": str(raw_summary),
                      "existing_TRAIN40_evidence_preserved": True, "no_case_replacement": True})
                raise RuntimeError(f"Frozen acquisition terminal {acquired['status']}; no CAD substitution; "
                                   f"summary={raw_summary}; terminal_reason={reason}")
            status(output, "waiting_for_acquisition", acquisition_summary=str(raw_summary),
                   available=acquired.get("counts") if acquired else None)
            time.sleep(30); acquired = existing(raw_summary)
    argv = [sys.executable, "-u", "-m", "geobc.coverage_comparison_data", "--plan", str(path(plan_path))]
    ensure_GPU()
    return child(output, argv, path(plan["data_cache"]) / "manifest.json", "prepare_inputs")


def run(plan_path, start=False):
    plan = read(plan_path); output = path(plan["training_output"])
    if not start:
        return {"status": "source_only_ready", "kind": plan["kind"], "output": str(output),
                "arm_order": arms(plan), "writes": False, "scientific_imports": 0,
                "actual_execution": "Only --start after the authentic root source/phase release; no implicit dry-run science"}
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        try:
            stack.enter_context(lock(output / ".coverage_coordinator.lock"))
        except (OSError, RuntimeError):
            return {"status": "already_running", "writes": False}
        status(output, "starting")
        try:
            # Before any source/raw preparation, verify the exact root source
            # pins and phase release without constructing a scientific model.
            while True:
                plan = read(plan_path)
                settings = plan["root_review"]
                marker = existing(settings.get("path", "notes/geobc_coverage_root_review.json"))
                if marker and marker.get("passed") is True and marker.get("preparation_and_training_released") is True:
                    if (marker.get("scientific_digest") != plan["scientific_digest"]
                            or marker.get("source_pins") != plan["source_sha256"]
                            or any(sha(name) != expected for name, expected in plan["source_sha256"].items())):
                        raise ValueError("Root science/source phase marker differs")
                    break
                status(output, "waiting_for_source_release", actual_scientific_execution=False)
                time.sleep(30)
            GPU_context = None
            shared = path("runs/geobc_exploration/.coordinator.lock")
            def ensure_GPU():
                nonlocal GPU_context
                if GPU_context is not None:
                    return
                while True:
                    try:
                        candidate = lock(shared); candidate.__enter__(); GPU_context = candidate; return
                    except (OSError, RuntimeError):
                        status(output, "waiting_for_single_GPU", shared_lock=str(shared)); time.sleep(30)
            def release_GPU():
                nonlocal GPU_context
                if GPU_context is not None:
                    previous = GPU_context; GPU_context = None; previous.__exit__(None, None, None)
            stack.callback(release_GPU)

            from geobc import coverage_comparison_data as data
            # This writes only a TRAIN120 metadata artifact, never new raw,
            # old DEV targets or a scientific array. The stable bootstrap
            # contract is also used after full-cache completion.
            data.bootstrap_TRAIN40(plan_path)
            def train_arm(arm):
                folder = arm_output(plan, arm)
                contract, _ = request_contract(plan_path, arm["method"], arm["train_CADs"], "train")
                release(contract, plan)
                saved = existing(folder / "summary.json")
                if saved and saved.get("status") == "completed":
                    if saved.get("config", {}).get("contract") != contract:
                        raise ValueError("Completed training arm identity differs: " + arm["id"])
                    return saved
                ensure_GPU()
                return child(output, [sys.executable, "-u", "-m", "geobc.coverage_comparison_train",
                      "--plan", str(path(plan_path)), "--stage", "train", "--method", arm["method"],
                      "--train-cads", str(arm["train_CADs"])], folder / "summary.json", "train", arm_id=arm["id"])
            def predict_arm(arm):
                folder = arm_output(plan, arm)
                infer = existing(folder / "prediction_summary.json")
                expected, _ = request_contract(plan_path, arm["method"], arm["train_CADs"], "predict")
                if infer and infer.get("status") == "completed":
                    if infer.get("config", {}).get("contract") != expected:
                        raise ValueError("Completed endpoint inventory identity differs: " + arm["id"])
                    return infer
                ensure_GPU()
                return child(output, [sys.executable, "-u", "-m", "geobc.coverage_comparison_train",
                      "--plan", str(path(plan_path)), "--stage", "predict", "--method", arm["method"],
                      "--train-cads", str(arm["train_CADs"])], folder / "prediction_summary.json", "predict", arm_id=arm["id"])
            ordered = arms(plan)
            if [a["train_CADs"] for a in ordered] != [40] * 5 + [200] * 5:
                raise ValueError("Bootstrap requires the already frozen TRAIN40 then TRAIN200 arm order")
            for arm in ordered[:5]:
                train_arm(arm)
            # A CPU-only acquisition wait owns no common GPU lock. Fresh
            # bootstrap training evidence remains committed if acquisition or
            # new preparation later ends negative.
            release_GPU()
            manifest = prepare(plan_path, plan, output, ensure_GPU)
            if manifest.get("feasibility_pass") is not True:
                record = {"status": "completed", "kind": KIND, "phase": "input_scientific_negative",
                          "fixed_arms": [a["id"] for a in arms(plan)], "fixed_DEV_denominator": 60,
                          "reason": manifest.get("reason", "Frozen input preparation incompatible"),
                          "input_manifest": str(path(plan["data_cache"]) / "manifest.json"),
                          "bootstrap_training_terminal": [a["id"] for a in ordered[:5]],
                          "new_scale_training_started": False, "no_case_replacement": True}
                write(output / "coordinator_summary.json", record); status(output, "completed", terminal_negative=True)
                return record
            evaluation = path(plan["training_output"]) / "evaluation" / "summary.json"
            completed = existing(evaluation)
            if completed and completed.get("status") == "completed":
                expected, _ = request_contract(plan_path, stage="evaluate")
                if completed.get("config", {}).get("contract") != expected:
                    raise ValueError("Completed global readout identity differs")
                status(output, "completed", reused=True); return completed
            # DEV predictions are delayed until all new input metadata is
            # ready; no labels are available before the complete inventory.
            ensure_GPU()
            for arm in ordered[:5]:
                predict_arm(arm)
            for arm in ordered[5:]:
                train_arm(arm); predict_arm(arm)
            result = child(output, [sys.executable, "-u", "-m", "geobc.coverage_comparison_train",
                           "--plan", str(path(plan_path)), "--stage", "evaluate"], evaluation, "evaluate")
            write(output / "coordinator_summary.json", {"status": "completed", "kind": KIND,
                  "scientific_digest": plan["scientific_digest"], "evaluation_summary": str(evaluation),
                  "fixed_arms": [a["id"] for a in arms(plan)], "fixed_DEV_denominator": 60})
            status(output, "completed", arms_terminal=10, DEV_readout_terminal=True)
            return result
        except BaseException as error:
            status(output, "failed", error=type(error).__name__ + ": " + str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="notes/geobc_coverage_comparison_plan.json")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args(); result = run(args.plan, args.start)
    print({"status": result["status"], "kind": result.get("kind", KIND)}, flush=True)


if __name__ == "__main__":
    main()
