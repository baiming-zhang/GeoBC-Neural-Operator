"""One-shot external wall guard for the reviewed TRAIN-only FNO local-condition pair."""
import time

ENTRY_EPOCH = time.time()
ENTRY_MONOTONIC = time.monotonic()

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_fno_local_condition_pair_plan.json"
BUDGET = 9000


def process_identity(pid):
    """Capture the actual Linux start identity before an endpoint can exit."""
    try:
        folder = Path("/proc") / str(pid)
        raw = (folder / "stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        argv = [part.decode("utf-8") for part in (folder / "cmdline").read_bytes().split(b"\0") if part]
        return {"pid": pid, "ppid": int(tail[1]), "start_ticks": int(tail[19]), "argv": argv}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def wait_dual_deadline(child, deadline_epoch, deadline_monotonic):
    """Recheck both actual clock axes instead of one long monotonic wait."""
    while True:
        remaining = min(deadline_epoch - time.time(), deadline_monotonic - time.monotonic())
        if remaining <= 0:
            raise subprocess.TimeoutExpired(child.args, 0)
        try:
            return child.wait(timeout=min(1., remaining))
        except subprocess.TimeoutExpired:
            pass


def terminate(child):
    """Keep an actual stop/reap failure in the terminal receipt."""
    try:
        if child.poll() is not None:
            return child.returncode, None
        child.terminate()
        try:
            return child.wait(timeout=5), None
        except subprocess.TimeoutExpired:
            child.kill()
            return child.wait(timeout=5), None
    except BaseException as error:
        return child.returncode, type(error).__name__


def elapsed(record):
    ended_epoch = time.time(); ended_monotonic = time.monotonic()
    mono = ended_monotonic - ENTRY_MONOTONIC
    epoch = ended_epoch - ENTRY_EPOCH
    record.update(ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat(),
                  ended_epoch=ended_epoch, ended_monotonic=ended_monotonic,
                  elapsed_launch_to_exit_seconds=mono,
                  elapsed_epoch_launch_to_exit_seconds=epoch,
                  clock_axis_difference_seconds=epoch - mono)


def within_budget(record):
    return (0 <= record["elapsed_launch_to_exit_seconds"] <= BUDGET
            and 0 <= record["elapsed_epoch_launch_to_exit_seconds"] <= BUDGET)


def main():
    if os.name == "nt":
        raise SystemExit("Use the project's WSL process namespace")
    plan = json.loads((ROOT / PLAN).read_text(encoding="utf-8"))
    review = json.loads((ROOT / plan["root_review"]).read_text(encoding="utf-8"))
    if not (review.get("passed") is True and review.get("fno_pair_released") is True
            and review.get("actual_execution_released") is True):
        raise SystemExit("Actual FNO pair has not been released by root")
    output = (ROOT / plan["output"]).resolve()
    if ROOT not in output.parents:
        raise SystemExit("Output must stay in this project")
    output.mkdir(parents=True, exist_ok=True)
    record = {"status": "starting", "started_at_utc": datetime.fromtimestamp(ENTRY_EPOCH, timezone.utc).isoformat(),
              "entry_epoch": ENTRY_EPOCH, "entry_monotonic": ENTRY_MONOTONIC,
              "deadline_epoch": ENTRY_EPOCH + BUDGET,
              "deadline_monotonic": ENTRY_MONOTONIC + BUDGET,
              "external_budget_seconds": BUDGET, "automatic_replay": False,
              "guard_process": process_identity(os.getpid()),
              "guard_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    marker = output / "external_launch.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
    argv = [sys.executable, "-u", "-m", "geobc.fno_local_condition_pair", "--plan", PLAN, "--start"]
    child = None; code = None; timed_out = False; interrupted = False
    try:
        if min(record["deadline_epoch"] - time.time(),
               record["deadline_monotonic"] - time.monotonic()) <= 0:
            timed_out = True
        else:
            with (output / "process.log").open("x", encoding="utf-8") as log:
                if min(record["deadline_epoch"] - time.time(),
                       record["deadline_monotonic"] - time.monotonic()) <= 0:
                    timed_out = True
                else:
                    child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    record.update(status="running", child_pid=child.pid, child_argv=argv,
                                  child_process=process_identity(child.pid))
                    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                    try:
                        code = wait_dual_deadline(child, record["deadline_epoch"], record["deadline_monotonic"])
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        code, stop_error = terminate(child)
                        if stop_error:
                            record["termination_error_class"] = stop_error
    except BaseException as error:
        interrupted = True
        record["exception_class"] = type(error).__name__
        raise
    finally:
        if child is not None and child.poll() is None:
            code, stop_error = terminate(child)
            if stop_error:
                record["termination_error_class"] = stop_error
        actual = record.get("child_process")
        current = process_identity(actual["pid"]) if actual is not None else None
        exited = (actual is not None and code is not None and child.poll() is not None
                  and (current is None or current["start_ticks"] != actual["start_ticks"]))
        record.update(status="terminal", exit_code=code, external_timeout=timed_out,
                      interrupted=interrupted, actual_child_exited=exited,
                      summary_present=(output / "summary.json").is_file(),
                      in_flight_costs_exact=not timed_out and not interrupted and exited
                          and "termination_error_class" not in record)
        elapsed(record)
        record["within_external_budget"] = (not timed_out and not interrupted and exited
            and "termination_error_class" not in record and within_budget(record))
        marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        elapsed(record)
        record["within_external_budget"] = record["within_external_budget"] and within_budget(record)
        marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        # Check after all normal persistence; any boundary overrun stays negative.
        if min(record["deadline_epoch"] - time.time(),
               record["deadline_monotonic"] - time.monotonic()) < 0:
            elapsed(record)
            record["within_external_budget"] = False
            marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    raise SystemExit(0 if code == 0 and record["within_external_budget"] else 1)


if __name__ == "__main__":
    main()
