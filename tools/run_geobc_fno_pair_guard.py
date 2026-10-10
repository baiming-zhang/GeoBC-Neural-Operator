"""One-shot external wall guard for the reviewed TRAIN-only FNO local-condition pair."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_fno_local_condition_pair_plan.json"
BUDGET = 9000


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
    record = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "external_budget_seconds": BUDGET, "automatic_replay": False,
              "guard_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    marker = output / "external_launch.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
    began = time.monotonic()
    argv = [sys.executable, "-u", "-m", "geobc.fno_local_condition_pair", "--plan", PLAN, "--start"]
    with (output / "process.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(status="running", child_pid=child.pid, child_argv=argv)
        marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        timed_out = False
        try:
            code = child.wait(timeout=max(0., BUDGET - (time.monotonic() - began)))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=5)
    elapsed = time.monotonic() - began
    record.update(status="terminal", exit_code=code, external_timeout=timed_out,
                  elapsed_launch_to_exit_seconds=elapsed, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  within_external_budget=not timed_out and elapsed <= BUDGET,
                  summary_present=(output / "summary.json").is_file(),
                  in_flight_costs_exact=not timed_out)
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    raise SystemExit(0 if code == 0 and not timed_out else 1)


if __name__ == "__main__":
    main()
