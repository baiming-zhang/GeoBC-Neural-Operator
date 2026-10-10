"""One-shot external wall guard for the independently released TRAIN diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/geobc_coverage_comparison/TRAIN_FIT_DIAGNOSTIC"


def main():
    if os.name == "nt":
        raise SystemExit("Run in the project WSL environment")
    review = json.loads((ROOT / "notes/geobc_train_fit_diagnostic_root_review.json").read_text(encoding="utf-8"))
    if review.get("diagnostic_released") is not True or review.get("passed") is not True:
        raise SystemExit("The actual diagnostic has not been released")
    marker = OUTPUT / "external_launch.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "external_budget_seconds": 600, "automatic_replay": False,
              "guard_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
    started = time.monotonic()
    argv = [sys.executable, "-u", "-m", "geobc.coverage_train_fit_diagnostic", "--start"]
    with (OUTPUT / "process.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(status="running", child_pid=child.pid, child_argv=argv)
        marker.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        timed_out = False
        try:
            code = child.wait(timeout=max(0., 600 - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=5)
    record.update(status="terminal", exit_code=code, external_timeout=timed_out,
                  elapsed_launch_to_exit_seconds=time.monotonic() - started,
                  ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  within_external600s=(not timed_out and time.monotonic() - started <= 600),
                  summary_present=(OUTPUT / "summary.json").is_file(),
                  in_flight_read_counts_exact=not timed_out)
    marker.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, allow_nan=False), flush=True)
    raise SystemExit(0 if code == 0 and not timed_out else 1)


if __name__ == "__main__":
    main()
