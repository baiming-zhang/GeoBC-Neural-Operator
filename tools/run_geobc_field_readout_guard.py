"""One-shot external wall guard for the two independently released field readouts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "S": ("notes/geobc_s_field_blind_plan.json", "geobc.coverage_s_blind_eval", 720),
    "knownU": ("notes/geobc_known_u_readout_plan.json", "geobc.coverage_known_u_readout", 120),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SPECS)
    args = parser.parse_args()
    if os.name == "nt":
        raise SystemExit("Run in the project WSL environment")
    filename, module, budget = SPECS[args.stage]
    plan = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    review = json.loads((ROOT / plan["root_review"]).read_text(encoding="utf-8"))
    if review.get("passed") is not True or review.get("actual_execution_released") is not True:
        raise SystemExit("Actual field readout has not been released")
    output = (ROOT / plan["output"]).resolve()
    if ROOT not in output.parents:
        raise SystemExit("Output is outside this project")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "external_launch.json"
    record = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "external_budget_seconds": budget, "automatic_replay": False,
              "guard_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
    started = time.monotonic()
    argv = [sys.executable, "-u", "-m", module, "--plan", filename, "--start"]
    with (output / "process.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(status="running", child_pid=child.pid, child_argv=argv)
        marker.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        timed_out = False
        try:
            code = child.wait(timeout=max(0., budget - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=5)
    seconds = time.monotonic() - started
    record.update(status="terminal", exit_code=code, external_timeout=timed_out,
                  elapsed_launch_to_exit_seconds=seconds,
                  ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  within_external_budget=(not timed_out and seconds <= budget),
                  summary_present=(output / "summary.json").is_file(),
                  in_flight_read_counts_exact=not timed_out)
    marker.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, allow_nan=False), flush=True)
    raise SystemExit(0 if code == 0 and not timed_out else 1)


if __name__ == "__main__":
    main()
