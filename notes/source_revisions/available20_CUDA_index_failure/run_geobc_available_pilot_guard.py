"""One-shot external wall guard for the frozen independent CAD20 pilot.

Default invocation does not read the plan or start a child.  Actual execution
requires a separate genuine root release and both upstream process exits.
The campaign owns exactly one preparation child followed by one evaluator;
it never replays either stage or resets an existing clock.
"""
import time

ENTRY_EPOCH = time.time()
ENTRY_MONOTONIC = time.monotonic()

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OWN = "tools/run_geobc_available_pilot_guard.py"
PLAN = "notes/geobc_available_cad_pilot_plan.json"
SCIENCE = "c7c994a44d462b94426ecfa034f2d535bbf7751331c1e16ab8925485e5a67770"
KIND = "GeoBC_available20_one_shot_guard"
STAGE_SECONDS = 3600
CAMPAIGN_SECONDS = 7200


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def owned(value):
    filename = Path(value)
    filename = (filename if filename.is_absolute() else ROOT / filename).resolve()
    require(ROOT.resolve() in filename.parents, "Guard file must remain in the project")
    return filename


def read(filename):
    return json.loads(owned(filename).read_text(encoding="utf-8-sig"))


def write(filename, record):
    filename = owned(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def sha(filename):
    result = hashlib.sha256()
    with owned(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def terminate(child):
    """Return a truthful stop outcome even if kill/reap cannot finish."""
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


def remaining_budget(deadline_epoch, deadline_monotonic):
    return min(deadline_epoch - time.time(), deadline_monotonic - time.monotonic())


def wait_dual_deadline(child, deadline_epoch, deadline_monotonic):
    """A clock-axis divergence cannot hide inside a long monotonic wait."""
    while True:
        remaining = remaining_budget(deadline_epoch, deadline_monotonic)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(child.args, 0)
        try:
            return child.wait(timeout=min(1., remaining))
        except subprocess.TimeoutExpired:
            pass


def elapsed(record, entry_epoch, entry_monotonic):
    ended_epoch = time.time(); ended_monotonic = time.monotonic()
    mono = ended_monotonic - entry_monotonic
    epoch = ended_epoch - entry_epoch
    record.update(ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat(),
                  ended_epoch=ended_epoch, ended_monotonic=ended_monotonic,
                  elapsed_seconds=mono, elapsed_epoch_seconds=epoch,
                  clock_axis_difference_seconds=epoch - mono)
    if "stages" in record:
        record["between_stage_and_guard_overhead_seconds"] = max(
            0., mono - sum(item.get("elapsed_seconds", 0.) for item in record["stages"].values()))
        record["between_stage_and_guard_overhead_epoch_seconds"] = (
            epoch - sum(item.get("elapsed_epoch_seconds", 0.) for item in record["stages"].values()))


def within_budget(record, seconds, deadline_epoch, deadline_monotonic):
    return (0 <= record["elapsed_seconds"] <= seconds
            and 0 <= record["elapsed_epoch_seconds"] <= seconds
            and record["ended_epoch"] <= deadline_epoch
            and record["ended_monotonic"] <= deadline_monotonic)


def run(plan_path=PLAN, start=False, resume_prelaunch=None):
    if not start:
        return {"status": "source_only", "actual_execution": False,
                "plan_reads": 0, "raw_reads": 0, "GT_reads": 0, "child_launches": 0,
                "writes": 0}
    require(os.name != "nt" and Path("/proc").is_dir(), "Use the project's WSL namespace")
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from geobc.available_cad_pilot_data import (
        exact_process_exited, predecessor_gate, process_snapshot, release, request_contract,
    )
    contract, plan = request_contract(plan_path)
    require(contract["scientific_digest"] == plan["scientific_digest"] == SCIENCE,
            "Frozen pilot science identity differs")
    for purpose in ("preparation", "evaluation"):
        release(contract, plan, purpose)
    require(contract["source_sha256"].get(OWN) == sha(OWN), "Guard source is not root-pinned")
    proof = predecessor_gate(plan)
    recovery = None
    if resume_prelaunch:
        # Explicit root-reviewed continuation is limited to the observed failure
        # before any native clock, data, array, model or GT access.  Original UTC
        # deadlines remain authoritative, including repair/downtime costs.
        global ENTRY_EPOCH
        recovery = read(resume_prelaunch)
        require(recovery.get("passed") is True and recovery.get("actual_continuation_released") is True
                and recovery.get("scientific_digest") == SCIENCE
                and recovery.get("source_pins") == contract["source_sha256"]
                and recovery.get("zero_raw_array_model_GT_reads") is True,
                "Prelaunch continuation lacks the exact root technical release")
        prior = recovery["prior_campaign_identity"]
        require(sha(prior["path"]) == prior["sha256"], "Original campaign archive identity changed")
        original = read(prior["path"])
        require(original.get("closed") is True and original.get("active") is False
                and original.get("status") == "terminal" and len(original.get("stages", {})) == 1
                and original["stages"]["preparation"]["exit_code"] == 1
                and original["stages"]["preparation"]["actual_child_exited"] is True
                and exact_process_exited(original["guard_process"])
                and exact_process_exited(original["stages"]["preparation"]["child_process"]),
                "Prior failure does not prove the exact guard and preparation child exited")
        for item in recovery["archived_files"]:
            require(sha(item["path"]) == item["sha256"], "Prelaunch failure archive changed")
        require(recovery["original_deadline_epoch"] == original["deadline_epoch"]
                == original["entry_epoch"] + CAMPAIGN_SECONDS,
                "Original absolute campaign deadline differs")
        ENTRY_EPOCH = original["entry_epoch"]
        require(time.time() < ENTRY_EPOCH + STAGE_SECONDS,
                "Original preparation deadline exhausted during repair")
    require(remaining_budget(ENTRY_EPOCH + STAGE_SECONDS, ENTRY_MONOTONIC + STAGE_SECONDS) > 0,
            "Preparation budget exhausted before launch")
    outputs = plan["scientific"]["outputs"]
    preparation = owned(outputs["input_cache"])
    evaluation = owned(outputs["readout"])
    launchfile = evaluation / "campaign_external_launch.json"
    terminalfile = evaluation / "campaign_external_terminal.json"
    for filename in (launchfile, terminalfile,
                     preparation / "clock.json", preparation / "manifest.json",
                     preparation / "preparation_external_terminal.json",
                     evaluation / "clock.json", evaluation / "summary.json",
                     evaluation / "evaluation_external_terminal.json"):
        require(not filename.exists(), "Prior pilot attempt exists; no automatic replay or budget reset")
    own_process = process_snapshot(os.getpid())
    require(own_process is not None, "Actual guard process identity unavailable")
    campaign = {"schema": 1, "kind": KIND, "scientific_digest": SCIENCE,
        "status": "starting", "actual_execution": True, "attempts": 2 if recovery else 1,
        "automatic_replay": False, "source_sha256": contract["source_sha256"],
        "guard_process": own_process,
        "started_at_utc": datetime.fromtimestamp(ENTRY_EPOCH, timezone.utc).isoformat(),
        "entry_epoch": ENTRY_EPOCH, "entry_monotonic": ENTRY_MONOTONIC,
        "deadline_epoch": ENTRY_EPOCH + CAMPAIGN_SECONDS,
        "deadline_monotonic": ENTRY_MONOTONIC + CAMPAIGN_SECONDS,
        "budget_seconds": CAMPAIGN_SECONDS, "predecessor_proof": proof, "stages": {}}
    if recovery:
        campaign["prelaunch_technical_continuation"] = {
            "root_review": str(resume_prelaunch), "root_review_sha256": sha(resume_prelaunch),
            "prior_campaign_identity": recovery["prior_campaign_identity"],
            "original_entry_monotonic": original["entry_monotonic"],
            "original_deadline_monotonic": original["deadline_monotonic"],
            "current_monotonic_is_new_measured_axis": True,
            "UTC_span_includes_original_failure_repair_and_downtime": True,
            "budget_reset": False, "data_or_model_replay": False}
    evaluation.mkdir(parents=True, exist_ok=True)
    with launchfile.open("x", encoding="utf-8") as stream:
        json.dump(campaign, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    def stage(purpose, output, module, stage_epoch, stage_monotonic):
        guard = {"schema": 1, "kind": KIND, "purpose": purpose,
            "scientific_digest": SCIENCE, "guard_source": OWN,
            "guard_source_sha256": contract["source_sha256"][OWN],
            "guard_process": own_process, "budget_seconds": STAGE_SECONDS,
            "automatic_replay": False, "entry_epoch": stage_epoch,
            "entry_monotonic": stage_monotonic,
            "deadline_epoch": stage_epoch + STAGE_SECONDS,
            "deadline_monotonic": stage_monotonic + STAGE_SECONDS,
            "campaign_entry_epoch": ENTRY_EPOCH,
            "campaign_entry_monotonic": ENTRY_MONOTONIC,
            "campaign_deadline_epoch": campaign["deadline_epoch"],
            "campaign_deadline_monotonic": campaign["deadline_monotonic"]}
        deadline_epoch = min(guard["deadline_epoch"], campaign["deadline_epoch"])
        deadline_monotonic = min(guard["deadline_monotonic"], campaign["deadline_monotonic"])
        require(remaining_budget(deadline_epoch, deadline_monotonic) > 0,
                "External stage/campaign budget exhausted")
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.pop("GEOBC_AVAILABLE20_PREP_GUARD", None)
        env.pop("GEOBC_AVAILABLE20_EVAL_GUARD", None)
        key = "GEOBC_AVAILABLE20_" + ("PREP" if purpose == "preparation" else "EVAL") + "_GUARD"
        env[key] = json.dumps(guard, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        argv = [sys.executable, "-u", "-m", module, "--plan", str(plan_path), "--start"]
        record = {"schema": 1, "kind": KIND, "purpose": purpose,
            "scientific_digest": SCIENCE, "guard": guard, "status": "starting",
            "started_at_utc": datetime.fromtimestamp(stage_epoch, timezone.utc).isoformat(),
            "automatic_replay": False,
            "external_timeout": False, "child_argv": argv}
        launch = output / (purpose + "_external_launch.json")
        terminal = output / (purpose + "_external_terminal.json")
        require(not launch.exists() and not terminal.exists(), "Existing stage receipt cannot be replaced")
        with launch.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        child = None
        code = None
        interrupted = False
        try:
            with (output / (purpose + "_process.log")).open("x", encoding="utf-8") as log:
                if remaining_budget(deadline_epoch, deadline_monotonic) <= 0:
                    record["external_timeout"] = True
                    raise subprocess.TimeoutExpired(argv, 0)
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                child_process = process_snapshot(child.pid)
                require(child_process is not None and child_process["ppid"] == own_process["pid"]
                        and child_process["argv"] == argv, "Actual child identity differs")
                record.update(status="running", child_process=child_process)
                write(launch, record)
                campaign["status"] = "running"
                campaign["current_stage"] = purpose
                campaign["stages"][purpose] = record
                write(launchfile, campaign)
                try:
                    code = wait_dual_deadline(child, deadline_epoch, deadline_monotonic)
                except subprocess.TimeoutExpired:
                    record["external_timeout"] = True
                    code, stop_error = terminate(child)
                    if stop_error:
                        record["termination_error_class"] = stop_error
        except BaseException as error:
            interrupted = True
            record["exception_class"] = type(error).__name__
            if child is not None:
                code, stop_error = terminate(child)
                if stop_error:
                    record["termination_error_class"] = stop_error
            raise
        finally:
            if child is not None and child.poll() is None:
                code, stop_error = terminate(child)
                if stop_error:
                    record["termination_error_class"] = stop_error
            record.update(status="terminal", exit_code=code, interrupted=interrupted)
            elapsed(record, stage_epoch, stage_monotonic)
            record["actual_child_exited"] = (
                "child_process" in record and exact_process_exited(record["child_process"]))
            record["within_external_budget"] = (
                not record["external_timeout"] and not interrupted
                and record["actual_child_exited"] and code is not None
                and "termination_error_class" not in record
                and within_budget(record, STAGE_SECONDS, deadline_epoch, deadline_monotonic))
            write(terminal, record)
            # Include actual terminal persistence in the authoritative elapsed time.
            elapsed(record, stage_epoch, stage_monotonic)
            record["within_external_budget"] = (
                record["within_external_budget"]
                and within_budget(record, STAGE_SECONDS, deadline_epoch, deadline_monotonic))
            write(terminal, record)
            campaign["stages"][purpose] = record
            write(launchfile, campaign)
            elapsed(record, stage_epoch, stage_monotonic)
            record["within_external_budget"] = (
                record["within_external_budget"]
                and within_budget(record, STAGE_SECONDS, deadline_epoch, deadline_monotonic))
            write(terminal, record)
            # The acceptance check follows all normal receipt writes.  A final
            # boundary overrun stays negative even if its failure receipt is slow.
            if remaining_budget(deadline_epoch, deadline_monotonic) < 0:
                elapsed(record, stage_epoch, stage_monotonic)
                record["within_external_budget"] = False
                write(terminal, record)
        return record

    success = False
    try:
        first = stage("preparation", preparation, "geobc.available_cad_pilot_data",
                      ENTRY_EPOCH, ENTRY_MONOTONIC)
        require(first["exit_code"] == 0 and first["within_external_budget"]
                and first["actual_child_exited"], "Preparation did not close normally; root review required")
        manifest = read(preparation / "manifest.json")
        clock = read(preparation / "clock.json")
        require(manifest.get("status") == "completed" and manifest.get("terminal") is True
                and clock.get("closed") is True and clock.get("active") is False
                and manifest.get("run_process") == clock.get("run_process") == first["child_process"],
                "Preparation native terminal/process proof differs")
        # An authentic input-negative is deliberately delivered to the evaluator:
        # it produces the fixed failure inventory and reads zero numeric targets.
        second = stage("evaluation", evaluation, "geobc.available_cad_pilot_eval",
                       time.time(), time.monotonic())
        require(second["exit_code"] == 0 and second["within_external_budget"]
                and second["actual_child_exited"], "Evaluation did not close normally; root review required")
        summary = read(evaluation / "summary.json")
        clock = read(evaluation / "clock.json")
        require(summary.get("status") in ("completed", "completed_negative")
                and summary.get("terminal") is True and clock.get("closed") is True
                and clock.get("active") is False
                and summary.get("run_process") == clock.get("run_process") == second["child_process"],
                "Evaluation native terminal/process proof differs")
        campaign["scientific_outcome"] = summary.get("status")
        success = True
    except BaseException as error:
        campaign["exception_class"] = type(error).__name__
        raise
    finally:
        campaign.update(status="terminal", all_requested_stages_exited_normally=success,
                        closed=True, active=False)
        elapsed(campaign, ENTRY_EPOCH, ENTRY_MONOTONIC)
        campaign["within_external_budget"] = (
            success and within_budget(campaign, CAMPAIGN_SECONDS,
                                      campaign["deadline_epoch"], campaign["deadline_monotonic"])
            and all(item.get("within_external_budget") is True for item in campaign["stages"].values()))
        write(terminalfile, campaign)
        elapsed(campaign, ENTRY_EPOCH, ENTRY_MONOTONIC)
        campaign["within_external_budget"] = (
            campaign["within_external_budget"] and within_budget(
                campaign, CAMPAIGN_SECONDS, campaign["deadline_epoch"], campaign["deadline_monotonic"]))
        write(terminalfile, campaign)
        write(launchfile, campaign)
        if remaining_budget(campaign["deadline_epoch"], campaign["deadline_monotonic"]) < 0:
            elapsed(campaign, ENTRY_EPOCH, ENTRY_MONOTONIC)
            campaign["within_external_budget"] = False
            write(terminalfile, campaign)
            write(launchfile, campaign)
    return {"status": "terminal", "actual_execution": True,
            "within_external_budget": campaign["within_external_budget"],
            "scientific_outcome": campaign.get("scientific_outcome"),
            "elapsed_seconds": campaign["elapsed_seconds"],
            "elapsed_epoch_seconds": campaign["elapsed_epoch_seconds"],
            "clock_axis_difference_seconds": campaign["clock_axis_difference_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--resume-prelaunch", default=None)
    args = parser.parse_args()
    result = run(args.plan, args.start, args.resume_prelaunch)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if not args.start or result.get("within_external_budget") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
