"""One explicit checkpoint recovery with the original PP campaign deadline.

Default invocation has zero file/process/GPU IO. Actual --start requires a
separate root release and never modifies the original launch or its archive.
The interrupted whole-campaign monotonic elapsed is unknown, not fabricated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OWN = "tools/run_geobc_pp_recovery_guard.py"
PLAN = "notes/geobc_transolver_pp_regime_plan.json"
REVIEW = "notes/pp_checkpoint_resume_root_review.json"
RUN = "runs/geobc_coverage_comparison/PP_REGIME_CONFIRMATION"
ATTEMPT = "1791622838087995693"
ARCHIVE = f"{RUN}/recovery/attempt_{ATTEMPT}"
SCI = "dad6be58506fd5079470e8d31d231ff8cf52234c23cece4007740f812164dc89"
DRIVER = "geobc/transolver_pp_regime_confirmation.py"
DRIVER_SHA = "7b35c294495727ef8b117841046c3d31fda980147769a44ec00f06346cb2956e"
ORIGINAL_GUARD = "tools/run_geobc_pp_regime_guard.py"
ORIGINAL_GUARD_SHA = "dce73a03ac4ced3d68be7a22ba1e5f1dbd9893d018808631dcad67db87cdfea6"
BUDGET = 30000


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def local(value):
    target = Path(value)
    target = (target if target.is_absolute() else ROOT / target).resolve()
    require(ROOT in target.parents, "Recovery path must stay in this project")
    return target


def read(value):
    return json.loads(local(value).read_text(encoding="utf-8-sig"))


def sha(value):
    h = hashlib.sha256()
    with local(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(value, record):
    filename = local(value)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, filename)


def process(pid):
    directory = Path(f"/proc/{pid}")
    try:
        raw = (directory / "stat").read_text(encoding="utf-8")
        tail = raw[raw.rindex(")") + 2:].split()
        argv = (directory / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    return {"pid": int(pid), "ppid": int(tail[1]), "start_ticks": int(tail[19]),
            "state": tail[0], "argv": [x.decode("utf-8", errors="replace") for x in argv if x]}


def exact_exit(identity):
    require(type(identity.get("pid")) is int and type(identity.get("start_ticks")) is int,
            "Exact historical Linux process identity is required")
    current = process(identity["pid"])
    require(current is None or current["start_ticks"] != identity["start_ticks"]
            or current["state"] in ("Z", "X"), "Historical PP child/guard is still active")
    return {"pid": identity["pid"], "start_ticks": identity["start_ticks"],
            "exact_process_exited": True, "observed_at_utc": utc(),
            "observation": "PID absent" if current is None else
                           "PID reused" if current["start_ticks"] != identity["start_ticks"] else
                           "exited process state"}


def prepare():
    """Metadata/source gates only; never import/load a model, array or PT."""
    require(os.name != "nt" and Path("/proc").is_dir(), "Use the project's common WSL namespace")
    review, plan = read(REVIEW), read(PLAN)
    require(review.get("passed") is True and review.get("recovery_released") is True
            and review.get("scientific_digest") == SCI
            and review.get("guard_source_sha256") == sha(OWN)
            and review.get("source_sha256") == DRIVER_SHA
            and review.get("original_attempt_id") == ATTEMPT,
            "Authentic separately reviewed checkpoint recovery release is required")
    science = plan["scientific"]
    science_sha = hashlib.sha256(json.dumps(science, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(plan.get("scientific_digest") == science_sha == SCI
            and science["budget"] == {"training_seconds": 28800, "nontraining_seconds": 1200,
                                      "whole_wall_seconds": BUDGET, "peak_GPU_GiB": 24}
            and local(plan["output"]) == local(RUN)
            and sha(DRIVER) == DRIVER_SHA and sha(ORIGINAL_GUARD) == ORIGINAL_GUARD_SHA,
            "Original PP scientific recipe, output or driver/guard changed")
    require(local(review["archive_path"]) == local(ARCHIVE), "Reviewed immutable archive path differs")
    inspection_id = review["inspection_identity"]
    require(local(inspection_id["path"]) == local(f"{ARCHIVE}/inspection.json")
            and sha(inspection_id["path"]) == inspection_id["sha256"], "Root checkpoint inspection identity differs")
    inspection = read(inspection_id["path"])
    require(inspection.get("original_attempt_id") == ATTEMPT
            and inspection.get("absence_epoch_is_upper_bound_not_exact_exit") is True
            and inspection.get("original_external_guard_terminal_missing") is True,
            "Actual interrupted attempt inspection is required")
    history = inspection["immutable_history"]
    for name in ("status.json", "clock.json", "execution.json", "external_launch.json",
                 "costs.json", "read_receipt.json"):
        require(sha(f"{ARCHIVE}/{name}") == history[name]["sha256"], "Archived original metadata changed")
    launch = read(f"{ARCHIVE}/external_launch.json")
    launch_sha = sha(f"{ARCHIVE}/external_launch.json")
    require(launch_sha == review["original_launch_sha256"] == sha(f"{RUN}/external_launch.json")
            and launch.get("status") == "running" and launch.get("guard_source_sha256") == ORIGINAL_GUARD_SHA
            and launch.get("external_budget_seconds") == BUDGET and launch.get("automatic_replay") is False,
            "Original unfinished external launch must remain unchanged")
    clock = read(f"{RUN}/clock.json")
    require(sha(f"{RUN}/clock.json") == history["clock.json"]["sha256"]
            and clock.get("attempt_id") == ATTEMPT and clock.get("active") is True
            and clock.get("closed") is False and clock.get("phase") == "training",
            "Only the genuine still-open original checkpoint clock can be resumed")
    execution = read(f"{ARCHIVE}/execution.json")
    require(execution.get("attempt_id") == ATTEMPT
            and execution["run_process"]["pid"] == launch.get("child_pid") == clock.get("pid"),
            "Archived native/external attempt identities disagree")
    exit_proofs = inspection["exact_process_exit_proofs"]
    require(execution["run_process"]["pid"] == exit_proofs["child"]["pid"]
            and execution["run_process"]["start_ticks"] == exit_proofs["child"]["start_ticks"],
            "Inspection process identity differs")
    process_proofs = {name: exact_exit(exit_proofs[name]) for name in ("child", "guard")}
    observed = read(f"{RUN}/observed_attempt_exit.json")
    absence_epoch = datetime.fromisoformat(inspection["first_observed_absence_UTC"]).timestamp()
    require(observed.get("attempt_id") == ATTEMPT and finite(observed.get("exit_epoch"))
            and observed["exit_epoch"] == absence_epoch,
            "Root actual first-observed absence upper bound is required; no invented exact exit")
    require(finite(clock.get("last_epoch")) and finite(clock.get("total_seconds"))
            and finite(clock.get("training_seconds")) and finite(clock.get("nontraining_seconds"))
            and clock["last_epoch"] <= absence_epoch <= time.time(), "Original clock/observed epoch is invalid")
    require(math.isclose(clock["training_seconds"] + clock["nontraining_seconds"],
                         clock["total_seconds"], rel_tol=1e-10, abs_tol=1e-6), "Original clock ledger differs")
    gap = max(0., absence_epoch - clock["last_epoch"])
    charged_total = clock["total_seconds"] + gap
    charged_training = clock["training_seconds"] + gap
    require(charged_total < BUDGET and charged_training < 28800 and clock["nontraining_seconds"] < 1200,
            "Original cumulative ledger is already beyond a frozen budget")
    checkpoint = inspection["checkpoint_metadata"]
    require(checkpoint["step"] == checkpoint["next_case_position"] == checkpoint["scheduler_last_epoch"]
            == checkpoint["case_exposure_sum"] == 23161 and checkpoint["case_count"] == 120
            and checkpoint["rng_sections"] == ["python", "numpy", "torch", "cuda"]
            and checkpoint["optimizer_state_entries"] == 371,
            "Root inspected full Adam/scheduler/RNG/cursor checkpoint differs")
    checkpoint_id = checkpoint["checkpoint"]
    require(local(checkpoint_id["path"]) == local(f"{RUN}/last.pt"), "Recovery must use its original last checkpoint")
    checkpoint_stat = local(checkpoint_id["path"]).stat()
    require(checkpoint_stat.st_size == checkpoint_id["bytes"] and checkpoint_stat.st_mtime_ns == checkpoint_id["mtime_ns"],
            "Own checkpoint stat identity changed after root CPU inspection")
    prefix = review["loss_prefix_identity"]
    require(local(prefix["path"]) == local(f"{RUN}/loss.jsonl")
            and prefix.get("record_count") == 23161 and sha(prefix["path"]) == prefix["sha256"],
            "Root-preserved checkpoint-committed scalar loss prefix differs")
    started_epoch = datetime.fromisoformat(launch["started_at_utc"]).timestamp()
    absolute_deadline = started_epoch + BUDGET
    require(time.time() < absolute_deadline, "Original absolute campaign deadline has expired")
    require(not local(f"{RUN}/summary.json").exists(), "A genuine native terminal summary already exists")
    return {"review": review, "inspection": inspection, "original_launch": launch,
            "original_launch_sha256": launch_sha, "process_exit_proofs": process_proofs,
            "original_start_epoch": started_epoch, "absolute_deadline_epoch": absolute_deadline,
            "unknown_interruption_charge_to_add_seconds": gap,
            "native_cumulative_total_with_upper_bound_gap_seconds": charged_total,
            "remaining_monotonic_budget_seconds": BUDGET - charged_total}


def run(start=False):
    if not start:
        return {"status": "source_only_ready", "writes": False, "actual_execution": False}
    prepared = prepare()
    output = local(RUN)
    started_mono, started_epoch = time.monotonic(), time.time()
    guard = process(os.getpid())
    require(guard is not None, "Cannot prove actual recovery guard process")
    argv = [sys.executable, "-u", "-m", "geobc.transolver_pp_regime_confirmation", "--plan", PLAN, "--start"]
    record = {"schema": 1, "kind": "one_shot_PP_original_checkpoint_recovery_guard",
              "status": "starting", "scientific_digest": SCI, "automatic_replay": False,
              "guard_source_sha256": sha(OWN), "original_guard_source_sha256": ORIGINAL_GUARD_SHA,
              "original_driver_source_sha256": DRIVER_SHA, "original_attempt_id": ATTEMPT,
              "original_launch_path": f"{RUN}/external_launch.json",
              "original_launch_sha256": prepared["original_launch_sha256"],
              "archive_path": ARCHIVE, "inspection_identity": prepared["review"]["inspection_identity"],
              "root_review_identity": {"path": REVIEW, "sha256": sha(REVIEW)},
              "checkpoint_identity_from_root_CPU_inspection": prepared["inspection"]["checkpoint_metadata"]["checkpoint"],
              "resume_checkpoint_step": prepared["inspection"]["checkpoint_metadata"]["step"],
              "original_status_step": prepared["inspection"]["original_status_step"],
              "original_loss_rows": prepared["inspection"]["original_loss_rows"],
              "rollback_uncommitted_updates": prepared["inspection"]["rollback_uncommitted_updates"],
              "exact_whole_campaign_computation_counts": False,
              "loss_prefix_identity": prepared["review"]["loss_prefix_identity"],
              "historical_process_exit_proofs": prepared["process_exit_proofs"],
              "started_at_utc": prepared["original_launch"]["started_at_utc"],
              "recovery_started_at_utc": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
              "external_budget_seconds": BUDGET,
              "original_start_epoch": prepared["original_start_epoch"],
              "absolute_deadline_epoch": prepared["absolute_deadline_epoch"],
              "recovery_remaining_monotonic_budget_seconds": prepared["remaining_monotonic_budget_seconds"],
              "native_cumulative_total_with_upper_bound_gap_seconds": prepared["native_cumulative_total_with_upper_bound_gap_seconds"],
              "unknown_interruption_charge_to_add_seconds": prepared["unknown_interruption_charge_to_add_seconds"],
              "original_guard_terminal_missing": True, "original_guard_elapsed_monotonic_seconds": None,
              "elapsed_launch_to_exit_seconds": None, "whole_campaign_monotonic_elapsed_seconds": None,
              "whole_campaign_elapsed_accounting_exact": False, "in_flight_costs_exact": False,
              "guard_process": guard, "child_argv": argv,
              "whole_wall_budget_not_reset": True, "PT_loads_by_guard": 0, "GPU_operations_by_guard": 0}
    record_file = output / "recovery_external_launch.json"
    with record_file.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    child = None; code = None; timed_out = False; failure = None
    try:
        with (output / "recovery_process.log").open("x", encoding="utf-8") as log:
            require(time.time() < prepared["absolute_deadline_epoch"]
                    and time.monotonic() - started_mono < prepared["remaining_monotonic_budget_seconds"],
                    "Original campaign deadline expired before child launch")
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            child_identity = process(child.pid)
            record.update(child_pid=child.pid, child_process=child_identity)
            require(child_identity is not None and child_identity["ppid"] == os.getpid(),
                    "Cannot prove actual recovery child identity")
            record.update(status="running", child_pid=child.pid, child_process=child_identity)
            write(record_file, record)
            while child.poll() is None:
                if (time.time() >= prepared["absolute_deadline_epoch"] or
                        time.monotonic() - started_mono >= prepared["remaining_monotonic_budget_seconds"]):
                    timed_out = True
                    break
                try:
                    child.wait(timeout=min(1., max(0.001, prepared["absolute_deadline_epoch"] - time.time()),
                                           max(0.001, prepared["remaining_monotonic_budget_seconds"] - (time.monotonic() - started_mono))))
                except subprocess.TimeoutExpired:
                    pass
            if timed_out:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
            code = child.poll()
    except Exception as error:
        failure = type(error).__name__
        if child is not None and child.poll() is None:
            try:
                child.terminate(); child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=5)
            except Exception as cleanup_error:
                record["cleanup_error_type"] = type(cleanup_error).__name__
        code = None if child is None else child.poll()
    finally:
        ended_epoch, ended_mono = time.time(), time.monotonic()
        recovery_mono = ended_mono - started_mono
        recovery_utc = ended_epoch - started_epoch
        whole_utc = ended_epoch - prepared["original_start_epoch"]
        native_clock = read(f"{RUN}/clock.json")
        native_status = read(f"{RUN}/status.json")
        native_execution = read(f"{RUN}/execution.json")
        actual_exit = child is not None and code is not None
        record.update(status="terminal" if actual_exit else "failed_unclosed",
                      exit_code=code, external_timeout=timed_out, error_type=failure,
                      ended_at_utc=datetime.fromtimestamp(ended_epoch, timezone.utc).isoformat(),
                      recovery_elapsed_monotonic_seconds=recovery_mono,
                      recovery_elapsed_UTC_seconds=recovery_utc,
                      elapsed_launch_to_exit_UTC_seconds=whole_utc,
                      whole_campaign_elapsed_UTC_seconds=whole_utc,
                      recovery_UTC_minus_monotonic_seconds=recovery_utc - recovery_mono,
                      within_external_budget=bool(actual_exit and not timed_out and failure is None
                          and 0 <= whole_utc <= BUDGET
                          and 0 <= recovery_mono <= prepared["remaining_monotonic_budget_seconds"]),
                      actual_child_exited=actual_exit,
                      summary_present=(output / "summary.json").is_file(),
                      native_attempt_id=native_execution.get("attempt_id"),
                      native_run_process=native_execution.get("run_process"),
                      native_status_at_guard_exit=native_status.get("status"),
                      native_clock_closed_at_guard_exit=native_clock.get("closed") is True
                          and native_clock.get("active") is False,
                      native_unknown_interruption_charge_seconds=native_clock.get("unknown_interruption_charge_seconds"))
        if actual_exit and isinstance(record.get("child_process"), dict):
            record["child_exact_exit_proof"] = exact_exit(record["child_process"])
        write(record_file, record)
        # Receipt persistence belongs to the original campaign budget too.
        final_epoch, final_mono = time.time(), time.monotonic()
        record.update(ended_at_utc=datetime.fromtimestamp(final_epoch, timezone.utc).isoformat(),
                      recovery_elapsed_monotonic_seconds=final_mono - started_mono,
                      recovery_elapsed_UTC_seconds=final_epoch - started_epoch,
                      elapsed_launch_to_exit_UTC_seconds=final_epoch - prepared["original_start_epoch"],
                      whole_campaign_elapsed_UTC_seconds=final_epoch - prepared["original_start_epoch"],
                      recovery_UTC_minus_monotonic_seconds=(final_epoch - started_epoch) - (final_mono - started_mono))
        record["within_external_budget"] &= (final_epoch <= prepared["absolute_deadline_epoch"]
            and final_mono - started_mono <= prepared["remaining_monotonic_budget_seconds"])
        write(record_file, record)
    print(json.dumps({k: record.get(k) for k in ("status", "exit_code", "external_timeout",
          "within_external_budget", "actual_child_exited", "native_clock_closed_at_guard_exit")}), flush=True)
    return 0 if code == 0 and not timed_out and failure is None and record["within_external_budget"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    result = run(args.start)
    if args.start:
        raise SystemExit(result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
