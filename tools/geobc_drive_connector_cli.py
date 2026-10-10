"""Compact metadata CLI for the Codex connector session; never accepts a URI.

Large inventories/final summaries stay in owned project JSON files rather than
being truncated by a terminal tool output limit. Scientific source pins stay fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tools import prepare_geobc_drive_recovery as recovery

PENDING = "data/raw_geobc_coverage/.connector_recovery/session_pending.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=recovery.PLAN)
    parser.add_argument("--action", choices=("prepare", "pending", "pending_page", "reserve", "fail", "finalize"), required=True)
    args = parser.parse_args()
    payload = json.load(sys.stdin) if args.action in ("reserve", "fail", "pending_page") else None
    if args.action == "pending_page":
        stored = recovery.read(PENDING)
        offset, limit = payload["offset"], payload.get("limit", 16)
        recovery.require(type(offset) is int and offset >= 0 and type(limit) is int and 1 <= limit <= 16,
                         "Invalid metadata page")
        result = {"status": "pending_page", "pending": stored["pending"][offset:offset + limit],
                  "pending_count": len(stored["pending"])}
    elif args.action == "pending":
        result = recovery.pending(args.plan)
        recovery.write(PENDING, {"status": result["status"], "pending": result["pending"],
                                  "created_at_utc": recovery.now()})
        result = {"status": result["status"], "pending_count": len(result["pending"]), "inventory_path": PENDING}
    elif args.action == "prepare":
        value = recovery.prepare(args.plan)
        result = {"status": value["status"], "contract_sha256": value["contract_sha256"],
                  "clock": value["clock"]}
    elif args.action == "reserve":
        value = recovery.reserve(args.plan, payload)
        result = {key: value[key] for key in ("status", "relative_path", "reservation_id", "attempt_id") if key in value}
    elif args.action == "fail":
        try:
            value = recovery.record_fetch_failure(args.plan, payload)
            result = {"status": value["status"], "reservation_id": value["attempt_id"]}
        except ValueError as error:
            if str(error) != "Attempt already terminal":
                raise
            contract, marker, directory = recovery.context(args.plan)
            with recovery.LedgerLock(directory):
                value = recovery.load_attempt(directory, payload["reservation_id"])
                recovery.require(value.get("open_attempt") is False and value.get("status") in ("completed", "failed"),
                                 "Reservation terminal identity unresolved")
                result = {"status": value["status"], "reservation_id": value["attempt_id"], "already_terminal": True}
    else:
        value = recovery.finalize(args.plan)
        result = {"status": value["status"], "completed_files": value["completed_files"],
                  "total_files": value["total_files"], "missing_file_count": len(value["missing_files"]),
                  "completion_gate": value["completion_gate"], "costs": value["costs"]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
