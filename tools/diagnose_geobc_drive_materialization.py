"""Run the frozen materializer with numeric-only transport error diagnostics.

The reference stays in the original stdin/memory path. This wrapper changes
neither materialization, source identity nor acquisition ledger accounting.
"""
import json
import sys
import urllib.error

from tools import materialize_geobc_coverage_drive_ref as materializer


def main():
    original = materializer.error_code
    seen = set()

    def observed_error_code(error):
        if isinstance(error, urllib.error.URLError) and id(error) not in seen:
            seen.add(id(error))
            reason = getattr(error, "reason", None)
            safe = {"phase": "transport_error_diagnostic",
                    "reason_type": type(reason).__name__}
            for name in ("errno", "winerror", "verify_code"):
                value = getattr(reason, name, None)
                if type(value) is int:
                    safe[name] = value
            print(json.dumps(safe, allow_nan=False), file=sys.stderr, flush=True)
        return original(error)

    materializer.error_code = observed_error_code
    try:
        return materializer.main()
    finally:
        materializer.error_code = original


if __name__ == "__main__":
    raise SystemExit(main())
