"""One metadata-only card inventory after the actual FORMAT reader failure."""
from collections import Counter
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
deck = ROOT / "data/raw/SolverDeck/101_479.fem"
began = time.monotonic()
counts, controls = Counter(), []
bulk = False
with deck.open(encoding="utf-8") as stream:
    for lineno, raw in enumerate(stream, 1):
        line = raw.split("$", 1)[0].rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            continue
        if not bulk:
            if stripped.upper() == "BEGIN BULK":
                bulk = True
                continue
            if stripped.startswith(("FORMAT", "DEBUG", "METHOD", "ELFORCE", "ESE")):
                controls.append({"line": lineno, "entry": stripped})
            continue
        first = line.split(",", 1)[0].strip() if "," in line else line[:8].strip()
        if not first or first.startswith(("+", "*")):
            continue
        counts[first] += 1
        if first in {"PARAM", "GAPPRM", "EIGRL"}:
            controls.append({"line": lineno, "entry": stripped})
info = deck.stat()
probe = Path("/proc/420/cmdline")
record = {"scope": "Direct recovery diagnosis only: card names/counts and controls; no GRID/element array values interpreted or printed",
          "source": str(deck), "source_stat": {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns},
          "card_counts": dict(counts), "controls": controls,
          "elapsed_seconds": time.monotonic()-began,
          "raw_deck_read": True, "scientific_array_values_interpreted": 0, "labels_read": 0, "GPU_calls": 0,
          "old_coordinator_live": b"run_geobc_exploration.py" in probe.read_bytes() if probe.exists() else False}
(ROOT / "notes/elastic_deck_control_diagnosis.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
print(json.dumps(record))
