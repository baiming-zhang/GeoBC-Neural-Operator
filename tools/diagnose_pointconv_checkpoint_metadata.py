"""One CPU-only diagnosis of the failed endpoint metadata comparison."""
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
run = root / "runs/geobc_exploration/pointconv_seed2026_480"
saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
record = json.loads((run / "config.json").read_text(encoding="utf8"))
differences = []

def compare(left, right, path="config"):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append({"path": path, "reason": "different keys"})
        for key in set(left) & set(right):
            compare(left[key], right[key], path + "." + key)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right):
            differences.append({"path": path, "reason": "sequence type",
                                "checkpoint_type": type(left).__name__, "JSON_type": type(right).__name__})
        if len(left) != len(right):
            differences.append({"path": path, "reason": "different length"})
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, path + f"[{index}]")
    elif left != right:
        differences.append({"path": path, "reason": "different value"})

compare(saved["config"], record)
receipt = {"checkpoint_step": saved["step"], "differences": differences,
           "canonical_JSON_equal": json.dumps(saved["config"], sort_keys=True, allow_nan=False)
               == json.dumps(record, sort_keys=True, allow_nan=False),
           "scope": "CPU checkpoint metadata only; no tensor values printed, model instantiated, labels or scientific arrays accessed, or CUDA initialized"}
(root / "notes/pointconv_checkpoint_metadata_diagnosis.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
print(json.dumps(receipt))
