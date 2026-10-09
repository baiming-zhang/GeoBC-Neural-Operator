"""Release the matched queue after one metadata-only interface acceptance."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_gradient_teacher_pair import differences


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    helper = read(ROOT / "notes/gradient_supervision_ready.json")
    trainer = read(ROOT / "notes/gradient_teacher_ready.json")
    if helper.get("status") != "ready" or trainer.get("status") != "ready":
        raise ValueError("Both implementations must finish their bounded acceptance checks")
    for record in (helper, trainer):
        for relative, expected in record["source_sha256"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Ready source changed: {relative}")
    from geobc.gradient_train import parser, _scope
    plan = read(ROOT / "notes/geobc_exploration_plan.json")
    jobs = plan["jobs"]
    if len(jobs) != 2 or any(job["stage"] != "matched_displacement_gradient_supervision" for job in jobs):
        raise ValueError("Exactly the two predeclared gradient jobs must be pending")
    contracts = []
    for job in jobs:
        command = job["command"]
        args = parser().parse_args(command[command.index("geobc.gradient_train") + 1:])
        contracts.append(_scope(args)[4])
    changed = differences(*contracts)
    if not changed or any(path[-1] != "gradient_weight" for path in changed):
        raise ValueError(f"Matched metadata differs beyond gradient weight: {changed}")
    if [value["requested"]["gradient_weight"] for value in contracts] != [0, 1]:
        raise ValueError("Expected lambda0 followed by lambda1")
    result = {
        "status": "ready", "phase": plan["stage"], "jobs": [job["name"] for job in jobs],
        "source_sha256": contracts[0]["code_sha256"],
        "cache_manifest_sha256": contracts[0]["gradient_supervision"]["manifest_sha256"],
        "contract_difference_paths": [".".join(path) for path in changed],
        "scope": "metadata/source comparison only; no array/weight/model/CUDA loading or repeated synthetic tests",
        "field_or_weight_arrays_opened": False, "CUDA_used": False,
        "acceptance": "Existing bounded helper/trainer checks, frozen original modules, locked units/Frobenius/volume normalization/common mixed-query design accepted. First native parameter-gradient check remains in each training run before optimizer step."
    }
    path = ROOT / "notes/gradient_parent_review.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)
    print(json.dumps({"status": "ready", "jobs": result["jobs"], "differences": result["contract_difference_paths"]}))


if __name__ == "__main__":
    main()
