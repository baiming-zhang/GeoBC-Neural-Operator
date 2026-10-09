"""Release the predeclared BC-role pair after a small source/metadata review."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_gradient_teacher_pair import differences


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def computing_functions(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name not in ("_scope", "parser", "run")}


def common_role_contract(value):
    result = copy.deepcopy(value)
    result.pop("version")
    result["requested"].pop("bc_role_mode")
    result.pop("bc_roles")
    for name in ("bc_role_train.py", "bc_role_field.py", "bc_role_condition.py"):
        result["code_sha256"].pop(name)
    return result


def main():
    from geobc.bc_role_train import parser as role_parser, _scope as role_scope
    from geobc.gradient_balance_train import parser as base_parser, _scope as base_scope
    for filename in ("bc_role_field_ready.json", "bc_role_train_ready.json"):
        ready = read(ROOT / "notes" / filename)
        if ready.get("status") != "ready":
            raise ValueError(f"{filename} is not ready")
        for relative, expected in ready["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Ready source changed: {relative}")
    original = ROOT / "geobc/gradient_balance_train.py"
    candidate = ROOT / "geobc/bc_role_train.py"
    if computing_functions(original) != computing_functions(candidate):
        raise ValueError("Unrelated loss/sampling/evaluation compute functions changed")
    plan = read(ROOT / "notes/geobc_exploration_plan.json")
    jobs = plan["jobs"]
    if [job["name"] for job in jobs] != ["displacement_bc_role_zero_10k", "displacement_bc_role_actual_10k"]:
        raise ValueError("Only the predeclared matched seed2026 pair is released")
    contracts = []
    for job, expected_mode in zip(jobs, ("zero", "actual")):
        command = job["command"]
        args = role_parser().parse_args(command[command.index("geobc.bc_role_train") + 1:])
        if (args.bc_role_mode != expected_mode or args.seed != 2026 or args.steps != 10000
                or args.cloud_points != 1024 or args.eval_every != 1000 or args.learning_rate != .003):
            raise ValueError("Predeclared role/seed/cloud/budget changed")
        contracts.append(role_scope(args)[4])
    changed = differences(*contracts)
    allowed = {("requested", "bc_role_mode"), ("bc_roles", "bc_role_mode")}
    if set(changed) != allowed:
        raise ValueError(f"Pair contracts differ beyond input mode: {changed}")
    baseline_job = next(job for job in plan["historical_jobs"]
                        if job["name"] == "displacement_gradient_calibrated_10k")
    command = baseline_job["command"]
    baseline = base_scope(base_parser().parse_args(command[command.index("geobc.gradient_balance_train") + 1:]))[4]
    baseline.pop("version")
    if differences(baseline, common_role_contract(contracts[0])):
        raise ValueError("Shared query/loss/data/scaler contract changed beyond declared role inputs")
    record = {"status": "ready", "stage": "global_BC_role_displacement_control",
              "jobs": [job["name"] for job in jobs], "source_sha256": contracts[0]["code_sha256"],
              "matched_contract_differences": [".".join(path) for path in changed],
              "unchanged_compute_functions": list(computing_functions(original)),
              "historical_three_channel_model_reused_as_control": False,
              "common_query_loss_cache_contract_equal": True,
              "runtime_checks": "Exact saved xyz/mask/node_indices reuse; matched initialization and sampling witnesses; native encoder/all-decoder gradients",
              "review_scope": "Source AST, ready hashes and strict metadata/stat only; no arrays/checkpoints/models/CUDA",
              "design": "notes/geobc_bc_role_plan.json"}
    path = ROOT / "notes/bc_role_parent_review.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": record["status"], "pair_contract_only_role_mode_differs": True,
                      "common_query_loss_cache_contract_equal": True}))


if __name__ == "__main__":
    main()
