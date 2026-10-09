"""Release one predeclared query-context pair through source/metadata review."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_gradient_teacher_pair import differences
from tools.review_bc_role_setup import computing_functions


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def common_contract(value, query=False):
    result = copy.deepcopy(value)
    result.pop("version")
    result["bc_roles"].pop("matched_encoder")
    if query:
        result["requested"].pop("query_context_mode")
        result.pop("query_context")
        for filename in ("query_context_field.py", "query_context_train.py"):
            result["code_sha256"].pop(filename)
    return result


def main():
    from geobc.query_context_train import parser, _scope
    from geobc.bc_role_train import parser as base_parser, _scope as base_scope
    for name in ("query_context_field_ready.json", "query_context_train_ready.json"):
        ready = read(ROOT / "notes" / name)
        if ready["status"] != "ready":
            raise ValueError(f"{name} is not ready")
        for relative, expected in ready["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Ready source changed: {relative}")
    base_functions = computing_functions(ROOT / "geobc/bc_role_train.py")
    new_functions = computing_functions(ROOT / "geobc/query_context_train.py")
    if any(new_functions.get(name) != value for name, value in base_functions.items()):
        raise ValueError("Loss/sampling/evaluation compute functions changed")
    plan = read(ROOT / "notes/geobc_exploration_plan.json")
    jobs = plan["jobs"]
    if [job["name"] for job in jobs] != ["displacement_query_context_zero_10k", "displacement_query_context_actual_10k"]:
        raise ValueError("Only the predeclared query pair is released")
    contracts = []
    for job, mode in zip(jobs, ("zero", "actual")):
        command = job["command"]
        args = parser().parse_args(command[command.index("geobc.query_context_train") + 1:])
        if (args.query_context_mode != mode or args.bc_role_mode != "actual" or args.steps != 10000
                or args.seed != 2026 or args.eval_every != 1000 or args.learning_rate != .003 or args.cloud_points != 1024):
            raise ValueError("Predeclared common recipe changed")
        contracts.append(_scope(args)[4])
    changed = differences(*contracts)
    if set(changed) != {("requested", "query_context_mode"), ("query_context", "query_context_mode")}:
        raise ValueError(f"Nonmatched input contract changes: {changed}")
    old_job = next(job for job in plan["historical_jobs"] if job["name"] == "displacement_bc_role_actual_10k")
    command = old_job["command"]
    old = base_scope(base_parser().parse_args(command[command.index("geobc.bc_role_train") + 1:]))[4]
    if differences(common_contract(old), common_contract(contracts[0], query=True)):
        raise ValueError("Query data/cache/loss/sampling changed beyond declared architecture")
    context = contracts[0]["query_context"]
    if context["sigma"] != .15 or context["dimension"] != 32 or context["query_chunk"] != 4096:
        raise ValueError("Fixed query descriptor settings changed")
    record = {"status": "ready", "jobs": [job["name"] for job in jobs],
              "source_sha256": contracts[0]["code_sha256"], "pair_contract_differences": [".".join(path) for path in changed],
              "unchanged_computation_functions": list(base_functions), "same_data_loss_cache_sampler_as_BC": True,
              "old_control_reused": False, "sigma": .15, "query_dimension": 32,
              "runtime_checks": "Cloud arrays exact equality; matched complete initial parameters/native layout/sampling; native global/query input gradients",
              "control_scope": "Same stored parameter architecture; activating query branch may change effective function capacity",
              "review_scope": "Source AST/ready hashes and strict metadata/stat only; no arrays, models, checkpoints or CUDA"}
    path = ROOT / "notes/query_context_parent_review.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "ready", "pair_contract_only_query_mode_differs": True,
                      "same_data_loss_cache_sampler_as_BC": True, "old_control_reused": False}))


if __name__ == "__main__":
    main()
