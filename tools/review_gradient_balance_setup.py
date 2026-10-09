"""Verify the calibrated entry point can reuse its immutable zero control."""
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


def computing_hashes(path):
    tree = ast.parse(path.read_text(encoding="utf8"))
    functions = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name not in ("_scope", "parser")}
    constants = {}
    locked = {"ROOT", "BASIS_ORDER", "STRAIN_FLOOR", "UNIFORM_NODES", "CELLS_PER_CASE", "NODES_PER_CASE"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in locked for t in node.targets):
            constants[ast.dump(node.targets[0])] = ast.dump(node, include_attributes=False)
    imports = [ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    return functions, constants, imports


def common_contract(value, calibrated=False):
    result = copy.deepcopy(value)
    result.pop("version")
    result["requested"].pop("gradient_weight")
    result["gradient_supervision"].pop("gradient_weight")
    if calibrated:
        result["requested"].pop("gradient_calibration")
        result.pop("gradient_calibration")
        result["code_sha256"].pop("gradient_balance_train.py")
    return result


def main():
    ready = read(ROOT / "notes/gradient_balance_ready.json")
    if ready.get("status") != "ready":
        raise ValueError("Calibrated entry point is not ready")
    for relative, expected in ready["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Ready source changed: {relative}")
    original, calibrated = (ROOT / "geobc" / name for name in ("gradient_train.py", "gradient_balance_train.py"))
    if computing_hashes(original) != computing_hashes(calibrated):
        raise ValueError("Compute functions, query constants or imports differ")
    from geobc.gradient_train import parser as old_parser, _scope as old_scope
    from geobc.gradient_balance_train import parser as new_parser, _scope as new_scope
    plan = read(ROOT / "notes/geobc_exploration_plan.json")
    if len(plan["jobs"]) != 1 or plan["jobs"][0]["name"] != "displacement_gradient_calibrated_10k":
        raise ValueError("Only one fixed calibrated run is authorized in this stage")
    control = next(job for job in plan["historical_jobs"] if job["name"] == "displacement_gradient_zero_10k")
    baseline = read(ROOT / control["output"] / "summary.json")
    old_command, new_command = control["command"], plan["jobs"][0]["command"]
    old = old_scope(old_parser().parse_args(old_command[old_command.index("geobc.gradient_train") + 1:]))[4]
    new = new_scope(new_parser().parse_args(new_command[new_command.index("geobc.gradient_balance_train") + 1:]))[4]
    if baseline["status"] != "completed" or baseline["config"]["contract"] != old:
        raise ValueError("Immutable zero control has drifted")
    changed = differences(common_contract(old), common_contract(new, calibrated=True))
    if changed:
        raise ValueError(f"Common controls differ: {changed}")
    record = {"status": "ready", "baseline": control["output"],
              "calibrated_job": plan["jobs"][0]["name"], "baseline_reuse_accepted": True,
              "gradient_weight": new["requested"]["gradient_weight"],
              "source_sha256": new["code_sha256"],
              "computing_function_ast_sha256": computing_hashes(original)[0],
              "scope": "Metadata/AST/source acceptance only; no arrays, weights, CUDA or repeated synthetic tests",
              "common_contract_equal_after_declared_recipe_metadata": True,
              "runtime_followup": "Compare initialization and first-case/node/cell/corner witness after the calibrated run completes"}
    path = ROOT / "notes/gradient_balance_parent_review.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: record[key] for key in ("status", "baseline_reuse_accepted", "gradient_weight")}))


if __name__ == "__main__":
    main()
