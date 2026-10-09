"""Freeze a displacement teacher using completed known-load validation only.

Reads configuration/metrics, weight stat and saved geometry metadata, never FEM
field labels. The frozen gate is immutable; use a new gate for a new stage.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["displacement_linear_basis_local_actual_10k",
              "displacement_linear_basis_relative_case_10k"]
GATE = ROOT / "notes/geobc_u_stress_teacher_frozen.json"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def identity(relative):
    path = ROOT / relative
    stat = path.stat()
    return {"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def small_identity(relative):
    result = identity(relative)
    result["sha256"] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return result


def main():
    if GATE.exists():
        record = read(GATE)
        for name in ("config", "summary", "metrics_best"):
            item = record["teacher"][name]
            if small_identity(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("Existing frozen teacher metadata changed")
        if identity(record["teacher"]["checkpoint"]["path"]) != record["teacher"]["checkpoint"]:
            raise ValueError("Existing frozen checkpoint stat changed")
        print(json.dumps({"reused": True, "teacher": record["teacher"]["run"]}))
        return
    split_path = "notes/geobc_split.json"
    split = read(ROOT / split_path)
    candidates = []
    for name in CANDIDATES:
        run = "runs/geobc_exploration/" + name
        summary = read(ROOT / run / "summary.json")
        config = read(ROOT / run / "config.json")
        metrics = read(ROOT / run / "metrics_best.json")
        if summary["status"] != "completed" or summary["best_metrics"] != metrics:
            raise ValueError("Teacher candidate has not completed consistently")
        if config["physics_mode"] != "linear_basis" or config["groups"] != split["groups"]:
            raise ValueError("Teacher scope/model differs from the strict experiment")
        requested = config["contract"]["requested"]
        expected = {"steps": 10000, "hidden_dim": 128, "seed": 2026,
                    "variant": "geometry_pointnet", "local_mode": "actual"}
        if any(requested[k] != v for k, v in expected.items()):
            raise ValueError("Teacher candidate settings differ")
        if config["contract"]["split_sha256"] != small_identity(split_path)["sha256"]:
            raise ValueError("Teacher split was modified")
        candidates.append({"run": run, "best_step": summary["best_step"],
                           "known_validation_u_mean_relative_l2": metrics["validation"]["case_mean_relative_l2"],
                           "training_elapsed_seconds": summary["elapsed_seconds"],
                           "u_loss": config.get("u_loss", {"name": "point_mse"})["name"]})
    chosen = min(candidates, key=lambda item: item["known_validation_u_mean_relative_l2"])
    run = chosen["run"]
    config = read(ROOT / run / "config.json")
    teacher = dict(chosen)
    for name, filename in (("checkpoint", "best.pt"), ("geometry_clouds", "geometry_clouds.npz")):
        teacher[name] = identity(run + "/" + filename)
    for name in ("config", "summary", "metrics_best"):
        teacher[name] = small_identity(run + "/" + name + ".json")
    teacher.update(scaler=config["scaler"], model=config["model"],
                   geometry=config["geometry"], local_geometry=config["local_geometry"],
                   source_code_sha256=config["contract"]["code_sha256"])
    record = {"version": 1, "status": "frozen", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection_rule": "Minimum completed known-load validation15-case physical displacement mean L2; ties use listed candidate order",
              "selection_uses_dia_or_stress_metrics": False, "candidates": candidates, "teacher": teacher,
              "split": small_identity(split_path), "load_basis": small_identity("notes/geobc_load_basis_45.json"),
              "cache_directory": "data/geobc_frozen_u_response",
              "cache_CADs": split["groups"]["train"] + split["groups"]["validation"],
              "cache_load": "ver", "basis_order": ["ver", "hor", "tor"],
              "component_order": ["ux", "uy", "uz"], "student_u_features": 3,
              "cache_generation_label_members_forbidden": ["stress", "displacement"],
              "final_test_excluded": True, "dia_excluded_from_this_stage": True,
              "cost_policy": "Teacher used train u supervision and known-load validation selection. Report both candidate training costs, cache generation and each student cost.",
              "limitations": "Same known-load development validation reused for teacher and student selection. Prior validation-dia probe was observed; final five CAD fields stay locked."}
    GATE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"teacher": run, "known_validation_u_mean_relative_l2": chosen["known_validation_u_mean_relative_l2"],
                      "gate": str(GATE)}))


if __name__ == "__main__":
    main()
