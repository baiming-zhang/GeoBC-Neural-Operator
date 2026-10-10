"""Shared inputs and deferred DEV targets for the frozen coverage comparison.

Metadata entry points use only the standard library. Actual preparation reads
new H5 vertices/cells, real deck inputs and CSV GRID/coordinates. New TRAIN
known3 targets are permitted after release; DEV targets require the immutable
complete fixed prediction-or-terminal-failure inventory for all ten arms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/coverage_comparison_data.py"
PLAN = "notes/geobc_coverage_comparison_plan.json"
LOADS = ["ver", "hor", "tor"]
SCHEMA = "coverage_original25_shared_inputs_and_deferred_DEV_v1"
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
COORDS = ["nodeID", "coord_x(mm)", "coord_y(mm)", "coord_z(mm)"]
DEPENDENCIES = [SOURCE, "geobc/blind_solver_inputs.py", "geobc/elastic_pilot_data.py",
                "geobc/conditioned_field.py", "geobc/solver_selection_data.py",
                "geobc/benchmark.py", "tools/mesh_boundary.py", "tools/prepare_deepjeb.py"]
COUNTERS = ("raw_h5_input_reads", "raw_deck_input_reads", "raw_deck_header_reads",
            "CSV_coordinate_reads", "input_member_reads", "TRAIN_source_target_parse_attempts",
            "TRAIN_source_target_parses", "DEV_source_target_parse_attempts", "DEV_source_target_parses",
            "TRAIN_U_target_case_reads", "TRAIN_stress_target_case_reads", "DEV_U_target_case_reads",
            "DEV_stress_target_case_reads", "owned_DEV_target_cache_reads")
FORBIDDEN = ("H5_nodal_variable_reads", "source_strain_reads", "dia_target_reads",
             "old_DEV_target_reads", "old_final_reads", "sealed_new_test_reads", "scaler_fits")


class UnsupportedInput(RuntimeError):
    pass


class UncommittedDEVRead(RuntimeError):
    pass


def require(value, message, error=ValueError):
    if not bool(value):
        raise error(message)


def path(value):
    text = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", text):
        text = "/mnt/" + text[0].lower() + text[2:]
    result = Path(text)
    return result if result.is_absolute() else ROOT / result


def relative(value):
    return path(value).resolve().relative_to(ROOT.resolve()).as_posix()


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sha(value):
    result = hashlib.sha256()
    with path(value).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def identity(value, hashed=False):
    filename = path(value); stat = filename.stat()
    result = {"path": relative(filename), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hashed:
        result["sha256"] = sha(filename)
    return result


def stat_matches(record):
    actual = identity(record["path"])
    require(all(actual[key] == record[key] for key in ("bytes", "mtime_ns")),
            "File stat differs: " + record["path"])


def write(value, record):
    filename = path(value); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(7):
        try:
            temporary.replace(filename); return
        except PermissionError:
            if attempt == 6: raise
            time.sleep(.05 * (attempt + 1))


def keys(cads):
    return [cad + "/" + load for cad in cads for load in LOADS]


def _groups(science):
    result = {"train40": science["train40_CADs"], "train_new160": science["additional_train_CADs"],
              "train200": science["train200_CADs"], "validation20": science["DEV_CADs"]}
    require([len(result[name]) for name in result] == [40, 160, 200, 20], "Frozen group counts differ")
    require(result["train200"] == result["train40"] + result["train_new160"], "TRAIN200 order differs")
    require(len(set(result["train200"] + result["validation20"])) == 220, "CAD groups overlap/repeat")
    require(science["loads"] == LOADS, "Known-load scope differs")
    return result


def _legacy(contract, source_manifest):
    records = {}
    for key in contract["train_keys"]["40"]:
        item = source_manifest["records"][key]
        files = {name: identity(item[name]) for name in item if name.endswith("_path")}
        records[key] = {"key": key, "cad_id": key.split("/")[0], "group": "train", "num_nodes": item["num_nodes"],
                        "status": "completed", "feasibility_pass": True, "reused_TRAIN40": True,
                        "label_alignment": ALIGNMENT, "file_identities": files,
                        **{name: files[name]["path"] for name in files}}
    return records


def request_contract(plan_path=PLAN):
    """Source/small-JSON hashes and stats only; no raw header/array reads."""
    plan = read(plan_path); science = plan["scientific"]; groups = _groups(science)
    require(digest(science) == plan["scientific_digest"], "Frozen scientific digest differs")
    recipe = science["recipe"]
    require(recipe["cloud_points"] == 1024 and recipe["cloud_seed"] == 2026
            and recipe["grid_resolution"] == 64 and recipe["channels"] == [0, 1, 2, 3], "Input layout changed")
    old_manifest_path = path(science["existing_TRAIN40_cache"]) / "manifest.json"
    old = read(old_manifest_path); old_identity = identity(old_manifest_path, True)
    require(old_identity["sha256"] == science["scaler"]["source_manifest_sha256"]
            and old.get("status") == "completed" and old["scaler"] == science["scaler"]["values"], "Frozen TRAIN40 scaler/source differs")
    train = {"40": keys(groups["train40"]), "200": keys(groups["train200"])}; validation = keys(groups["validation20"])
    require(set(train["40"]) == set(old["train_keys"]), "Legacy TRAIN40 scope differs")
    pinned = plan.get("source_sha256", {})
    sources = {name: sha(name) for name in sorted(set(DEPENDENCIES) | set(pinned))}
    for name, expected in pinned.items():
        require(sources[name] == expected, "Pinned source differs: " + name)
    parent_path = plan.get("input_reader_policy", "notes/geobc_elastic_crosscad_plan.json")
    parent = read(parent_path)
    acquisition = plan["acquisition"]; raw_root = path(acquisition["output"])
    summary_path = raw_root / "summary.json"
    received = read(summary_path) if summary_path.is_file() else None
    receipt_files = received.get("files", []) if received else []
    if isinstance(receipt_files, dict): receipt_files = list(receipt_files.values())
    files = {}; available = 0
    require(len(acquisition["files"]) == 540, "Frozen raw-file inventory differs")
    for item in acquisition["files"]:
        cad, component = item["cad_id"], item["component"]
        require(cad in groups["train_new160"] + groups["validation20"]
                and component in {"FieldMesh", "Field", "SolverDeck"}, "Unfrozen acquisition entry")
        filename = raw_root / item["relative_path"]
        matches = [entry for entry in receipt_files if entry.get("relative_path") == item["relative_path"]]
        source = {"path": relative(filename), "source_url": item["source_url"], "role": item["role"], "available": False}
        if len(matches) == 1 and matches[0].get("status") == "completed":
            entry = matches[0]; stat = entry.get("file_stat", entry.get("stat"))
            require(stat is not None and entry.get("observed_sha256"), "Received file lacks honest byte identity")
            require(entry.get("source_url") == item["source_url"] and type(entry.get("actual_bytes")) is int
                    and entry["actual_bytes"] == stat["bytes"], "Received official URL/byte receipt differs")
            canonical_path = raw_root / ".receipts" / (item["relative_path"] + ".json")
            canonical = read(canonical_path)
            require(canonical.get("status") == "completed" and all(canonical.get(name) == entry.get(name)
                    for name in ("relative_path", "source_url", "actual_bytes", "observed_sha256", "file_stat")), "Canonical file receipt differs")
            source.update(bytes=stat["bytes"], mtime_ns=stat["mtime_ns"], sha256=entry["observed_sha256"], available=True,
                          checksum_origin=entry.get("checksum_origin", "locally_observed_received_bytes_not_upstream_checksum"))
            stat_matches(source); available += 1
        require(component not in files.setdefault(cad, {}), "Duplicate acquisition member")
        files[cad][component] = source
    require(set(files) == set(groups["train_new160"] + groups["validation20"])
            and all(set(value) == {"FieldMesh", "Field", "SolverDeck"} for value in files.values()), "Frozen raw triplets incomplete")
    input_recipe = {"H5_members": ["vertices", "cells"], "coordinate_columns": COORDS,
        "known_loads": LOADS, "features": 25, "GRID_H5_tolerance_mm": science["CSV_GRID_to_H5_tolerance_mm"],
        "cloud_points": 1024, "cloud_seed": 2026, "grid_resolution": 64, "grid_channels": 29,
        "normalization": "original FP32 xyz_mm; (xyz_mm-bbox_min)/max_bbox_extent",
        "cloud": "unique original quadratic-boundary FEM IDs; old BLAKE2b seed2026:CAD, without replacement, zero padding/bool mask",
        "support_flags": "actual RBE2 CM on mapped members, separately recorded actual SPC, never role-implied clamping",
        "material": "explicit real MAT1 E/nu/rho; no defaults", "reader_scope": parent["recipe"]["reader_scope"],
        "load_transfer": parent["recipe"]["load_transfer"], "seed": 2026,
        "scaler_fit": "none; exact original TRAIN40-only frozen scaler", "DEV_input_targets": False}
    contract = {"schema": SCHEMA, "plan": relative(plan_path), "scientific_digest": plan["scientific_digest"],
                "source_sha256": sources, "groups": groups, "train_keys": train, "validation_keys": validation,
                "scaler": science["scaler"]["values"], "legacy_manifest": old_identity,
                "reader_policy": identity(parent_path, True), "input_recipe": input_recipe,
                "raw_sources": files, "acquisition_available_files": available,
                "acquisition_complete": received is not None and received.get("status") == "completed" and available == 540,
                "acquisition_receipt": identity(summary_path, True) if received else None,
                "required_predictions": [{"arm": arm["id"], "step": step, "key": key}
                    for arm in science["arm_order"] for step in recipe["evaluation_updates"] for key in validation]}
    contract["legacy_records"] = _legacy(contract, old)
    return contract, plan


def _released(contract, plan):
    settings = plan.get("root_review", {})
    review_path = settings.get("path", "notes/geobc_coverage_root_review.json") if isinstance(settings, dict) else settings
    review = read(review_path)
    require(review.get("passed") is True and review.get("scientific_digest") == contract["scientific_digest"]
            and review.get("source_pins") == plan.get("source_sha256")
            and review.get("preparation_and_training_released") is True
            and set(DEPENDENCIES) <= set(plan.get("source_sha256", {})), "Root input/training release missing")
    require(contract["acquisition_complete"], "Fixed540 source acquisition incomplete")


def request_bootstrap_contract(plan_path=PLAN):
    """Stable old-TRAIN120 identity, independent of ongoing new acquisition."""
    plan = read(plan_path); science = plan["scientific"]; groups = _groups(science)
    require(digest(science) == plan["scientific_digest"], "Frozen scientific digest differs")
    filename = path(science["existing_TRAIN40_cache"]) / "manifest.json"
    old = read(filename); old_id = identity(filename, True)
    require(old.get("status") == "completed" and old_id["sha256"] == science["scaler"]["source_manifest_sha256"]
            and old["scaler"] == science["scaler"]["values"], "Frozen bootstrap scaler/manifest differs")
    selected = keys(groups["train40"])
    require(set(selected) == set(old["train_keys"]), "Bootstrap old TRAIN120 scope differs")
    pinned = plan.get("source_sha256", {})
    sources = {name: sha(name) for name in sorted(set(DEPENDENCIES) | set(pinned))}
    for name, expected in pinned.items(): require(sources[name] == expected, "Bootstrap source differs: " + name)
    contract = {"schema": "coverage_TRAIN40_bootstrap_v1", "phase": "train40_only",
        "plan": relative(plan_path), "scientific_digest": plan["scientific_digest"], "source_sha256": sources,
        "train_keys": {"40": selected}, "validation_keys": keys(groups["validation20"]),
        "scaler": science["scaler"]["values"], "legacy_manifest": old_id,
        "new_acquisition_or_DEV_arrays_used": False, "complete_shared660": False}
    contract["legacy_records"] = _legacy(contract, old)
    return contract, plan


def bootstrap_TRAIN40(plan_path=PLAN):
    """Metadata-only completed TRAIN40 scope; never pretend to prepare full660."""
    contract, plan = request_bootstrap_contract(plan_path)
    settings = plan.get("root_review", {})
    review_path = settings.get("path", "notes/geobc_coverage_root_review.json") if isinstance(settings, dict) else settings
    review = read(review_path)
    require(review.get("passed") is True and review.get("scientific_digest") == contract["scientific_digest"]
            and review.get("source_pins") == plan.get("source_sha256")
            and review.get("preparation_and_training_released") is True
            and set(DEPENDENCIES) <= set(plan.get("source_sha256", {})), "Root TRAIN40 bootstrap release missing")
    filename = path(plan["data_cache"]) / "train40_bootstrap.json"
    if filename.exists(): return read_bootstrap_TRAIN40(plan_path)
    result = {"kind": "coverage_TRAIN40_bootstrap", "status": "completed", "phase": "train40_only",
        "feasibility_pass": True, "complete_shared660": False, "contract": contract,
        "train_keys": contract["train_keys"], "validation_keys": contract["validation_keys"],
        "records": contract["legacy_records"], "scaler": contract["scaler"],
        "read_receipt": {"new_raw_reads": 0, "old_DEV_target_reads": 0, "target_array_reads": 0, "scaler_fits": 0},
        "scope": "Completed old TRAIN120 metadata references only; DEV60 metadata names, no DEV arrays or targets"}
    write(filename, result)
    return result


def read_bootstrap_TRAIN40(plan_path=PLAN):
    contract, plan = request_bootstrap_contract(plan_path)
    result = read(path(plan["data_cache"]) / "train40_bootstrap.json")
    require(result.get("kind") == "coverage_TRAIN40_bootstrap" and result.get("phase") == "train40_only"
            and result.get("status") == "completed" and result.get("complete_shared660") is False
            and result.get("contract") == contract and result.get("records") == contract["legacy_records"],
            "Bootstrap identity/scope differs")
    return result


class Ledger:
    def __init__(self, filename, contract):
        self.filename = path(filename); self.value = read(filename) if self.filename.exists() else {
            "input_contract_sha256": digest(contract), **{name: 0 for name in COUNTERS + FORBIDDEN},
            "attempted_reads": {}, "completed_reads": {}}
        require(self.value["input_contract_sha256"] == digest(contract), "Persistent input ledger contract differs")
        require(all(type(self.value.get(name)) is int and self.value[name] >= 0 for name in COUNTERS)
                and all(type(self.value.get(name)) is int and self.value[name] == 0 for name in FORBIDDEN), "Read ledger native schema differs")
    def attempt(self, filename, member=""):
        name = str(filename) + (":" + member if member else "")
        self.value["attempted_reads"][name] = self.value["attempted_reads"].get(name, 0) + 1
        write(self.filename, self.value)
    def complete(self, filename, counter=None, member=""):
        name = str(filename) + (":" + member if member else "")
        self.value["completed_reads"][name] = self.value["completed_reads"].get(name, 0) + 1
        if counter: self.value[counter] += 1
        write(self.filename, self.value)


def _save(np, filename, value):
    filename = path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name + ".tmp")
    with temporary.open("wb") as handle: np.save(handle, value, allow_pickle=False)
    temporary.replace(filename)
    return {**identity(filename, True), "shape": list(value.shape), "dtype": str(value.dtype)}


def _columns(filename):
    with path(filename).open(encoding="utf-8-sig", newline="") as handle:
        result = next(csv.reader(handle))
    require(len(result) == len(set(result)), "Duplicate CSV header columns", UnsupportedInput)
    return result


def _targets(np, filename, columns, permutation, node_count, on_loaded=None):
    names = [name for load in LOADS for name in
             [load + "_x_disp(mm)", load + "_y_disp(mm)", load + "_z_disp(mm)", load + "_stress(MPa)"]]
    require(all(name in columns for name in names), "Known3 original U/S columns absent", UnsupportedInput)
    values = np.loadtxt(path(filename), delimiter=",", skiprows=1, usecols=[columns.index(name) for name in names],
                        dtype=np.float64, ndmin=2)
    if on_loaded: on_loaded()
    require(values.shape == (node_count, 12) and np.isfinite(values).all(), "Known3 target shape/nonfinite differs", UnsupportedInput)
    result = np.ascontiguousarray(values[permutation], dtype=np.float32)
    require(np.isfinite(result).all(), "Original four-field FP32 representation overflows", UnsupportedInput)
    return result


def _targets_completed(ledger, scope, filename):
    ledger.complete(filename, scope + "_source_target_parses", "known3_U_S")
    ledger.value[scope + "_U_target_case_reads"] += 3
    ledger.value[scope + "_stress_target_case_reads"] += 3
    write(ledger.filename, ledger.value)


def _deck_case(deck, controls, load):
    label = {"ver": "vertical", "hor": "horizontal", "tor": "torsion"}[load]
    matches = [value for value in controls.values() if value.get("LABEL", "").lower() == label]
    require(len(matches) == 1 and matches[0].get("ANALYSIS", "").upper() == "STATICS"
            and {"SPC", "LOAD"} <= matches[0].keys() and "METHOD" not in matches[0], "Known static subcase unsupported: " + load, UnsupportedInput)
    def expand(sid, kind, stack=(), scale=1.):
        require(sid not in stack, "Cyclic real LOAD/SPCADD", UnsupportedInput)
        combined, direct = ("loadadd", "actions") if kind == "load" else ("spcadd", "spc")
        if sid in deck[combined]:
            require(sid not in deck[direct], "Real LOAD/SPC SID collision", UnsupportedInput)
            if kind == "load":
                return [item for factor, child in deck[combined][sid] for item in expand(child, kind, stack + (sid,), scale * factor)]
            return [item for child in deck[combined][sid] for item in expand(child, kind, stack + (sid,))]
        require(sid in deck[direct], "Undefined real LOAD/SPC set", UnsupportedInput)
        if kind == "load": return [{**item, "vector": [scale*x for x in item["vector"]]} for item in deck[direct][sid]]
        return deck[direct][sid]
    actions = expand(matches[0]["LOAD"], "load"); constraints = expand(matches[0]["SPC"], "spc")
    references = {group["reference"] for group in deck["rbe2"].values()}
    require(len(references) == 4 and references <= {item[0] for item in constraints}, "Four actual constrained support references required", UnsupportedInput)
    loaded = next(iter(deck["rbe3"].values()))
    require(actions and all(item["node"] == loaded["reference"] for item in actions), "Actual load/reference mapping differs", UnsupportedInput)
    return {**deck, "active_actions": actions, "active_constraints": constraints, "selected_controls": matches[0]}


def _prepare_CAD(contract, plan, cad, output, ledger, status_callback):
    import numpy as np
    import h5py
    import torch
    from scipy.spatial import cKDTree
    from geobc import elastic_pilot_data as parser
    from geobc.blind_solver_inputs import feature_arrays, UnsupportedInput as FeatureUnsupported
    from tools.mesh_boundary import quadratic_boundary_triangles
    from geobc.solver_selection_data import _stencil, _splat
    from geobc.benchmark import FieldScaler
    began = time.monotonic(); folder = output / "geometry" / cad; manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        old = read(manifest_path)
        require(old["input_contract_sha256"] == digest(contract), "Owned CAD input contract differs")
        if old.get("status") == "completed":
            for record in old.get("records", {}).values():
                for item in record.get("file_identities", {}).values(): stat_matches(item)
            return old
    result = {"cad_id": cad, "status": "running", "feasibility_pass": False, "input_contract_sha256": digest(contract), "records": {}, "costs": {}}
    folder.mkdir(parents=True, exist_ok=True)
    if status_callback: status_callback({"phase": "preparing", "cad_id": cad})
    def phase(name, function):
        tick = time.monotonic(); value = function(); result["costs"][name] = time.monotonic() - tick; return value
    sources = contract["raw_sources"][cad]
    try:
        for source in sources.values():
            stat_matches(source); ledger.attempt(source["path"], "byte_hash")
            value = sha(source["path"]); ledger.complete(source["path"], member="byte_hash")
            require(value == source["sha256"], "Acquired byte identity differs")
        def mesh():
            filename = sources["FieldMesh"]["path"]; ledger.attempt(filename)
            with h5py.File(path(filename), "r") as handle:
                values = {}
                for member in ("vertices", "cells"):
                    ledger.attempt(filename, member); values[member] = handle[member][:]
                    ledger.complete(filename, "input_member_reads", member)
            ledger.complete(filename, "raw_h5_input_reads")
            xyz_mm = np.asarray(values["vertices"], np.float32); cells = values["cells"]
            require(xyz_mm.ndim == 2 and xyz_mm.shape[1] == 3 and len(xyz_mm) and np.isfinite(xyz_mm).all()
                    and cells.ndim == 2 and cells.shape[1] == 10 and len(cells) and cells.dtype.kind in "iu"
                    and cells.min() >= 0 and cells.max() < len(xyz_mm), "H5 input schema unsupported", UnsupportedInput)
            return xyz_mm, np.asarray(cells, np.int64)
        xyz_mm, cells = phase("H5_input_seconds", mesh)
        filename = sources["SolverDeck"]["path"]; ledger.attempt(filename)
        try:
            deck = phase("deck_parse_seconds", lambda: parser._parse_deck(path(filename), contract["input_recipe"], began, 3600))
            ledger.complete(filename, "raw_deck_input_reads")
            ledger.attempt(filename, "case_control_header")
            generator = parser._cards(path(filename), contract["input_recipe"]["reader_scope"], began, 3600)
            try: controls = next(generator)[3]
            finally: generator.close()
            ledger.complete(filename, "raw_deck_header_reads", "case_control_header")
        except (parser.UnsupportedInput, ValueError) as error:
            raise UnsupportedInput(str(error)) from error
        def coordinates():
            filename = sources["Field"]["path"]; ledger.attempt(filename, "GRID_coordinate_columns")
            columns = _columns(filename)
            require(all(name in columns for name in COORDS), "GRID/coordinate columns absent", UnsupportedInput)
            data = np.loadtxt(path(filename), delimiter=",", skiprows=1, usecols=[columns.index(name) for name in COORDS], dtype=np.float64, ndmin=2)
            ledger.complete(filename, "CSV_coordinate_reads", "GRID_coordinate_columns")
            require(data.shape == (len(xyz_mm), 4) and np.isfinite(data).all() and (data[:, 0] > 0).all()
                    and (data[:, 0] < 2**53).all() and np.array_equal(data[:, 0], np.rint(data[:, 0])), "GRID identity schema unsupported", UnsupportedInput)
            csv_ids = data[:, 0].astype(np.int64)
            distance, permutation = cKDTree(data[:, 1:]).query(xyz_mm, workers=-1)
            require(len(np.unique(csv_ids)) == len(xyz_mm) and len(np.unique(permutation)) == len(xyz_mm)
                    and distance.max() <= contract["input_recipe"]["GRID_H5_tolerance_mm"], "CSV/H5 non-bijective alignment", UnsupportedInput)
            ids = csv_ids[permutation]
            require(set(ids.tolist()) <= deck["grid"].keys(), "GRID absent from actual deck", UnsupportedInput)
            deck_xyz = np.asarray([deck["grid"][int(node)]["xyz"] for node in ids], np.float64)
            error = float(np.linalg.norm(deck_xyz - xyz_mm, axis=1).max())
            require(error <= contract["input_recipe"]["GRID_H5_tolerance_mm"], "Deck/H5 mapping disagreement", UnsupportedInput)
            return columns, permutation.astype(np.int64), ids, {"one_to_one": True, "CSV_H5_max_distance_mm": float(distance.max()), "deck_H5_max_distance_mm": error, "GRID_minus1": False}
        columns, permutation, ids, map_info = phase("coordinate_mapping_seconds", coordinates)
        require(all(item["PID"] in deck["psolid"] for item in deck["ctetra"].values()), "Undefined real PSOLID", UnsupportedInput)
        material_ids = {deck["psolid"][item["PID"]]["MID"] for item in deck["ctetra"].values()}
        require(len(deck["ctetra"]) == len(cells) and len(material_ids) == 1 and len(deck["materials"]) == 1
                and material_ids <= deck["materials"].keys(), "Actual single-material TET10 coverage unsupported", UnsupportedInput)
        mid = next(iter(material_ids))
        material = deck["materials"][mid]
        require(material["E"] is not None and material["E"] > 0 and material["nu"] is not None
                and -1 < material["nu"] < .5, "Real MAT1 E/nu unsupported; no default", UnsupportedInput)
        mapped = set(ids.tolist())
        require(len(deck["rbe3"]) == 1 and len(deck["rbe2"]) == 4
                and all(set(item["members"]) <= mapped and item["reference"] in deck["grid"]
                        for item in list(deck["rbe2"].values()) + list(deck["rbe3"].values())),
                "Actual four support/loaded memberships lack GRID mapping", UnsupportedInput)
        try: faces = phase("quadratic_boundary_seconds", lambda: quadratic_boundary_triangles(cells, xyz_mm))
        except ValueError as error: raise UnsupportedInput(str(error)) from error
        facts = {"material": {"MID": mid}}
        case_features = {}; normalization = None; selections = {}
        def build_features(selected):
            try:
                return feature_arrays(np, cKDTree, selected, xyz_mm, ids, facts)
            except (FeatureUnsupported, ValueError) as error:
                raise UnsupportedInput(str(error)) from error
        for load in LOADS:
            selected = _deck_case(deck, controls, load); selections[load] = selected["selected_controls"]
            xyz, features, normalization = phase("features_" + load + "_seconds", lambda selected=selected: build_features(selected))
            case_features[load] = features
        pool = np.unique(faces); count = min(1024, len(pool))
        seed = int.from_bytes(hashlib.blake2b(f"2026:{cad}".encode(), digest_size=8).digest(), "little")
        chosen = np.random.default_rng(seed).choice(pool, count, replace=False)
        cloud = np.zeros((1024, 3), np.float32); mask = np.zeros(1024, bool); cloud_indices = np.full(1024, -1, np.int64)
        cloud[:count], mask[:count], cloud_indices[:count] = xyz[chosen], True, chosen
        geometry = {name: _save(np, folder / (name + ".npy"), value) for name, value in
                    (("xyz", xyz), ("cloud", cloud), ("mask", mask), ("grid_node_ids", ids), ("h5_row_to_csv", permutation), ("cloud_indices", cloud_indices))}
        device = plan["scientific"]["recipe"]["device"]
        grid_began = time.monotonic(); xyz_gpu = torch.from_numpy(xyz).to(device)
        indices, weights = _stencil(xyz_gpu, 64)
        geometry["query_indices"] = _save(np, folder / "query_indices.npy", indices.cpu().numpy())
        geometry["query_weights"] = _save(np, folder / "query_weights.npy", weights.cpu().numpy())
        axis = torch.linspace(0, 1, 64, device=device); xyz_grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
        result["costs"]["geometry_stencil_transfer_persistence_seconds"] = time.monotonic() - grid_began
        group = "train" if cad in contract["groups"]["train_new160"] else "validation"
        targets = None
        if group == "train":
            filename = sources["Field"]["path"]; ledger.value["TRAIN_source_target_parse_attempts"] += 1; ledger.attempt(filename, "known3_U_S")
            targets = phase("TRAIN_target_parse_seconds", lambda: _targets(np, filename, columns, permutation, len(xyz),
                lambda: _targets_completed(ledger, "TRAIN", filename)))
        scaler = FieldScaler(**contract["scaler"])
        grid_began = time.monotonic()
        for index, load in enumerate(LOADS):
            key = cad + "/" + load; directory = output / "cases" / key.replace("/", "_")
            features = case_features[load]; raster, active = _splat(torch.from_numpy(features).to(device), indices, weights, 64)
            grid = torch.cat((active.float().reshape(1, 64, 64, 64), xyz_grid, raster.T.reshape(25, 64, 64, 64)), dim=0)
            files = {name + "_path": geometry[name] for name in ("xyz", "cloud", "mask", "query_indices", "query_weights")}
            files["features_path"] = _save(np, directory / "features.npy", features)
            files["grid_path"] = _save(np, directory / "grid.npy", grid.cpu().numpy())
            if targets is not None:
                physical = targets[:, 4*index:4*index+4]
                files["physical_path"] = _save(np, directory / "physical.npy", physical)
                files["target_path"] = _save(np, directory / "target.npy", scaler.encode(physical))
            result["records"][key] = {"key": key, "cad_id": cad, "group": group, "num_nodes": len(xyz),
                "status": "completed", "feasibility_pass": True, "reused_TRAIN40": False, "label_alignment": ALIGNMENT,
                **{name: value["path"] for name, value in files.items()}, "file_identities": files,
                "deferred_source": sources["Field"], "permutation": geometry["h5_row_to_csv"]}
        torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
        result["costs"]["grid_and_case_persistence_seconds"] = time.monotonic() - grid_began
        result.update(status="completed", feasibility_pass=True, num_nodes=len(xyz), cell_count=len(cells),
            geometry=geometry, source=sources, mapping=map_info, normalization=normalization,
            actual_material=deck["materials"][mid], real_support_CMs=[item["dofs"] for item in sorted(deck["rbe2"].values(), key=lambda item:item["reference"])],
            selected_known_controls=selections, cloud_pool_nodes=len(pool), cloud_valid_points=count,
            DEV_target_reads=0, scaler_fits=0)
    except UnsupportedInput as error:
        result.update(status="completed", feasibility_pass=False, reason=str(error), scientific_outcome="unsupported_input")
    except BaseException as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}"); raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - began; write(manifest_path, result)
    return result


def prepare_inputs(plan_path=PLAN, output=None, status_callback=None):
    began = time.monotonic(); contract, plan = request_contract(plan_path); _released(contract, plan)
    output = path(output if output is not None else plan["data_cache"])
    output.resolve().relative_to(ROOT.resolve()); manifest_path = output / "manifest.json"
    previous_wall = 0.0
    if manifest_path.exists():
        previous = read(manifest_path)
        require(previous["contract"] == contract, "Existing shared input contract differs")
        if previous.get("status") == "completed": return read_cache_metadata(plan_path, contract=contract, cache=output)
        previous_wall = float(previous.get("elapsed_seconds", 0.0))
    output.mkdir(parents=True, exist_ok=True); ledger = Ledger(output / "read_receipt.json", contract)
    manifest = {"kind": "coverage_comparison_inputs", "status": "running", "contract": contract,
        "train_keys": contract["train_keys"], "validation_keys": contract["validation_keys"], "scaler": contract["scaler"],
        "records": dict(contract["legacy_records"]), "CADs": {}, "feasibility_pass": False,
        "prior_recorded_preparation_wall_seconds": previous_wall}
    write(manifest_path, manifest)
    try:
        # Full29 cached grids are shared by arms. Do not silently shrink R or omit loads for disk pressure.
        maximum = int(plan.get("input_cache_maximum_bytes", 150 * 2**30))
        minimum_new_grids = 540 * 29 * 64**3 * 4
        require(minimum_new_grids < maximum and shutil.disk_usage(output).free > minimum_new_grids,
                "Insufficient bounded shared-cache storage", UnsupportedInput)
        for cad in contract["groups"]["train_new160"] + contract["groups"]["validation20"]:
            item = _prepare_CAD(contract, plan, cad, output, ledger, status_callback); manifest["CADs"][cad] = item
            if item["feasibility_pass"]: manifest["records"].update(item["records"])
            else:
                for key in keys([cad]):
                    manifest["records"][key] = {"key": key, "cad_id": cad, "group": "validation" if key in contract["validation_keys"] else "train",
                        "status": "completed", "feasibility_pass": False, "num_nodes": None, "reason": item["reason"]}
            manifest.update(elapsed_seconds=previous_wall+time.monotonic()-began, read_receipt=ledger.value)
            write(manifest_path, manifest)
        owned_bytes = sum(filename.stat().st_size for filename in output.rglob("*.npy"))
        require(owned_bytes <= maximum, "Shared-cache storage cap exceeded", UnsupportedInput)
        manifest.update(status="completed", feasibility_pass=all(item["feasibility_pass"] for item in manifest["CADs"].values()), total_owned_cache_bytes=owned_bytes)
    except UnsupportedInput as error:
        manifest.update(status="completed", feasibility_pass=False, scientific_outcome="readiness_negative", reason=str(error))
    except BaseException as error:
        manifest.update(status="failed", reason=f"{type(error).__name__}: {error}"); raise
    finally:
        manifest.update(elapsed_seconds=previous_wall+time.monotonic()-began,
                        this_call_seconds=time.monotonic()-began, read_receipt=ledger.value)
        write(manifest_path, manifest)
    return manifest


def read_cache_metadata(plan_path=PLAN, *, contract=None, cache=None):
    current, plan = request_contract(plan_path)
    require(contract is None or contract == current, "Shared cache contract differs")
    manifest = read(path(cache if cache is not None else plan["data_cache"]) / "manifest.json")
    require(manifest.get("status") == "completed" and manifest.get("contract") == current
            and manifest.get("train_keys") == current["train_keys"] and manifest.get("validation_keys") == current["validation_keys"], "Shared input manifest differs")
    if manifest.get("feasibility_pass"):
        require(set(manifest["records"]) == set(current["train_keys"]["200"] + current["validation_keys"]), "Shared case inventory differs")
        for record in manifest["records"].values():
            for item in record["file_identities"].values(): stat_matches(item)
    return manifest


def _cache(value):
    if isinstance(value, dict): return value
    filename = path(value)
    return read(filename / "manifest.json" if filename.is_dir() else filename)


def load_case(cache, key, with_targets=False, device="cuda", for_fno=False):
    """DEV inputs remain target-free even after readout; evaluator uses the separate label manifest."""
    manifest = _cache(cache); record = manifest["records"][key]
    require(manifest.get("status") == "completed" and manifest.get("feasibility_pass") is True
            and record.get("feasibility_pass") is True, "Negative shared input cache rejected before CUDA")
    train_scope = manifest["train_keys"].get("200", manifest["train_keys"]["40"])
    require(not with_targets or key in train_scope, "DEV targets are forbidden in input loader")
    import numpy as np
    import torch
    receipt = {"input_array_reads": 0, "cached_TRAIN_target_array_reads": 0, "DEV_target_reads": 0, "source_archive_reads": 0, "scaler_fits": 0, "members": []}
    def array(name, target=False):
        item = record["file_identities"][name + "_path"]; stat_matches(item)
        value = np.load(path(item["path"]), mmap_mode="r", allow_pickle=False)
        receipt["cached_TRAIN_target_array_reads" if target else "input_array_reads"] += 1; receipt["members"].append(name)
        if "shape" in item: require(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"], "Cached member schema differs")
        return value
    def tensor(name, target=False):
        return torch.as_tensor(np.array(array(name, target), copy=True), device=device)
    result = {"key": key, "xyz": tensor("xyz"), "features": tensor("features"), "cloud": tensor("cloud").reshape(1, 1024, 3), "mask": tensor("mask").reshape(1, 1024).bool()}
    if with_targets: result.update(target=tensor("target", True), physical=array("physical", True))
    if for_fno: result.update(grid=tensor("grid").reshape(1, 29, 64, 64, 64), query_indices=tensor("query_indices"), query_weights=tensor("query_weights"))
    result["read_receipt"] = receipt; result["_actual_read_receipt"] = receipt
    return result


def _barrier(barrier, contract, records):
    required = contract["required_predictions"]
    require(barrier.get("passed") is True and barrier.get("input_contract_sha256") == digest(contract)
            and barrier.get("validation_keys") == contract["validation_keys"]
            and barrier.get("required_predictions") == required
            and barrier.get("training_permanently_closed") is True, "Complete immutable prediction-or-failure inventory missing")
    inventory = barrier.get("prediction_manifests", {})
    expected = {f"{item['arm']}:{item['step']}:{item['key']}" for item in required}
    require(set(inventory) == expected, "Fixed2400 barrier inventory differs")
    for item in required:
        entry = inventory[f"{item['arm']}:{item['step']}:{item['key']}"]
        require(sha(entry["path"]) == entry["sha256"], "Committed terminal manifest changed")
        value = read(entry["path"])
        require(value.get("status") == "completed" and value.get("terminal") is True
                and value.get("arm") == item["arm"] and value.get("step") == item["step"] and value.get("key") == item["key"]
                and value.get("input_contract_sha256") == digest(contract), "Terminal prediction identity differs")
        if value.get("prediction_complete") is True:
            n = records[item["key"]]["num_nodes"]
            require(value.get("finite_allN") is True and value.get("all_original_nodes") is True and value.get("node_count") == n
                    and value.get("encoded_shape") == [n, 4]
                    and value.get("physical_shape") == [n, 4], "Finite fullN signed4 prediction proof incomplete")
            for name in ("encoded_identity", "physical_identity"): stat_matches(value[name])
        else:
            require(value.get("prediction_complete") is False and isinstance(value.get("reason"), str)
                    and bool(value["reason"]), "Explicit frozen terminal failure required")


def read_dev_targets_after_barrier(plan_path, cache, barrier):
    """One shared source parse per new DEV CAD, after all frozen terminal entries commit."""
    contract, plan = request_contract(plan_path); _released(contract, plan); manifest = _cache(cache)
    require(manifest.get("status") == "completed" and manifest.get("feasibility_pass") is True
            and manifest["contract"] == contract, "DEV target alignment inputs incomplete")
    _barrier(barrier, contract, manifest["records"])
    output = path(plan["data_cache"]) / "DEV_targets"; output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output / "read_receipt.json", contract); result_path = output / "manifest.json"
    if result_path.exists():
        previous = read(result_path)
        require(previous["contract"] == contract and previous["barrier_sha256"] == digest(barrier), "Owned DEV target barrier differs")
        if previous.get("status") == "completed":
            for record in previous["records"].values():
                for item in record["file_identities"].values(): stat_matches(item)
            return previous
    import numpy as np
    from geobc.benchmark import FieldScaler
    scaler = FieldScaler(**contract["scaler"])
    result = {"kind": "coverage_shared_DEV_targets", "status": "running", "contract": contract,
              "validation_keys": contract["validation_keys"], "barrier_sha256": digest(barrier),
              "training_permanently_closed": True, "records": {}, "read_receipt": ledger.value}
    write(result_path, result)
    try:
        for cad in contract["groups"]["validation20"]:
            folder = output / cad; committed = folder / "manifest.json"; claim = folder / "source_read_started.json"
            if committed.exists():
                old = read(committed)
                require(old["status"] == "completed" and old["contract_sha256"] == digest(contract)
                        and old["barrier_sha256"] == digest(barrier), "Owned DEV CAD target differs")
                result["records"].update(old["records"]); continue
            require(not claim.exists(), "DEV source parse interrupted without commit; never silently reread", UncommittedDEVRead)
            source = contract["raw_sources"][cad]["Field"]; stat_matches(source)
            require(sha(source["path"]) == source["sha256"], "Deferred DEV source changed")
            first = manifest["records"][cad + "/ver"]; item = first["permutation"]; stat_matches(item)
            ledger.attempt(item["path"]); permutation = np.load(path(item["path"]), allow_pickle=False)
            ledger.complete(item["path"], "input_member_reads")
            require(sha(item["path"]) == item["sha256"], "Frozen DEV alignment differs")
            columns = _columns(source["path"])
            write(claim, {"cad_id": cad, "contract_sha256": digest(contract), "barrier_sha256": digest(barrier), "source": source, "completed": False})
            ledger.value["DEV_source_target_parse_attempts"] += 1; ledger.attempt(source["path"], "known3_U_S")
            targets = _targets(np, source["path"], columns, permutation, first["num_nodes"],
                lambda: _targets_completed(ledger, "DEV", source["path"]))
            records = {}
            for index, load in enumerate(LOADS):
                key = cad + "/" + load; physical = targets[:, 4*index:4*index+4]
                files = {"physical_path": _save(np, folder / (load + "_physical.npy"), physical),
                         "target_path": _save(np, folder / (load + "_encoded.npy"), scaler.encode(physical))}
                records[key] = {"key": key, "num_nodes": first["num_nodes"], "physical_units": ["mm", "mm", "mm", "MPa"],
                    **{name: value["path"] for name, value in files.items()}, "file_identities": files}
            write(committed, {"status": "completed", "contract_sha256": digest(contract), "barrier_sha256": digest(barrier), "records": records})
            write(claim, {**read(claim), "completed": True}); result["records"].update(records)
            result["read_receipt"] = ledger.value; write(result_path, result)
        require(set(result["records"]) == set(contract["validation_keys"]), "Shared DEV target inventory incomplete")
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}"); raise
    finally:
        result["read_receipt"] = ledger.value; write(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output")
    args = parser.parse_args(); prepare_inputs(args.plan, args.output)


if __name__ == "__main__":
    main()
