"""Released-only input preparation for the frozen five unseen CAD/ver requests.

Metadata functions use the standard library. No original field target is read
by preparation/loading. The separate shared U reader requires the actual ten
prediction manifests; stress and H5 nodal_variables are never materialized.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/blind_solver_inputs.py"
PLAN = "notes/geobc_blind_solver_plan.json"
CASES = ["489_12/ver", "616_205/ver", "416_282/ver", "247_50/ver", "142_454/ver"]
ARMS = ["physics_full_FE", "FNO_frozen2880"]
COORD_COLUMNS = ["nodeID", "coord_x(mm)", "coord_y(mm)", "coord_z(mm)"]
U_COLUMNS = ["ver_x_disp(mm)", "ver_y_disp(mm)", "ver_z_disp(mm)"]
TOLERANCE_MM = 0.0002
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
SCHEMA = "blind_original25_and_full_TET10_input_v1"
DEPENDENCIES = [SOURCE, "geobc/elastic_pilot_data.py", "tools/mesh_boundary.py",
                "tools/prepare_deepjeb.py", "geobc/continuous_query_geometry.py",
                "geobc/conditioned_field.py", "geobc/solver_selection_data.py"]
FEATURE_NAMES = ([f"support_reference_constrained_dof_{i}" for i in range(1, 7)]
    + ["loaded_interface_mask"] + [f"total_force_{a}_over_42300N" for a in "xyz"]
    + [f"total_moment_{a}_over_565000Nmm" for a in "xyz"]
    + ["E_over_113800MPa", "poisson_ratio", "density_over_4.47e-9_t_per_mm3"]
    + [f"offset_from_load_reference_{a}_over_length" for a in "xyz"]
    + [f"distance_to_support_reference_{i}_over_length" for i in range(1, 5)]
    + ["distance_to_loaded_interface_over_length", "characteristic_length_over_200mm"])
ACTUAL = ("raw_h5_input_reads", "raw_deck_input_reads", "raw_CSV_coordinate_reads",
          "input_cache_reads", "input_member_reads", "source_U_target_parse_attempts",
          "source_U_target_parses", "owned_GT_cache_reads")
FORBIDDEN = ("source_S_target_reads", "H5_nodal_variable_reads", "source_strain_target_reads",
             "old_final_reads", "DEV_dia_reads", "model_or_weight_reads", "optimizer_steps",
             "scaler_fits", "new_physical_solves", "K_applications")


class UnsupportedInput(RuntimeError):
    """Unsupported fixed input/model scope; never substitute another CAD."""


class UncommittedTargetRead(RuntimeError):
    """A started source U parse cannot be silently repeated after interruption."""


def require(condition, message, error=ValueError):
    if not bool(condition):
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
                                     allow_nan=False).encode()).hexdigest()


def sha(value):
    result = hashlib.sha256()
    with path(value).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def identity(value, with_sha=False):
    value = path(value); stat = value.stat()
    result = {"path": relative(value), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if with_sha:
        result["sha256"] = sha(value)
    return result


def stat_matches(record):
    actual = identity(record["path"])
    require(all(actual[key] == record[key] for key in ("bytes", "mtime_ns")),
            "File stat differs: " + record["path"])


def write(value, record):
    value = path(value); value.parent.mkdir(parents=True, exist_ok=True)
    temporary = value.with_name(value.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(7):
        try:
            temporary.replace(value); return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * (attempt + 1))


def metadata(plan, key, default):
    supplied = plan.get(key, default)
    record = supplied if isinstance(supplied, dict) else {"path": supplied}
    value = read(record["path"]); actual = identity(record["path"], True)
    if record.get("sha256"):
        require(record["sha256"] == actual["sha256"], "Pinned metadata differs: " + key)
    return value, actual


def request_contract(plan_path=PLAN):
    """JSON/source hashes/file stats only; never inspect a new raw header."""
    plan = read(plan_path)
    require(plan.get("case_keys") == CASES, "Blind five-case order differs")
    root_recipe, root_id = metadata(plan, "root_recipe", "notes/geobc_blind_solver_root_recipe.json")
    require(root_recipe["case_keys"] == CASES and root_recipe["route"] ==
            "pure_full_FE_U_native_P2_plus_one_frozen_FNO_node_U_reference", "Blind route changed")
    acquire, acquire_id = metadata(plan, "acquisition_summary", "data/raw_next_cycle/summary.json")
    require(acquire.get("status") == "completed" and acquire.get("completed_files") == 15
            and acquire.get("labels_sealed") is True, "Fixed acquisition metadata incomplete")
    parent, parent_id = metadata(plan, "input_parent_plan", "notes/geobc_elastic_crosscad_plan.json")
    pinned = plan.get("source_sha256", {})
    names = sorted(set(DEPENDENCIES) | set(pinned))
    sources = {name: sha(name) for name in names}
    for name, expected in pinned.items():
        require(sources[name] == expected, "Pinned source changed: " + name)
    records = {}
    for key in CASES:
        cad = key.split("/")[0]; record = {}
        for component, extension in (("FieldMesh", ".h5"), ("Field", ".csv"), ("SolverDeck", ".fem")):
            received = [item for item in acquire["files"]
                        if item.get("relative_path") == f"{component}/{cad}{extension}"]
            require(len(received) == 1 and received[0].get("status") == "completed", "Received source missing")
            item = received[0]; source = {"path": f"data/raw_next_cycle/{component}/{cad}{extension}",
                "bytes": item["file_stat"]["bytes"], "mtime_ns": item["file_stat"]["mtime_ns"],
                "sha256": item["observed_sha256"], "source_url": item["source_url"],
                "checksum_origin": item["checksum_origin"]}
            stat_matches(source); record[component] = source
        records[key] = record
    recipe = {"seed": parent["recipe"]["seed"], "H5_members": ["vertices", "cells"], "coordinate_columns": COORD_COLUMNS,
              "deferred_U_columns": U_COLUMNS, "S_columns": [], "coordinate_tolerance_mm": TOLERANCE_MM,
              "feature_names": FEATURE_NAMES, "feature_source": "tools/prepare_deepjeb.py exact25 inline math",
              "support_flags": "Actual RBE2 CM on member rows, exactly as original feature builder; not guessed Dirichlet DOFs",
              "cloud_points": 1024, "cloud_seed": 2026, "grid_resolution": 64,
              "cloud_source": "geobc/conditioned_field.py load_geometry_cloud exact construction",
              "cloud_normalization": "H5 vertices cast FP32; xyz=(xyz_mm-bbox_min)/max(bbox_max-bbox_min)",
              "cloud_pool": "unique original quadratic-boundary FEM node IDs reconstructed from H5 cells/vertices",
              "cloud_sampling": "default_rng(blake2b('2026:'+CAD,digest_size=8),little-endian).choice without replacement",
              "cloud_padding": "1024 FP32 rows, unused rows zero with bool mask false and node_index -1",
              "reader_scope": parent["recipe"]["reader_scope"], "load_transfer": parent["recipe"]["load_transfer"],
              "missing_material_density_or_four_supports": "unsupported; no default",
              "GRID_mapping": "CSV GRID identity and coordinates -> unique H5 permutation; never GRID-1"}
    contract = {"schema": SCHEMA, "plan": relative(plan_path), "case_keys": CASES, "arms": ARMS,
                "scientific_digest": plan["scientific_digest"], "source_sha256": sources,
                "root_recipe": root_id, "acquisition_summary": acquire_id, "input_parent_plan": parent_id,
                "recipe": recipe, "input_recipe_sha256": digest(recipe), "records": records,
                "source_preparation_reads_new_contents": False}
    return contract, plan


def released(contract, plan):
    # The root intentionally leaves actual_execution_released=false. The
    # authentic final review marker is the sole actual execution release.
    require(plan.get("scientific_rules_frozen") is True, "Blind scientific rules are not frozen")
    require(set(DEPENDENCIES) <= set(plan.get("source_sha256", {})), "Input source pins incomplete")
    review = read(plan.get("root_review", "notes/blind_solver_service_root_review.json"))
    driver = plan.get("driver_source", "geobc/blind_solver_service.py")
    require(review.get("passed") is True and review.get("source_sha256") == contract["source_sha256"].get(driver)
            and isinstance(review.get("source_sha256"), str)
            and review.get("source_pins") == plan["source_sha256"]
            and review.get("scientific_digest") == contract["scientific_digest"]
            and review.get("decision") == "proceed_frozen_blind_full_FE_U_with_FNO_reference",
            "Authentic blind source/policy release missing")
    if "input_contract_sha256" in review:
        require(review["input_contract_sha256"] == digest(contract), "Optional input contract binding differs")


class Ledger:
    def __init__(self, output, contract, observer=None):
        self.filename = path(output) / "read_receipt.json"; self.observer = observer
        self.config = digest(contract)
        self.value = read(self.filename) if self.filename.is_file() else {
            "input_contract_sha256": self.config, **{key: 0 for key in ACTUAL + FORBIDDEN},
            "attempted_reads": {}, "completed_reads": {}}
        require(self.value["input_contract_sha256"] == self.config, "Read ledger contract differs")
        require(all(type(self.value.get(key)) is int and self.value[key] >= 0 for key in ACTUAL)
                and all(type(self.value.get(key)) is int and self.value[key] == 0 for key in FORBIDDEN),
                "Read ledger native count/schema differs")
    def event(self, event, source, member=None):
        if self.observer:
            self.observer(event, source, member)
    def attempt(self, source, member=None):
        name = str(source) + (":" + member if member else "")
        self.value["attempted_reads"][name] = self.value["attempted_reads"].get(name, 0) + 1
        write(self.filename, self.value); self.event("attempted", str(source), member)
    def complete(self, source, counter=None, member=None):
        name = str(source) + (":" + member if member else "")
        self.value["completed_reads"][name] = self.value["completed_reads"].get(name, 0) + 1
        if counter is not None:
            self.value[counter] += 1
        write(self.filename, self.value)
        self.event("completed", str(source), member)


def array_schema(np, value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def checkpoint(guard, began, allowance):
    if guard:
        guard()
    require(time.monotonic() - began < allowance, "Input preparation remaining wall exhausted", UnsupportedInput)


def verify_source(source, ledger, guard):
    if guard:
        guard()
    stat_matches(source); ledger.attempt(source["path"], "byte_hash")
    actual = sha(source["path"])
    ledger.complete(source["path"], member="byte_hash")
    require(actual == source["sha256"], "Received source byte identity differs")


def csv_header(filename):
    with path(filename).open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    require(len(columns) == len(set(columns)), "Repeated CSV column names", UnsupportedInput)
    return columns


def feature_arrays(np, cKDTree, deck, vertices, ids, facts):
    """Exact original25 arithmetic; no original target/member is available."""
    xyz_mm = vertices.astype(np.float32)
    lower, upper = xyz_mm.min(0), xyz_mm.max(0)
    length = float((upper - lower).max())
    require(math.isfinite(length) and length > 0, "Zero/nonfinite geometry length", UnsupportedInput)
    xyz = ((xyz_mm - lower) / length).astype(np.float32)
    features = np.zeros((len(xyz), 25), dtype=np.float32)
    rows = {int(grid): row for row, grid in enumerate(ids)}
    supports = sorted(deck["rbe2"].values(), key=lambda group: group["reference"])
    require(len(supports) == 4, "Original25 requires four actual support interfaces", UnsupportedInput)
    for group in supports:
        member = np.asarray([rows[node] for node in group["members"]], np.int64)
        for dof in group["dofs"]:
            features[member, int(dof) - 1] = 1
    require(len(deck["rbe3"]) == 1, "Original25 requires one actual loaded interface", UnsupportedInput)
    loaded = next(iter(deck["rbe3"].values()))
    member = np.asarray([rows[node] for node in loaded["members"]], np.int64)
    features[member, 6] = 1
    force, moment = np.zeros(3), np.zeros(3)
    for action in deck["active_actions"]:
        (force if action["kind"] == "FORCE" else moment)[:] += action["vector"]
    features[:, 7:10] = force / 42300
    features[:, 10:13] = moment / 565000
    material = deck["materials"][facts["material"]["MID"]]
    require(material["E"] is not None and material["nu"] is not None
            and material["remaining_fields"] and material["remaining_fields"][0].strip(),
            "Original25 requires explicit actual MAT1 E/nu/density", UnsupportedInput)
    from geobc.elastic_pilot_data import _number
    rho = _number(material["remaining_fields"][0])
    require(rho is not None and rho > 0, "Actual MAT1 density absent/nonpositive", UnsupportedInput)
    features[:, 13:16] = [material["E"] / 113800, material["nu"], rho / 4.47e-9]
    reference = np.asarray(deck["grid"][loaded["reference"]]["xyz"])
    features[:, 16:19] = (xyz_mm - reference) / length
    for index, group in enumerate(supports):
        master = np.asarray(deck["grid"][group["reference"]]["xyz"])
        features[:, 19 + index] = np.linalg.norm(xyz_mm - master, axis=1) / length
    features[:, 23] = cKDTree(xyz_mm[member]).query(xyz_mm, workers=-1)[0] / length
    features[:, 24] = length / 200
    require(np.isfinite(features).all() and np.isfinite(xyz).all(), "Nonfinite original25", UnsupportedInput)
    return xyz, features, {"bbox_min_mm": lower.tolist(), "bbox_max_mm": upper.tolist(),
                          "characteristic_length_mm": length, "rho_t_per_mm3": rho,
                          "support_flag_source": "Actual RBE2 CM, matching original builder"}


def prepare_case(plan_path, key, *, contract=None, guard=None, budget_seconds=300, output=None, observer=None):
    began = time.monotonic()
    current, plan = request_contract(plan_path)
    require(contract is None or contract == current, "Requested input contract differs")
    contract = current; released(contract, plan); require(key in CASES, "Unfrozen blind case")
    require(0 < budget_seconds <= 300, "Input remaining budget is not bounded by300s")
    output = path(output if output is not None else path(plan.get("input_cache", path(plan["output"]) / "input_cache")) / key.replace("/", "_"))
    output.resolve().relative_to(ROOT.resolve())
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        previous = read(manifest_path)
        require(previous["contract"] == contract and previous["key"] == key, "Owned input cache contract differs")
        if previous["status"] == "completed":
            if previous["feasibility_pass"]:
                stat_matches(previous["file_identity"])
            return previous
    ledger = Ledger(output, contract, observer)
    record = {"kind": "blind_solver_target_free_input", "status": "running", "contract": contract,
              "key": key, "cad_id": key.split("/")[0], "feasibility_pass": False,
              "source": contract["records"][key], "costs": {}, "preparation_seconds": 0.0}
    write(manifest_path, record)
    def phase(name, function):
        checkpoint(guard, began, budget_seconds); tick = time.monotonic()
        value = function(); record["costs"][name] = time.monotonic() - tick
        checkpoint(guard, began, budget_seconds); return value
    try:
        import numpy as np
        import h5py
        from scipy.spatial import cKDTree
        from geobc import elastic_pilot_data as physical
        from geobc.continuous_query_geometry import scan_face_conformity
        from tools.mesh_boundary import quadratic_boundary_triangles
        for source in record["source"].values():
            phase("received_" + Path(source["path"]).suffix[1:] + "_byte_identity_seconds",
                  lambda source=source: verify_source(source, ledger, guard))
        def mesh():
            filename = record["source"]["FieldMesh"]["path"]
            ledger.attempt(filename)
            with h5py.File(path(filename), "r") as handle:
                values = {}
                for name in ("vertices", "cells"):
                    ledger.attempt(filename, name); values[name] = handle[name][:]
                    ledger.complete(filename, "input_member_reads", name)
            ledger.complete(filename, "raw_h5_input_reads")
            require(values["vertices"].ndim == 2 and values["vertices"].shape[1] == 3
                    and len(values["vertices"]) and values["vertices"].dtype.kind == "f"
                    and np.isfinite(values["vertices"]).all(), "Unsupported H5 vertices schema", UnsupportedInput)
            cells = values["cells"]
            require(cells.ndim == 2 and cells.shape[1] == 10 and len(cells) and cells.dtype.kind in "iu"
                    and cells.min() >= 0 and cells.max() < len(values["vertices"]), "Unsupported H5 TET10 schema", UnsupportedInput)
            return np.asarray(values["vertices"], np.float64), np.asarray(cells, np.int64)
        vertices, cells = phase("H5_input_seconds", mesh)
        def deck_parse():
            filename = record["source"]["SolverDeck"]["path"]; ledger.attempt(filename)
            try:
                value = physical._parse_deck(path(filename), contract["recipe"], began, budget_seconds)
            except (physical.UnsupportedInput, ValueError) as error:
                raise UnsupportedInput(str(error)) from error
            ledger.complete(filename, "raw_deck_input_reads"); return value
        deck = phase("deck_input_parse_seconds", deck_parse)
        def align():
            filename = record["source"]["Field"]["path"]; ledger.attempt(filename, "coordinate_columns")
            columns = csv_header(filename)
            require(all(name in columns for name in COORD_COLUMNS), "Required coordinate-only CSV columns missing", UnsupportedInput)
            values = np.loadtxt(path(filename), delimiter=",", skiprows=1,
                                usecols=[columns.index(name) for name in COORD_COLUMNS], dtype=np.float64, ndmin=2)
            ledger.complete(filename, "raw_CSV_coordinate_reads", "coordinate_columns")
            require(values.shape == (len(vertices), 4) and np.isfinite(values).all()
                    and (values[:, 0] > 0).all() and (values[:, 0] < 2**53).all()
                    and np.array_equal(values[:, 0], np.rint(values[:, 0])), "CSV coordinates/GRID identity schema differs", UnsupportedInput)
            csv_ids = values[:, 0].astype(np.int64)
            require(len(np.unique(csv_ids)) == len(vertices), "Repeated CSV GRID IDs", UnsupportedInput)
            distances, permutation = cKDTree(values[:, 1:]).query(vertices, workers=-1)
            require(len(np.unique(permutation)) == len(vertices) and distances.max() <= TOLERANCE_MM,
                    "CSV/H5 mapping is not one-to-one within0.0002mm", UnsupportedInput)
            ids = csv_ids[permutation]
            require(set(ids.tolist()) <= deck["grid"].keys(), "CSV GRID absent from actual deck", UnsupportedInput)
            deck_xyz = np.asarray([deck["grid"][int(node)]["xyz"] for node in ids], np.float64)
            grid_error = float(np.linalg.norm(deck_xyz - vertices, axis=1).max())
            require(grid_error <= TOLERANCE_MM, "GRID/H5 coordinate agreement exceeds0.0002mm", UnsupportedInput)
            refs = {row["reference"] for row in deck["rbe2"].values()} | {row["reference"] for row in deck["rbe3"].values()}
            require(set(deck["grid"]) - set(ids.tolist()) <= refs, "Unmapped non-reference GRID unsupported", UnsupportedInput)
            inverse = np.empty(len(vertices), np.int64); inverse[permutation] = np.arange(len(vertices))
            return ids, permutation.astype(np.int64), inverse, csv_ids, {
                "CSV_H5_max_distance_mm": float(distances.max()), "GRID_H5_max_distance_mm": grid_error,
                "one_to_one": True, "tolerance_mm": TOLERANCE_MM,
                "CSV_header_sha256": digest(columns), "GRID_minus1_used": False}
        ids, h5_to_csv, csv_to_h5, csv_ids, map_facts = phase("coordinate_GRID_mapping_seconds", align)
        def build_physics():
            cast = vertices.astype(np.float32); length = float((cast.max(0) - cast.min(0)).max())
            try:
                return physical._build(deck, vertices, cells, ids,
                    {"recipe": contract["recipe"], "expected_geometry": {"characteristic_length_mm": length}}, began, budget_seconds)
            except (physical.UnsupportedInput, ValueError) as error:
                raise UnsupportedInput(str(error)) from error
        arrays, facts = phase("physical_constraint_material_load_seconds", build_physics)
        xyz, features, normalization = phase("original25_feature_seconds",
            lambda: feature_arrays(np, cKDTree, deck, vertices, ids, facts))
        def cloud_arrays():
            conformity = scan_face_conformity(cells, len(vertices), max_records=8)
            require(conformity.get("status") == "passed", "TET10 shared-face/midside conformity fails", UnsupportedInput)
            try:
                faces = quadratic_boundary_triangles(cells, vertices.astype(np.float32))
            except ValueError as error:
                raise UnsupportedInput("Original quadratic boundary construction unsupported: " + str(error)) from error
            pool = np.unique(faces); count = min(1024, len(pool))
            seed = int.from_bytes(hashlib.blake2b(f"2026:{record['cad_id']}".encode(), digest_size=8).digest(), "little")
            chosen = np.random.default_rng(seed).choice(pool, count, replace=False)
            cloud = np.zeros((1, 1024, 3), np.float32); mask = np.zeros((1, 1024), bool)
            indices = np.full(1024, -1, np.int64)
            cloud[0, :count] = xyz[chosen]; mask[0, :count] = True; indices[:count] = chosen
            return faces, cloud, mask, indices, {"cloud_seed": 2026, "cloud_points": 1024,
                "boundary_pool_nodes": len(pool), "valid_cloud_points": count,
                "pool": contract["recipe"]["cloud_pool"], "sampling": contract["recipe"]["cloud_sampling"],
                "normalization": contract["recipe"]["cloud_normalization"],
                "face_conformity": {name: value for name, value in conformity.items() if name != "interior_faces"}}
        faces, cloud, mask, cloud_ids, cloud_facts = phase("boundary_cloud_seconds", cloud_arrays)
        arrays.update(xyz=xyz, features=features, faces=faces, cloud=cloud, mask=mask, cloud_node_ids=cloud_ids,
                      h5_row_to_csv=h5_to_csv, csv_row_to_h5=csv_to_h5, csv_node_ids=csv_ids)
        filename = output / "input.npz"
        def persist():
            temporary = filename.with_name("input.npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, **arrays)
            temporary.replace(filename)
            record.update(path=relative(filename), file_identity=identity(filename, True),
                          arrays={name: array_schema(np, value) for name, value in arrays.items()})
        phase("cache_persistence_identity_seconds", persist)
        facts.update(coordinate_maps=map_facts, normalization=normalization, cloud=cloud_facts,
                     rho_t_per_mm3=normalization["rho_t_per_mm3"], node_order="Original H5 vertices order",
                     physical_coordinates="Original H5 values as FP64; FNO coordinates/features preserve old FP32 arithmetic")
        record.update(status="completed", feasibility_pass=True, facts=facts,
                      region_array_keys=[name for name in arrays if name.startswith("region_")],
                      deferred_U_source={**record["source"]["Field"], "columns": U_COLUMNS, "S_reads": 0})
    except UnsupportedInput as error:
        record.update(status="completed", feasibility_pass=False, scientific_outcome="unsupported_input_or_budget", reason=str(error))
    except BaseException as error:
        record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["preparation_seconds"] = time.monotonic() - began
        record["read_receipt"] = ledger.value
        write(manifest_path, record)
    return record


def prepare_data(plan_path=PLAN):
    contract, plan = request_contract(plan_path); released(contract, plan)
    records = {key: prepare_case(plan_path, key, contract=contract) for key in CASES}
    manifest = {"kind": "blind_solver_target_free_inputs", "status": "completed", "contract": contract,
                "case_keys": CASES, "records": records, "feasibility_pass": all(item["feasibility_pass"] for item in records.values()),
                "preparation_seconds": sum(item["preparation_seconds"] for item in records.values())}
    write(path(plan.get("input_cache", path(plan["output"]) / "input_cache")) / "manifest.json", manifest)
    return manifest


def read_cache_metadata(plan_path=PLAN, *, contract=None):
    current, plan = request_contract(plan_path)
    require(contract is None or contract == current, "Input cache request contract differs")
    manifest = read(path(plan.get("input_cache", path(plan["output"]) / "input_cache")) / "manifest.json")
    require(manifest.get("status") == "completed" and manifest.get("contract") == current
            and manifest.get("case_keys") == CASES and set(manifest.get("records", {})) == set(CASES), "Input manifest metadata differs")
    for record in manifest["records"].values():
        if record.get("feasibility_pass"):
            stat_matches(record["file_identity"])
    return manifest


def load_input(record, device, for_fno=False, *, guard=None, observer=None):
    require(record.get("status") == "completed" and record.get("feasibility_pass") is True,
            "Unsupported/incomplete input; do not initialize operator/model")
    contract = record["contract"]; current, plan = request_contract(contract["plan"])
    require(current == contract, "Input record source contract differs"); released(contract, plan)
    import numpy as np
    import torch
    ledger = Ledger(path(record["path"]).parent, contract, observer)
    if guard: guard()
    stat_matches(record["file_identity"]); ledger.attempt(record["path"], "byte_hash")
    actual = sha(record["path"])
    ledger.complete(record["path"], member="byte_hash")
    require(actual == record["file_identity"]["sha256"], "Input cache byte identity differs")
    names = ["xyz", "features", "cloud", "mask"] if for_fno else [
        name for name in record["arrays"] if name not in {"xyz", "features", "faces", "cloud", "mask", "cloud_node_ids", "h5_row_to_csv", "csv_row_to_h5", "csv_node_ids"}]
    result, regions = {}, {}
    ledger.attempt(record["path"])
    with np.load(path(record["path"]), allow_pickle=False) as stored:
        ledger.complete(record["path"], "input_cache_reads")
        require(set(stored.files) == set(record["arrays"]), "Input cache member schema differs")
        for name in names:
            if guard: guard()
            ledger.attempt(record["path"], name); value = stored[name]
            ledger.complete(record["path"], "input_member_reads", name)
            require(array_schema(np, value) == record["arrays"][name], "Input array identity differs: " + name)
            tensor = torch.from_numpy(value.copy()).to(device=device)
            if name.startswith("region_"):
                regions[name[7:]] = tensor
            else:
                result[name] = tensor
    if for_fno:
        from geobc.solver_selection_data import _stencil, _splat
        indices, weights = _stencil(result["xyz"], 64)
        raster, active = _splat(result["features"], indices, weights, 64)
        axis = torch.linspace(0, 1, 64, device=device)
        xyz_grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
        result.update(query_indices=indices, query_weights=weights,
                      grid=torch.cat((active.float().reshape(1,64,64,64), xyz_grid,
                                      raster.T.reshape(25,64,64,64)), dim=0).unsqueeze(0))
    else:
        result["regions"] = regions
    if guard: guard()
    result.update(key=record["key"], manifest=record, _actual_read_receipt=ledger.value)
    return result


def validate_barrier(barrier, contract):
    require(barrier.get("passed") is True and barrier.get("case_keys") == CASES and barrier.get("arms") == ARMS
            and barrier.get("all10_node_U_predictions_committed") is True and barrier.get("all5_native_complete") is True
            and barrier.get("input_contract_sha256") == digest(contract), "Real all-ten prediction barrier missing")
    nodes = barrier.get("node_prediction_manifests", {}); native = barrier.get("native_prediction_manifests", {})
    require(set(nodes) == {arm + ":" + key for arm in ARMS for key in CASES}
            and set(native) == set(CASES), "Prediction manifest barrier inventory incomplete")
    for record in list(nodes.values()) + list(native.values()):
        require(sha(record["path"]) == record["sha256"], "Committed prediction manifest identity differs")


def read_U(record, barrier, *, contract, output, guard=None, observer=None):
    """Shared once-only source U parse; never materialize any S CSV column."""
    current, plan = request_contract(contract["plan"])
    require(contract == current and record["contract"] == contract and record["feasibility_pass"] is True, "Deferred input identity differs")
    released(contract, plan); validate_barrier(barrier, contract)
    output = path(output); output.resolve().relative_to(ROOT.resolve())
    ledger = Ledger(output, contract, observer)
    cache_manifest = output / "manifest.json"; claim = output / "source_U_read_started.json"
    import numpy as np
    if cache_manifest.is_file():
        owned = read(cache_manifest)
        require(owned.get("status") == "completed" and owned["input_contract_sha256"] == digest(contract)
                and owned["barrier_sha256"] == digest(barrier) and owned["key"] == record["key"], "Owned U cache differs")
        stat_matches(owned["file_identity"])
        require(sha(owned["path"]) == owned["file_identity"]["sha256"], "Owned U cache byte identity differs")
        ledger.attempt(owned["path"]); value = np.load(path(owned["path"]), allow_pickle=False)
        ledger.complete(owned["path"], "owned_GT_cache_reads")
        require(array_schema(np, value) == owned["array"], "Owned U cache array differs")
        return value
    require(not claim.exists(), "Source U parse previously started without committed owned cache", UncommittedTargetRead)
    source = record["deferred_U_source"]; verify_source(source, ledger, guard)
    columns = csv_header(source["path"])
    require(all(name in columns for name in U_COLUMNS), "Fixed ver U columns missing; never substitute another load", UnsupportedInput)
    if guard: guard()
    write(claim, {"key": record["key"], "input_contract_sha256": digest(contract), "barrier_sha256": digest(barrier),
                  "columns": U_COLUMNS, "S_reads": 0, "source": source, "completed": False})
    ledger.value["source_U_target_parse_attempts"] += 1; ledger.attempt(source["path"], "U_only_columns")
    value = np.loadtxt(path(source["path"]), delimiter=",", skiprows=1,
                       usecols=[columns.index(name) for name in U_COLUMNS], dtype=np.float64, ndmin=2)
    ledger.complete(source["path"], "source_U_target_parses", "U_only_columns")
    require(value.shape == (record["facts"]["node_count"], 3) and np.isfinite(value).all(), "Original U target schema/nonfinite differs")
    ledger.attempt(record["path"])
    ledger.attempt(record["path"], "h5_row_to_csv")
    with np.load(path(record["path"]), allow_pickle=False) as stored:
        ledger.complete(record["path"], "input_cache_reads")
        permutation = stored["h5_row_to_csv"]
    ledger.complete(record["path"], "input_member_reads", "h5_row_to_csv")
    require(array_schema(np, permutation) == record["arrays"]["h5_row_to_csv"], "Frozen target alignment permutation differs")
    value = np.ascontiguousarray(value[permutation], dtype=np.float64)
    filename = output / "U_mm.npy"; temporary = output / "U_mm.npy.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(filename)
    owned = {"status": "completed", "key": record["key"], "path": relative(filename), "file_identity": identity(filename, True),
             "array": array_schema(np, value), "input_contract_sha256": digest(contract), "barrier_sha256": digest(barrier),
             "source": source, "alignment": ALIGNMENT, "S_reads": 0, "original_source_U_parse_count": 1}
    write(cache_manifest, owned); write(claim, {**read(claim), "completed": True})
    if guard: guard()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=PLAN)
    args = parser.parse_args()
    prepare_data(args.plan)


if __name__ == "__main__":
    main()
