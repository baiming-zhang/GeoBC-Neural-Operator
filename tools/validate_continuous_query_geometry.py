"""One frozen TRAIN101_479 input-only CPU geometry/query acceptance gate.

Metadata preflight imports only the standard library. Runtime reads exactly
the declared input members, scans one CAD's face IDs/bounds, and performs
bounded numerical tests on at most32 original cells and512 query occurrences.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "16169bbd20fc1d746cc8ccf88fa9d4c287fa410744d4ee8b6f0bb210543bb71f"
POLICY_PATH = "notes/continuous_query_geometry_implementation_policy.json"
POLICY_SHA256 = "13dd1a57dd368a9f44e74841408a30d6ac95da335fcba97a9cc1825e4eaf446f"
MODULE_PATH = "geobc/continuous_query_geometry.py"
MODULE_READY = "notes/continuous_query_geometry_module_ready.json"
MODULE_SHA256 = "95599c0001c2f50f1cabd5f2e74d91396f71072378a77b31763ee086e6a79c27"
MODULE_READY_SHA256 = "036c7aee66149d0c2070848996bcc54d9c25dd55feb314bfbb662e75197d5063"
TOOL_PATH = "tools/validate_continuous_query_geometry.py"
CAD, CASE = "101_479", "101_479/ver"
ORIGINAL_NAMES = ([f"support_reference_constrained_dof_{i}" for i in range(1, 7)]
    + ["loaded_interface_mask"] + [f"total_force_{a}_over_42300N" for a in "xyz"]
    + [f"total_moment_{a}_over_565000Nmm" for a in "xyz"]
    + ["E_over_113800MPa", "poisson_ratio", "density_over_4.47e-9_t_per_mm3"]
    + [f"offset_from_load_reference_{a}_over_length" for a in "xyz"]
    + [f"distance_to_support_reference_{i}_over_length" for i in range(1, 5)]
    + ["distance_to_loaded_interface_over_length", "characteristic_length_over_200mm"])
LOCAL_NAMES = ["nearest_boundary_offset_x_over_0.05", "nearest_boundary_offset_y_over_0.05",
    "nearest_boundary_offset_z_over_0.05", "nearest_boundary_distance_over_0.05",
    "neighbor_cov_xx_over_trace", "neighbor_cov_yy_over_trace", "neighbor_cov_zz_over_trace",
    "neighbor_cov_xy_over_trace", "neighbor_cov_xz_over_trace", "neighbor_cov_yz_over_trace",
    "neighbor_eigen_min_over_trace", "neighbor_eigen_mid_over_trace", "kth_boundary_distance_over_0.05"]
EUCLIDEAN_NAMES = [f"support_member_{i}_distance_over_length" for i in range(1, 5)] + ["loaded_member_distance_over_length"]


class SourceIdentityError(ValueError):
    pass


class InputContractError(ValueError):
    pass


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and re.match(r"^/mnt/[a-zA-Z]/", value):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:/", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Geometry-gate paths must remain inside this project")
    return path


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8"))


def identity(path, *, sha=False):
    path = host_path(path); stat = path.stat()
    result = {"path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if sha: result["sha256"] = sha_bytes(path.read_bytes())
    return result


def require(condition, message, error=InputContractError):
    if not condition: raise error(message)


def verify_stat(path, expected):
    actual = identity(path)
    require(actual["bytes"] == expected["bytes"] and actual["mtime_ns"] == expected["mtime_ns"],
            f"Frozen input file stat changed: {path}", SourceIdentityError)
    return actual


def write_json(path, record):
    path = host_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def request_contract(plan_path=ROOT / "notes/geobc_continuous_geometry_plan.json"):
    """Stable coordinator identity: standard-library JSON/source/hash/stat only."""
    plan_path = host_path(plan_path); raw = plan_path.read_bytes(); plan = json.loads(raw)
    require(sha_bytes(raw) == PLAN_SHA256 and plan["status"] == "predeclared", "Frozen geometry plan changed", SourceIdentityError)
    policy_raw = host_path(POLICY_PATH).read_bytes(); policy = json.loads(policy_raw)
    require(sha_bytes(policy_raw) == POLICY_SHA256 and policy["formal_plan_sha256"] == PLAN_SHA256,
            "Frozen root implementation policy changed", SourceIdentityError)
    fixed, scope = plan["root_fixed_numeric_contract"], plan["fixed_future_scope"]
    require(fixed["CAD"] == CAD and fixed["canonical_case"] == CASE and scope["CAD"] == CAD
            and fixed["maximum_original_cells"] == 32 and fixed["maximum_faces"] == 8 and fixed["maximum_query_occurrences"] == 512
            and fixed["CPU_workers"] == fixed["BLAS_threads"] == 1, "One-CAD CPU bounds changed")
    frozen_json, frozen_sources = {}, {}
    for item in plan["verified_metadata_and_source_identities"]:
        path = host_path(item["path"]); raw = path.read_bytes()
        require(sha_bytes(raw) == item["sha256"], f"Frozen metadata/source changed: {item['path']}", SourceIdentityError)
        if path.suffix == ".json": frozen_json[item["path"]] = json.loads(raw)
        else: frozen_sources[item["path"]] = item["sha256"]
    ancestors = [(plan["proposal_source"]["path"], plan["proposal_source"]["sha256"]),
        (plan["parent"]["frozen_query_contract"], plan["parent"]["frozen_query_contract_sha256"]),
        (plan["parent"]["matched_pilot_result"], plan["parent"]["matched_pilot_result_sha256"])]
    for path, expected in ancestors:
        require(sha_bytes(host_path(path).read_bytes()) == expected, f"Frozen ancestor changed: {path}", SourceIdentityError)
    split = frozen_json["notes/geobc_split.json"]
    require(CAD in split["groups"]["train"] and CAD not in split["groups"]["validation"] and CAD not in split["groups"]["test"], "Geometry CAD must remain TRAIN only")
    physical = plan["physical_XYZ_path_closed_at_metadata_level"]
    grad = frozen_json["data/geobc_gradient_supervision/manifest.json"]["CADs"][CAD]
    graph = frozen_json["data/geobc_graph_context/manifest.json"]["CADs"][CAD]
    local = next(v for v in frozen_json["data/geobc_local_geometry/manifest.json"]["cases"] if v["cad_id"] == CAD)
    hashes = physical["inherited_array_identity"]
    require(grad["xyz_sha256"] == graph["xyz_sha256"] == local["xyz_sha256"] == hashes["canonical_normalized_xyz_float32_sha256"]
            and graph["vertices_mm_sha256"] == hashes["vertices_float32_le_Corder_sha256"]
            and graph["cells_sha256"] == hashes["cells10_int64_le_Corder_sha256"]
            and grad["grid_node_ids_sha256"] == hashes["grid_node_ids_int64_le_Corder_sha256"], "Inherited one-CAD array/order identity differs")
    # The normalization object also includes an explanatory future check;
    # compare only the two physical numerical fields below.
    normalization = {k: physical["normalization"][k] for k in ("bbox_min_mm", "characteristic_length_mm")}
    require(graph["normalization"] == normalization and all(grad["normalization"][k] == v for k, v in normalization.items()), "Physical normalization identity differs")
    files = []
    for item in plan["future_explicit_member_allowlist"]:
        expected = physical["stat_observed"] if item["path"].endswith(".h5") else item
        files.append({**verify_stat(item["path"], expected), "array_members": item["array_members"],
                      "metadata_members": item.get("metadata_members", [])})
    h5_path = files[0]["path"]
    require(host_path(grad["raw_geometry_source"]["path"]) == host_path(h5_path)
            and host_path(graph["input_sources"]["H5"]["path"]) == host_path(h5_path), "Physical vertex source path differs")
    verify_stat(h5_path, grad["raw_geometry_source"]); verify_stat(h5_path, graph["input_sources"]["H5"])
    module_ready = read_json(MODULE_READY)
    module_sha = sha_bytes(host_path(MODULE_PATH).read_bytes())
    ready_sha = module_ready["source_sha256"]
    if isinstance(ready_sha, dict): ready_sha = ready_sha[MODULE_PATH]
    require(module_ready["status"] == "source_ready_author_synthetic_fixture_passed" and module_ready["formal_plan_sha256"] == PLAN_SHA256
            and module_ready["implementation_policy_sha256"] == POLICY_SHA256 and ready_sha == module_sha == MODULE_SHA256
            and sha_bytes(host_path(MODULE_READY).read_bytes()) == MODULE_READY_SHA256,
            "CPU geometry module/source must match its ready record", SourceIdentityError)
    return {"schema_version": 1, "stage": plan["stage"], "plan": identity(plan_path, sha=True), "plan_sha256": PLAN_SHA256,
        "implementation_policy": identity(POLICY_PATH, sha=True), "implementation_policy_sha256": POLICY_SHA256,
        "source_sha256": {**frozen_sources, MODULE_PATH: module_sha, TOOL_PATH: sha_bytes(host_path(TOOL_PATH).read_bytes())},
        "module_ready": identity(MODULE_READY, sha=True), "metadata_sha256": {v["path"]: v["sha256"] for v in plan["verified_metadata_and_source_identities"] if v["path"].endswith(".json")},
        "ancestor_sha256": dict(ancestors), "input_files": files, "array_hashes": hashes, "normalization": normalization,
        "CAD": CAD, "canonical_case": CASE, "node_count": scope["node_count_metadata"], "cell_count": scope["cell_count_metadata"],
        "numeric_contract": fixed, "same_element_reference_root_equivalence_tolerance": policy["same_element_reference_root_equivalence_tolerance"],
        "interior_reference_points": plan["future_gate_one_fixed_recipe"]["interior_reference_points"],
        "synthetic_failures": plan["fixed_synthetic_failures"], "source_feature_names": ORIGINAL_NAMES + LOCAL_NAMES + EUCLIDEAN_NAMES,
        "euclidean_sha256": graph["euclidean_features_sha256"], "result": plan["module_interface"]["result"],
        "default_output": plan["module_interface"]["run"], "scientific_limits": plan["scientific_limits"],
        "physical_geometry": "verified sourcefloat32 H5 vertices, then float64; normalized cached xyz is never the physical map",
        "query_counting": "scheduled physical occurrences including duplicates; reference/Jacobian/Newton evaluations separately",
        "labels_read": 0, "weights_read": 0, "model_calls": 0, "CUDA_initialized": False}


def preflight(plan_path=ROOT / "notes/geobc_continuous_geometry_plan.json"):
    began = time.monotonic(); contract = request_contract(plan_path)
    return {"status": "ready", "plan_sha256": PLAN_SHA256, "policy_sha256": POLICY_SHA256,
        "contract_sha256": sha_bytes(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()),
        "source_sha256": contract["source_sha256"], "CAD": CAD, "array_members_opened": 0, "H5_datasets_read": [],
        "NPZ_members_opened": [], "labels_read": 0, "weights_read": 0, "models_constructed": 0, "model_calls": 0,
        "CUDA_initialized": False, "scientific_dependencies_imported": False, "elapsed_seconds": time.monotonic() - began}


def _array_sha(array, dtype):
    import numpy as np
    return sha_bytes(np.ascontiguousarray(array, dtype=dtype).tobytes(order="C"))


def _jsonable(value):
    import numpy as np
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return _jsonable(value.tolist())
    if isinstance(value, np.generic): return _jsonable(value.item())
    if isinstance(value, float) and not __import__("math").isfinite(value): return None
    return value


def _load_inputs(contract, accessed):
    """No generic gradient/local/native loaders; every member is allowlisted."""
    import h5py
    import numpy as np
    inputs = {v["path"]: v for v in contract["input_files"]}
    def npz_members(path, names):
        item = inputs[path]
        permitted = set(item["array_members"]) | set(item["metadata_members"])
        require(set(names).issubset(permitted), "Attempted NPZ member outside frozen allowlist")
        verify_stat(path, item)
        result = {}
        with np.load(host_path(path), allow_pickle=False) as archive:
            for name in names:
                result[name] = archive[name]
                accessed.setdefault(path, []).append(name)
        return result
    h5_path = "data/raw/FieldMesh/101_479.h5"
    verify_stat(h5_path, inputs[h5_path])
    with h5py.File(host_path(h5_path), "r") as stored:
        require(sorted(stored.keys()) == ["cells", "faces", "nodal_variables", "vertices"], "H5 top-level schema differs")
        dataset = stored["vertices"]
        require(dataset.dtype == np.dtype("float32") and dataset.shape == (contract["node_count"], 3), "Original physical vertices dtype/shape differs")
        vertices = dataset[...]
        accessed[h5_path] = ["vertices"]
    require(np.isfinite(vertices).all(), "Nonfinite source vertices")
    hashes = contract["array_hashes"]
    require(_array_sha(vertices, "<f4") == hashes["vertices_float32_le_Corder_sha256"], "Source vertices hash differs", SourceIdentityError)
    # No other scientific member is opened until physical vertices are verified.
    physical = vertices.astype(np.float64)
    gradient = npz_members("data/geobc_gradient_supervision/101_479.npz", ["cells10"])
    prepared_names = inputs["data/prepared_geobc_multiload/101_479_ver.npz"]["array_members"] + inputs["data/prepared_geobc_multiload/101_479_ver.npz"]["metadata_members"]
    prepared = npz_members("data/prepared_geobc_multiload/101_479_ver.npz", prepared_names)
    local_names = inputs["data/geobc_local_geometry/101_479.npz"]["array_members"] + inputs["data/geobc_local_geometry/101_479.npz"]["metadata_members"]
    local = npz_members("data/geobc_local_geometry/101_479.npz", local_names)
    graph_names = inputs["data/geobc_graph_context/101_479.npz"]["array_members"] + inputs["data/geobc_graph_context/101_479.npz"]["metadata_members"]
    graph = npz_members("data/geobc_graph_context/101_479.npz", graph_names)
    xyz, original, grid_ids, cells = (prepared["xyz"], prepared["features"], prepared["grid_node_ids"], gradient["cells10"])
    n, m = contract["node_count"], contract["cell_count"]
    require(xyz.dtype == original.dtype == np.dtype("float32") and xyz.shape == (n, 3) and original.shape == (n, 25)
            and cells.dtype == grid_ids.dtype == np.dtype("int64") and cells.shape == (m, 10) and grid_ids.shape == (n,), "Canonical input/connectivity dtype or shape differs")
    require(np.isfinite(xyz).all() and np.isfinite(original).all() and (cells >= 0).all() and (cells < n).all()
            and (np.diff(np.sort(cells, axis=1), axis=1) > 0).all() and len(np.unique(grid_ids)) == n, "Corrupt finite input or original node-index connectivity")
    require(_array_sha(cells, "<i8") == hashes["cells10_int64_le_Corder_sha256"] and _array_sha(grid_ids, "<i8") == hashes["grid_node_ids_int64_le_Corder_sha256"]
            and _array_sha(xyz, "<f4") == hashes["canonical_normalized_xyz_float32_sha256"], "Canonical cells/GRID/xyz identity differs", SourceIdentityError)
    scalar = lambda record, name: np.asarray(record[name]).item()
    require(scalar(prepared, "sample_key") == CASE and scalar(prepared, "label_alignment") == "official CSV GRID coordinates -> H5 vertices"
            and prepared["feature_names"].tolist() == ORIGINAL_NAMES, "Prepared case/alignment/original25 schema differs")
    normalization = contract["normalization"]
    origin = np.asarray(normalization["bbox_min_mm"], np.float32); length = float(normalization["characteristic_length_mm"])
    require(np.array_equal(prepared["bbox_min_mm"], origin) and float(scalar(prepared, "characteristic_length_mm")) == length, "Prepared physical normalization differs")
    reproduced = ((vertices - origin) / length).astype(np.float32)
    require(np.array_equal(reproduced, xyz) and _array_sha(reproduced, "<f4") == hashes["canonical_normalized_xyz_float32_sha256"], "Source float32 canonical xyz reproduction is not exact", SourceIdentityError)
    require(local["features"].dtype == graph["euclidean"].dtype == np.dtype("float32") and local["features"].shape == (n, 13)
            and graph["euclidean"].shape == (n, 5) and np.isfinite(local["features"]).all() and np.isfinite(graph["euclidean"]).all()
            and (graph["euclidean"] >= 0).all(), "Finite local13/Euclidean5 schema differs")
    require(scalar(local, "cad_id") == scalar(graph, "cad_id") == CAD and scalar(local, "canonical_case") == CASE
            and int(scalar(local, "node_count")) == int(scalar(graph, "node_count")) == n
            and scalar(local, "xyz_sha256") == scalar(graph, "xyz_sha256") == hashes["canonical_normalized_xyz_float32_sha256"]
            and local["feature_names"].tolist() == LOCAL_NAMES and graph["feature_names"].tolist() == EUCLIDEAN_NAMES
            and int(scalar(local, "K")) == 32 and float(scalar(local, "geometry_scale")) == .05 and int(scalar(local, "schema_version")) == 1
            and bool(scalar(local, "target_independent")) and bool(scalar(graph, "target_independent"))
            and float(scalar(graph, "characteristic_length_mm")) == length, "Local/Euclidean input identity/order/scales differ")
    require(_array_sha(graph["euclidean"], "<f4") == contract["euclidean_sha256"], "Euclidean5 hash differs", SourceIdentityError)
    masked = original.copy(); masked[:, 7:13] = 0
    raw43 = np.concatenate((masked, local["features"], graph["euclidean"]), axis=1)
    require(raw43.shape == (n, 43) and (raw43[:, 7:13] == 0).all() and np.isin(raw43[:, :7], [0, 1]).all(), "Masked original43 role/input contract differs")
    receipt = {"vertices_float32_sha256": _array_sha(vertices, "<f4"), "cells10_int64_sha256": _array_sha(cells, "<i8"),
        "grid_node_ids_int64_sha256": _array_sha(grid_ids, "<i8"), "canonical_xyz_float32_sha256": _array_sha(xyz, "<f4"),
        "raw43_float32_sha256": _array_sha(raw43, "<f4"), "local13_float32_sha256": _array_sha(local["features"], "<f4"),
        "Euclidean5_float32_sha256": _array_sha(graph["euclidean"], "<f4"), "source_float32_reproduction_exact": True,
        "node_count": n, "cell_count": m, "geometry_dtype": "float64 promoted only after sourcefloat32 hash verification",
        "raw43_feature_names": ORIGINAL_NAMES + LOCAL_NAMES + EUCLIDEAN_NAMES, "new_fit_clip_or_reclassification": False}
    return physical, cells, xyz, raw43, receipt


def _geometry_gate(contract, physical, cells, xyz, raw43):
    import numpy as np
    from geobc import continuous_query_geometry as geometry
    fixed = contract["numeric_contract"]
    checks, failures = [], []
    arithmetic_counts = {"shape_reference_evaluations": 0, "physical_map_reference_evaluations": 0,
                         "descriptor_reference_evaluations": 0, "jacobian_reference_evaluations": 0}
    def evaluation_count(L):
        values = np.asarray(L)
        return 1 if values.ndim == 1 else len(values)
    def shape(L):
        arithmetic_counts["shape_reference_evaluations"] += evaluation_count(L)
        return geometry.quadratic_shape(L)
    def map_point(nodes, L):
        count = evaluation_count(L)
        arithmetic_counts["physical_map_reference_evaluations"] += count
        arithmetic_counts["shape_reference_evaluations"] += count
        return geometry.map_points(nodes, L)
    def interpolate(values, connectivity, rows, L, dtype):
        count = evaluation_count(L)
        arithmetic_counts["descriptor_reference_evaluations"] += count
        arithmetic_counts["shape_reference_evaluations"] += count
        return geometry.interpolate_nodal(values, connectivity, rows, L, dtype=dtype)
    def jacobian(nodes, L):
        arithmetic_counts["jacobian_reference_evaluations"] += evaluation_count(L)
        return geometry.jacobian(nodes, L)
    def check(name, passed, **details):
        record = {"name": name, "passed": bool(passed), **_jsonable(details)}
        checks.append(record)
        if not passed: failures.append(record)
    def block_error(value, expected):
        value, expected = np.asarray(value, np.float64), np.asarray(expected, np.float64)
        if not np.isfinite(value).all() or not np.isfinite(expected).all(): return float("inf")
        return float(np.max(np.abs(value - expected)) / max(1., float(np.max(np.abs(expected)))))
    def coordinate_error(value, expected):
        return float(np.max(np.abs(np.asarray(value, np.float64) - np.asarray(expected, np.float64))) / length)
    length = float(contract["normalization"]["characteristic_length_mm"])
    origin = np.asarray(contract["normalization"]["bbox_min_mm"], np.float64)
    scan = geometry.scan_face_conformity(cells, len(physical))
    selected_rows, faces = geometry.select_cells_and_faces(scan, len(cells), max_cells=32, max_faces=8)
    selected_rows = np.asarray(sorted(int(row) for row in selected_rows), dtype=np.int64)
    require(len(selected_rows) <= 32 and len(faces) <= 8 and len(np.unique(selected_rows)) == len(selected_rows), "Module exceeded deterministic cell/face selection bounds")
    check("whole_CAD_face_ID_conformity", scan["status"] == "passed", scan=scan)
    if scan["status"] != "passed":
        require(len(selected_rows) == len(faces) == 0, "Nonconforming topology must produce empty numeric selection")
        return _jsonable({"gate_pass": False, "checks": checks, "scientific_failures": failures,
            "conformity": scan, "selected_original_rows": [], "selected_shared_faces": [], "query_results": [],
            "counts": {"selected_original_cells": 0, "selected_shared_faces": 0, "logical_query_occurrences": 0,
                "unique_physical_points_exact_float64": 0, "by_category": {}, "synthetic_cells": 0,
                "all_CAD_face_scan_cells": len(cells), "all_CAD_bounds_cell_evaluations": 0,
                "additional_known_reference_J_evaluations": 0, "locator_counters": {}, "synthetic_locator_counters": []},
            "unexecuted_reason": "Mandatory whole-CAD face-ID conformity failed; root stop policy forbids repair/substitution or numeric continuation",
            "unproved_properties": contract["scientific_limits"]})
    require(len(selected_rows) > 0, "Conforming nonempty source mesh yielded no selected original cells")
    # This is the only whole-CAD numerical geometry scan: conservative bounds.
    # No whole-CAD Jacobian/Newton pass is performed.
    lower = np.full(3, np.inf); upper = np.full(3, -np.inf)
    bounds_cell_evaluations = 0
    for start in range(0, len(cells), 4096):
        lo, hi = geometry.bezier_bounds(physical[cells[start:start + 4096]])
        lower = np.minimum(lower, np.asarray(lo).min(0)); upper = np.maximum(upper, np.asarray(hi).max(0))
        bounds_cell_evaluations += len(lo)
    require(np.isfinite(lower).all() and np.isfinite(upper).all() and (upper >= lower).all(), "Nonfinite conservative global bounds")
    node_ids = np.arange(len(physical), dtype=np.int64)
    q_node = geometry.post_ln_surrogate(node_ids)
    require(q_node.dtype == np.dtype("float64") and q_node.shape == (len(physical), 32) and np.isfinite(q_node).all(), "Fixed post-LN/tanh Q32 surrogate differs")
    blocks = {"raw43": raw43, "Q32": q_node, "cachedxyz": xyz}
    blocks64 = {name: np.asarray(values, dtype=np.float64) for name, values in blocks.items()}
    blocks32 = {name: np.asarray(values, dtype=np.float32) for name, values in blocks.items()}
    physical32 = physical.astype(np.float32)
    locator_config = {**fixed, "same_element_reference_root_equivalence_tolerance": contract["same_element_reference_root_equivalence_tolerance"]}
    locator = geometry.GeometryLocator(physical, cells[selected_rows], selected_rows, length,
        config=locator_config, global_bounds=(lower, upper), descriptor_blocks=blocks)
    reference_nodes = np.asarray(geometry.REFERENCE_BARYCENTRIC, np.float64)
    require(reference_nodes.shape == (10, 4), "Frozen full10 reference node order differs")
    node_weights = shape(reference_nodes)
    onehot_error = block_error(node_weights, np.eye(10))
    check("all10_nodal_shape_onehot", onehot_error <= fixed["shape_partition_and_direct_replay_relative_tolerance"], maximum_error=onehot_error)
    spike = np.concatenate((np.zeros(4), np.ones(6)))
    spike_replay = node_weights @ spike
    check("independent_midside_spike_replay_and_corner_P1_rejection", block_error(spike_replay, spike) <= 1e-12
          and np.array_equal(spike_replay[:4], np.zeros(4)) and np.all(spike_replay[4:] == 1),
          full10_replay=spike_replay, corner_P1_midside_values=[0] * 6, corner_P1_midside_maximum_error=1.)
    schedule = []
    for row in selected_rows:
        for local_id, barycentric in enumerate(reference_nodes):
            schedule.append({"category": "local_node", "source_row": int(row), "local_node": local_id,
                "original_node_id": int(cells[row, local_id]), "L": barycentric.copy(),
                "x": map_point(physical[cells[row]], barycentric)})
    interior = np.asarray(contract["interior_reference_points"], np.float64)
    interior_budget = min(124, 4 * len(selected_rows))
    count = 0
    for row in selected_rows:
        for barycentric in interior:
            if count >= interior_budget: break
            schedule.append({"category": "interior", "source_row": int(row), "L": barycentric.copy(),
                "x": map_point(physical[cells[row]], barycentric)})
            count += 1
    face_points = np.asarray([[.5, .5, 0], [.5, 0, .5], [0, .5, .5], [1/3, 1/3, 1/3],
                             [.2, .3, .5], [.3, .5, .2], [.5, .2, .3]], np.float64)
    face_reports, additional_reference_J = [], []
    max_face64 = {"physical_mm_over_L": 0., **{name: 0. for name in blocks}}
    max_face32 = dict(max_face64)
    max_face32_vs64 = {name: 0. for name in blocks}
    def face_L(row, corners, face_barycentric):
        L = np.zeros(4, np.float64)
        for node, value in zip(corners, face_barycentric):
            hits = np.nonzero(cells[row, :4] == node)[0]
            require(len(hits) == 1, "Selected face corner identity not found in incident cell")
            L[int(hits[0])] = value
        return L
    for face in faces:
        rows = [int(row) for row in face["cell_rows"]]
        require(len(rows) == 2 and all(row in selected_rows for row in rows), "Selected paired face cells missing from bounded submesh")
        corners = [int(v) for v in face["corner_ids"]]
        report = {"corner_ids": corners, "cell_rows": rows, "points": []}
        for point_number, face_barycentric in enumerate(face_points):
            Ls = [face_L(row, corners, face_barycentric) for row in rows]
            X = [map_point(physical[cells[row]], L) for row, L in zip(rows, Ls)]
            e64, e32, e32_vs64 = {"physical_mm_over_L": coordinate_error(X[0], X[1])}, {}, {}
            X32 = [interpolate(physical32, cells, np.asarray([row]), L[None], dtype=np.float32)[0] for row, L in zip(rows, Ls)]
            e32["physical_mm_over_L"] = coordinate_error(X32[0], X32[1])
            for name, values in blocks.items():
                values64 = [interpolate(blocks64[name], cells, np.asarray([row]), L[None], dtype=np.float64)[0] for row, L in zip(rows, Ls)]
                values32 = [interpolate(blocks32[name], cells, np.asarray([row]), L[None], dtype=np.float32)[0] for row, L in zip(rows, Ls)]
                e64[name] = block_error(values64[0], values64[1]); e32[name] = block_error(values32[0], values32[1])
                e32_vs64[name] = max(block_error(values32[i], values64[i]) for i in (0, 1))
            for name in max_face64:
                max_face64[name] = max(max_face64[name], e64[name]); max_face32[name] = max(max_face32[name], e32[name])
            for name in max_face32_vs64: max_face32_vs64[name] = max(max_face32_vs64[name], e32_vs64[name])
            report["points"].append({"face_barycentric": face_barycentric.tolist(), "direct_FP64_trace_errors": e64,
                                      "direct_FP32_trace_errors": e32, "FP32_vs_FP64_errors": e32_vs64})
            schedule.append({"category": "shared_face", "source_row": rows[0], "L": Ls[0], "x": X[0],
                "face_corner_ids": corners, "face_cell_rows": rows, "face_point_number": point_number})
            additional_reference_J.extend((row, L) for row, L in zip(rows, Ls))
        face_reports.append(report)
    check("direct_FP64_paired_face_traces", max(max_face64.values()) <= fixed["direct_shared_face_trace_relative_tolerance"],
          maximum_normalized_errors=max_face64, face_occurrences=7 * len(faces))
    check("signed_FP32_paired_face_traces", max(max_face32.values()) <= 2e-6,
          maximum_normalized_errors=max_face32, FP32_vs_FP64_extension_errors=max_face32_vs64,
          claim="CPU adapter arithmetic only; native FP16 parity untested")
    inside_count = len(schedule)
    rows = np.asarray([item["source_row"] for item in schedule], np.int64)
    Ls = np.stack([item["L"] for item in schedule])
    points = np.stack([item["x"] for item in schedule])
    partition_error = float(np.abs(shape(Ls).sum(-1) - 1).max())
    check("partition_unity_all_inside_reference_occurrences", partition_error <= fixed["shape_partition_and_direct_replay_relative_tolerance"], maximum_error=partition_error)
    direct64, direct32 = {}, {}
    node_mask = np.asarray([item["category"] == "local_node" for item in schedule])
    scheduled_ids = np.asarray([item.get("original_node_id", 0) for item in schedule], np.int64)
    node_errors64, node_errors32, precision_errors = {}, {}, {}
    for name, values in blocks.items():
        direct64[name] = interpolate(blocks64[name], cells, rows, Ls, dtype=np.float64)
        direct32[name] = interpolate(blocks32[name], cells, rows, Ls, dtype=np.float32)
        expected = values[scheduled_ids[node_mask]]
        node_errors64[name] = block_error(direct64[name][node_mask], expected)
        node_errors32[name] = block_error(direct32[name][node_mask], expected.astype(np.float32))
        precision_errors[name] = block_error(direct32[name], direct64[name])
    check("direct_FP64_nodal_descriptor_replay", max(node_errors64.values()) <= fixed["shape_partition_and_direct_replay_relative_tolerance"], maximum_normalized_errors=node_errors64)
    check("signed_FP32_nodal_descriptor_replay", max(node_errors32.values()) <= 2e-6, maximum_normalized_errors=node_errors32,
          all_inside_FP32_vs_FP64_errors=precision_errors)
    cache_vs_physical = float(np.max(np.abs(direct64["cachedxyz"] - (points - origin) / length)))
    check("cached_coordinate_vs_direct_physical_normalization", cache_vs_physical <= fixed["canonical_xyz_vs_direct_physical_normalization_maximum"],
          maximum_dimensionless_difference=cache_vs_physical, interpretation="source float32 cache rounding; not exact off-node identity")
    signs = {int(record["original_row"]): record["orientation_sign"] for record in locator.regularity_records}
    additional_reference_J.extend((item["source_row"], item["L"]) for item in schedule if item["category"] == "interior")
    bad_J, minimum_det, minimum_sigma, maximum_condition = [], float("inf"), float("inf"), 0.
    for row, L in additional_reference_J:
        J = jacobian(physical[cells[row]], L) / length
        finite = bool(np.isfinite(J).all())
        if finite:
            try:
                singular_values = np.linalg.svd(J, compute_uv=False); determinant = float(np.linalg.det(J))
                sigma = float(singular_values[-1]); condition = float(singular_values[0] / sigma) if sigma > 0 else float("inf")
            except np.linalg.LinAlgError:
                finite = False; determinant, sigma, condition = 0., 0., float("inf")
        else: determinant, sigma, condition = 0., 0., float("inf")
        minimum_det = min(minimum_det, abs(determinant)); minimum_sigma = min(minimum_sigma, sigma); maximum_condition = max(maximum_condition, condition)
        sign = int(np.sign(determinant))
        valid = finite and abs(determinant) > 1e-18 and sigma > fixed["jacobian_sigma_min_over_L_minimum"] and condition < fixed["jacobian_condition_maximum"] and sign == signs[row]
        if not valid: bad_J.append({"original_row": row, "barycentric": L.tolist(), "det_J_over_L": determinant,
            "sigma_min_over_L": sigma, "condition": condition, "sign": sign, "center_orientation_sign": signs[row]})
    check("scheduled_interior_face_J_regular_and_orientation_consistent", not bad_J,
          additional_reference_evaluations=len(additional_reference_J), minimum_abs_det=minimum_det,
          minimum_sigma_min=minimum_sigma, maximum_condition=maximum_condition, failures=bad_J)
    inside_results = locator.locate(points)
    require(len(inside_results) == inside_count, "Locator did not preserve scheduled occurrence order/count")
    query_reports, inverse_node_errors, inverse_observed_errors, tie_failures = [], {k: 0. for k in blocks}, {k: 0. for k in blocks}, []
    max_residual = 0.
    for query_index, (scheduled, result) in enumerate(zip(schedule, inside_results)):
        accepted = result["status"] == "inside_accepted"
        fields = locator.interpolate_result(result)
        require((fields is not None) == accepted, "Failed inverse unexpectedly returned a field or accepted inverse omitted field")
        report = {"query_index": query_index, "category": scheduled["category"], "source_row": scheduled["source_row"],
            "status": result["status"], "element_row": result.get("element_row"), "field_returned": fields is not None,
            "residual_over_L": result.get("residual_over_L"), "candidate_rows": result.get("candidate_rows", [])}
        if accepted:
            max_residual = max(max_residual, float(result["residual_over_L"]))
            for name in blocks:
                error = block_error(fields[name], direct64[name][query_index])
                inverse_observed_errors[name] = max(inverse_observed_errors[name], error)
                if scheduled["category"] == "local_node": inverse_node_errors[name] = max(inverse_node_errors[name], error)
            diagnostics = result.get("diagnostics", {})
            accepted_rows = diagnostics.get("accepted_candidate_rows", [result["element_row"]])
            if accepted_rows and int(result["element_row"]) != min(int(v) for v in accepted_rows): tie_failures.append(query_index)
            trace_error = diagnostics.get("compatible_tie_trace_max", 0.)
            if trace_error is not None and float(trace_error) > fixed["inverse_tie_trace_relative_tolerance"]: tie_failures.append(query_index)
            report.update(barycentric=result.get("barycentric"), weights=result.get("weights"), diagnostics=diagnostics)
        else:
            report["diagnostics"] = result.get("diagnostics", {})
        query_reports.append(report)
    bad_inside = [v for v in query_reports if v["status"] != "inside_accepted"]
    check("all_scheduled_inside_inverse_status_and_residual", not bad_inside and max_residual <= fixed["physical_inverse_final_residual_over_L_maximum"],
          accepted_count=inside_count - len(bad_inside), scheduled_count=inside_count, maximum_physical_residual_over_L=max_residual, failures=bad_inside)
    check("inverse_nodal_descriptor_replay", not bad_inside and max(inverse_node_errors.values()) <= fixed["inverse_nodal_descriptor_replay_relative_tolerance"],
          maximum_normalized_errors=inverse_node_errors, all_inside_inverse_vs_direct_observations=inverse_observed_errors)
    check("inverse_compatible_ties_choose_smallest_original_row", not tie_failures, failed_query_indices=sorted(set(tie_failures)),
          tolerance=fixed["inverse_tie_trace_relative_tolerance"])
    midpoint = .5 * (lower + upper)
    outside_points = []
    for axis in range(3):
        for side in (0, 1):
            x = midpoint.copy(); x[axis] = lower[axis] - length if side == 0 else upper[axis] + length
            outside_points.append(x)
    outside_points.extend((lower - length, upper + length))
    outside_results = locator.locate(np.stack(outside_points))
    outside_ok = True
    for number, (point, result) in enumerate(zip(outside_points, outside_results)):
        field = locator.interpolate_result(result)
        outside_ok &= result["status"] == "outside_certified" and field is None
        query_reports.append({"query_index": inside_count + number, "category": "outside", "physical_mm": point.tolist(),
            "status": result["status"], "field_returned": field is not None, "diagnostics": result.get("diagnostics", {})})
    check("eight_whole_CAD_bound_excluded_outside_statuses", outside_ok and len(outside_results) == 8,
          statuses=[v["status"] for v in outside_results], field_returned=[locator.interpolate_result(v) is not None for v in outside_results])
    synthetic_reports, synthetic_locators, synthetic_points = [], [], []
    reference_xyz = reference_nodes[:, 1:].copy()
    for fixture in contract["synthetic_failures"]:
        name = fixture["name"]; synthetic_cells = np.arange(10, dtype=np.int64)[None]
        if name == "singular":
            synthetic_xyz = reference_xyz.copy(); synthetic_xyz[:, 2] = 0
            r = np.asarray(fixture["query_reference"], np.float64); synthetic_point = np.asarray([r[0], r[1], 0.])
        elif name == "evidenced_fold":
            synthetic_xyz = reference_xyz.copy(); synthetic_xyz[:, 0] -= 2 * synthetic_xyz[:, 0] ** 2
            r = np.asarray(fixture["query_reference"], np.float64); synthetic_point = np.asarray([r[0] - 2*r[0]**2, r[1], r[2]])
        elif name == "overlapping_interiors":
            synthetic_xyz = np.concatenate((reference_xyz, reference_xyz)); synthetic_cells = np.arange(20, dtype=np.int64).reshape(2, 10)
            synthetic_point = np.asarray([.25, .25, .25])
        elif name == "unaccepted_conservative_candidate":
            synthetic_xyz = reference_xyz.copy(); synthetic_point = np.asarray([.8, .8, .8])
        else: raise InputContractError("Unknown frozen synthetic fixture")
        lo, hi = geometry.bezier_bounds(synthetic_xyz[synthetic_cells])
        synthetic_locator = geometry.GeometryLocator(synthetic_xyz, synthetic_cells, np.arange(len(synthetic_cells)), 1.,
            config=locator_config, global_bounds=(np.asarray(lo).min(0), np.asarray(hi).max(0)),
            descriptor_blocks={"synthetic": np.arange(len(synthetic_xyz), dtype=np.float64)[:, None]})
        result = synthetic_locator.locate(synthetic_point[None])[0]
        field = synthetic_locator.interpolate_result(result)
        passed = result["status"] == fixture["expected_status"] and field is None
        synthetic_reports.append({"name": name, "expected_status": fixture["expected_status"], "actual_status": result["status"],
            "passed": passed, "cells": len(synthetic_cells), "field_returned": field is not None, "result": result,
            "regularity": synthetic_locator.regularity_records, "counters": synthetic_locator.counters})
        synthetic_locators.append(synthetic_locator); synthetic_points.append(synthetic_point)
        query_reports.append({"query_index": inside_count + 8 + len(synthetic_reports) - 1, "category": "synthetic_failure",
            "name": name, "status": result["status"], "field_returned": field is not None, "diagnostics": result.get("diagnostics", {})})
    check("four_fixed_synthetic_failure_classifications_no_field", all(v["passed"] for v in synthetic_reports),
          fixtures=synthetic_reports, synthetic_cells=sum(v["cells"] for v in synthetic_reports))
    all_points = np.concatenate((points, np.stack(outside_points), np.stack(synthetic_points)))
    counts = dict(Counter(v["category"] for v in query_reports))
    require(len(query_reports) <= 512 and len(synthetic_reports) == 4 and sum(v["cells"] for v in synthetic_reports) == 5,
            "Frozen logical-query or synthetic-cell budget exceeded")
    return _jsonable({"gate_pass": all(v["passed"] for v in checks), "checks": checks, "scientific_failures": failures,
        "conformity": scan, "selected_original_rows": selected_rows, "selected_shared_faces": faces,
        "counts": {"selected_original_cells": len(selected_rows), "selected_shared_faces": len(faces),
            "logical_query_occurrences": len(query_reports), "unique_physical_points_exact_float64": len(np.unique(all_points, axis=0)),
            "by_category": counts, "synthetic_cells": 5, "all_CAD_face_scan_cells": len(cells),
            "all_CAD_bounds_cell_evaluations": bounds_cell_evaluations, "all_CAD_bounds_control_points": 10 * bounds_cell_evaluations,
            "direct_arithmetic_reference_evaluations": arithmetic_counts,
            "paired_face_trace_map_evaluations": 14 * len(faces), "additional_known_reference_J_evaluations": len(additional_reference_J),
            "locator_counters": locator.counters, "synthetic_locator_counters": [v.counters for v in synthetic_locators]},
        "global_conservative_Bezier_bounds_mm": {"lower": lower, "upper": upper, "outside_margin_mm": length},
        "regularity_selected_cells_11_samples": locator.regularity_records, "face_trace_reports": face_reports,
        "query_results": query_reports, "synthetic_failures": synthetic_reports,
        "surrogate": {"definition": "fixed S_node32 -> nonaffine LN(eps1e-5) -> tanh per node, then signed all10 interpolation",
            "Q_node_float64_sha256": _array_sha(q_node, "<f8"), "shape": list(q_node.shape), "model_or_GT": False},
        "scope": "one TRAIN CAD; full-CAD input topology/bounds, at most32 original cells numerical locator; no global Jacobian/injectivity proof",
        "unproved_properties": contract["scientific_limits"]})


def _peak_rss():
    try:
        import resource
        return {"bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "method": "process lifetime Linux ru_maxrss"}
    except ImportError:
        return {"bytes": None, "method": "resource.ru_maxrss unavailable on this host"}


def run(args):
    if args.preflight:
        record = preflight(args.plan)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        return record
    began = time.monotonic(); accessed, contract = {}, None
    output = host_path(args.output)
    try:
        contract = request_contract(args.plan)
    except Exception as error:
        # Pure metadata identity failure is still an operationally failed run.
        # A pre-existing completed artifact is preserved for root disposition.
        if not output.exists() or not any(output.iterdir()):
            config = {"contract": None, "requested_plan": str(args.plan), "method": "CPU geometry gate metadata failure"}
            failure_class = "source_identity_mismatch" if isinstance(error, SourceIdentityError) else "corrupt_input" if isinstance(error, InputContractError) else "exception"
            failure = {"status": "failed", "gate_pass": False, "failure_class": failure_class, "error": f"{type(error).__name__}: {error}",
                "config": config, "elapsed_seconds": time.monotonic() - began,
                "receipt": {"accessed_members": {}, "labels_read": 0, "weights_read": 0, "model_calls": 0,
                            "CUDA_initialized": False, "array_members_opened": 0, "plan_sha256_expected": PLAN_SHA256, "policy_sha256_expected": POLICY_SHA256}}
            write_json(output / "config.json", config); write_json(output / "summary.json", failure)
            write_json(output / "status.json", {"phase": "failed", "failure_class": failure_class, "error": failure["error"]})
            write_json(ROOT / "notes/geobc_continuous_geometry_result.json", failure)
        raise
    summary_path = output / "summary.json"
    if summary_path.is_file():
        stored = read_json(summary_path)
        if stored["status"] == "completed":
            require(stored["config"]["contract"] == contract and read_json(output / "config.json") == stored["config"],
                    "Completed CPU gate has a different strict source/input contract", SourceIdentityError)
            require(read_json(contract["result"]) == stored, "Completed formal result differs from summary", SourceIdentityError)
            print(f"Reusing completed geometry gate {output}; gate_pass={stored['gate_pass']}", flush=True)
            return stored
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("One fixed gate output must be empty; a failed attempt requires root disposition")
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "method": "One TRAIN CAD input-only CPU continuous-query geometry gate",
              "plan": contract["plan"], "source_revision": contract["source_sha256"], "CPU_workers": 1, "BLAS_threads": 1,
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "config.json", config)
    status_record = {"phase": "loading_verified_geometry", "CAD": CAD, "started_at_utc": config["started_at_utc"], "labels_read": 0,
                     "weights_read": 0, "model_calls": 0, "CUDA_initialized": False}
    def status(phase, **details):
        status_record.update(phase=phase, elapsed_seconds=time.monotonic() - began,
                             updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
        write_json(output / "status.json", status_record)
    status("loading_verified_geometry")
    try:
        # Set before the first scientific import. No multiprocessing or GPU path.
        thread_env = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        for name in thread_env: os.environ[name] = "1"
        if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
        require("torch" not in sys.modules and "tinycudann" not in sys.modules, "CPU gate process must not import Torch/native models")
        from contextlib import nullcontext
        try:
            from threadpoolctl import threadpool_limits, threadpool_info
            thread_context = threadpool_limits(limits=1)
        except ImportError:
            thread_context, threadpool_info = nullcontext(), lambda: []
        input_began = time.monotonic()
        with thread_context:
            physical, cells, xyz, raw43, input_receipt = _load_inputs(contract, accessed)
            input_seconds = time.monotonic() - input_began
            status("bounded_topology_and_geometry_gate", input_seconds=input_seconds)
            geometry_began = time.monotonic()
            gate = _geometry_gate(contract, physical, cells, xyz, raw43)
            geometry_seconds = time.monotonic() - geometry_began
            pools = threadpool_info()
            require(all(int(v["num_threads"]) == 1 for v in pools if v.get("user_api") == "blas"), "BLAS thread count exceeded fixed one-worker contract")
        require("torch" not in sys.modules and "tinycudann" not in sys.modules, "Forbidden native/model import occurred")
        receipt = {"source_identities": contract["input_files"], "source_sha256": contract["source_sha256"],
            "metadata_sha256": contract["metadata_sha256"], "plan_sha256": PLAN_SHA256, "policy_sha256": POLICY_SHA256,
            "accessed_members": accessed, "H5_datasets_read": accessed.get("data/raw/FieldMesh/101_479.h5", []),
            "scientific_input_files_read": len(accessed), "array_identity": input_receipt,
            "labels_read": 0, "weights_read": 0, "model_calls": 0, "models_constructed": 0,
            "Torch_imported": False, "native_model_imported": False, "CUDA_initialized": False,
            "CPU_workers": 1, "BLAS_threads": 1, "thread_environment": {name: os.environ[name] for name in thread_env},
            "loaded_threadpools": pools, "input_seconds": input_seconds, "geometry_gate_seconds": geometry_seconds,
            "elapsed_seconds": time.monotonic() - began, "peak_RSS": _peak_rss(),
            "LOCATOR_status_counts": dict(Counter(v["status"] for v in gate["query_results"])),
            "scope": "101_479/ver TRAIN inputs only; at most32 real cells,8 face pairs,512 physical occurrences;4 fixtures/5 synthetic cells unless mandatory conformity stop",
            "no_tolerance_retuning_or_scope_expansion": True}
        summary = {"status": "completed", "scientific_outcome": "passed" if gate["gate_pass"] else "negative",
            "gate_pass": gate["gate_pass"], "config": config, "geometry_gate": gate, "receipt": receipt,
            "counts": gate["counts"], "LOCATOR_status_counts": receipt["LOCATOR_status_counts"],
            "scientific_failures": gate["scientific_failures"], "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_RSS": receipt["peak_RSS"], "labels_read": 0, "weights_read": 0, "model_calls": 0,
            "CUDA_initialized": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(summary_path, summary); write_json(contract["result"], summary)
        status("completed", gate_pass=summary["gate_pass"], scientific_outcome=summary["scientific_outcome"], counts=summary["counts"])
        print(f"CPU geometry gate completed; gate_pass={summary['gate_pass']} logical_queries={summary['counts']['logical_query_occurrences']} elapsed={summary['elapsed_seconds']:.3f}s", flush=True)
        return summary
    except Exception as error:
        failure_class = "source_identity_mismatch" if isinstance(error, SourceIdentityError) else "corrupt_input" if isinstance(error, InputContractError) else "exception"
        summary = {"status": "failed", "gate_pass": False, "failure_class": failure_class,
            "error": f"{type(error).__name__}: {error}", "config": config, "elapsed_seconds": time.monotonic() - began,
            "receipt": {"accessed_members": accessed, "source_identities": contract["input_files"],
                "plan_sha256": PLAN_SHA256, "policy_sha256": POLICY_SHA256, "source_sha256": contract["source_sha256"],
                "labels_read": 0, "weights_read": 0, "model_calls": 0, "CUDA_initialized": False}, "peak_RSS": _peak_rss()}
        write_json(summary_path, summary); write_json(contract["result"], summary)
        status("failed", failure_class=failure_class, error=summary["error"])
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=Path("notes/geobc_continuous_geometry_plan.json"))
    result.add_argument("--output", type=Path, default=Path("runs/geobc_exploration/continuous_query_geometry_train101_479"))
    result.add_argument("--preflight", action="store_true", help="Standard-library JSON/source/hash/stat only; no scientific arrays/imports")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
