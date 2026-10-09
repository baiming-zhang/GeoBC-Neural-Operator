"""Frozen, bounded saved-field kinematics diagnostic; no model or training imports.

request_contract is deliberately standard-library-only. Scientific imports and
array member access occur only in the explicit toy or actual execution branch.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PLAN = "notes/geobc_quadratic_kinematics_plan.json"
PLAN_SHA = "f752c5d45b7cff7b3e13d6e070f1f91980eae7a26df8fb26fc276936d7d35d2d"
MODULE_SHA = "95599c0001c2f50f1cabd5f2e74d91396f71072378a77b31763ee086e6a79c27"
SELECTED_ROWS = list(range(26)) + [29680, 34114, 58936, 86485, 92016, 94185]
ENDPOINT_SCHEMA = {
    "xyz": ("<f4", [179316, 3]),
    "u_true_mm": ("<f8", [179316, 3]),
    "u_pred_mm": ("<f8", [179316, 3]),
    "alpha": ("<f4", [3]), "sample_key": ("<U11", []),
    "step": ("<i8", []), "channels": ("<U5", [3]),
    "u_scale_mm": ("<f8", []),
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), "Path outside the frozen project")
    return path


def identity(relative, expected_sha=None, expected_bytes=None, expected_mtime=None):
    path = local_path(relative)
    before = path.stat()
    require(path.is_file(), f"Missing source file: {relative}")
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, f"Source size mismatch: {relative}")
    if expected_mtime is not None:
        require(before.st_mtime_ns == expected_mtime, f"Source mtime mismatch: {relative}")
    digest = sha_file(path) if expected_sha is not None else None
    if expected_sha is not None:
        require(digest == expected_sha, f"Source SHA mismatch: {relative}")
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Source changed during identity check: {relative}")
    result = {"path": relative, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    if digest is not None:
        result["sha256"] = digest
    return result


def npz_headers(path):
    """NPY headers and ZIP member names only; no array payload interpretation."""
    result = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate NPZ ZIP members")
        for name in names:
            require(name.endswith(".npy") and "/" not in name, "Unexpected NPZ member")
            with z.open(name) as f:
                require(f.read(6) == b"\x93NUMPY", "Invalid NPY magic")
                version = tuple(f.read(2))
                require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY version")
                nbytes = 2 if version == (1, 0) else 4
                size = struct.unpack("<H" if nbytes == 2 else "<I", f.read(nbytes))[0]
                require(size <= 65536, "Oversized NPY header")
                header = ast.literal_eval(f.read(size).decode("utf-8" if version == (3, 0) else "latin1"))
            require(set(header) == {"descr", "fortran_order", "shape"}, "Unexpected NPY schema")
            result[name[:-4]] = {"dtype": header["descr"], "shape": list(header["shape"]),
                                "fortran_order": header["fortran_order"]}
    return result


def request_contract(planPath=PLAN):
    """Pure stdlib preflight: identities, metadata, fixed scope; no scientific imports."""
    plan_path = Path(planPath)
    if not plan_path.is_absolute():
        plan_path = local_path(str(plan_path))
    require(plan_path.resolve() == local_path(PLAN), "Only the frozen formal plan is permitted")
    require(sha_file(plan_path) == PLAN_SHA, "Formal plan SHA mismatch")
    plan = read_json(plan_path)
    scope = plan["fixed_scope"]
    require(scope["selected_original_rows"] == SELECTED_ROWS and
            scope["case"] == "101_479/ver" and scope["group"] == "TRAIN" and
            scope["endpoint"] == 10000 and scope["maximum_original_cells"] == 32 and
            scope["quadrature_points_per_cell"] == 4 and
            scope["total_actual_reference_points"] == 128, "Frozen scope mismatch")
    identities = {"plan": identity(PLAN, PLAN_SHA)}
    for name, expected in plan["frozen_source"].items():
        identities[name] = identity(name, expected)
    for key in ("geometry_result", "geometry_plan", "specialist_plan"):
        identities[key] = identity(plan["parent"][key], plan["parent"][key + "_sha256"])
    for key in ("geometry", "connectivity", "endpoint", "endpoint_index",
                "endpoint_config", "endpoint_summary"):
        item = plan["inputs"][key]
        identities[key] = identity(item["path"], item.get("sha256", item.get("file_sha256")),
                                   item.get("bytes"), item.get("mtime_ns"))
    headers = npz_headers(local_path(plan["inputs"]["endpoint"]["path"]))
    require(set(headers) == set(ENDPOINT_SCHEMA), "Endpoint member allowlist mismatch")
    for key, (dtype, shape) in ENDPOINT_SCHEMA.items():
        require(headers[key] == {"dtype": dtype, "shape": shape, "fortran_order": False},
                f"Endpoint header schema mismatch: {key}")
    connectivity_names = None
    with zipfile.ZipFile(local_path(plan["inputs"]["connectivity"]["path"])) as z:
        connectivity_names = z.namelist()
    require(connectivity_names.count("cells10.npy") == 1, "Missing/duplicate cells10 member")
    index = read_json(local_path(plan["inputs"]["endpoint_index"]["path"]))
    config = read_json(local_path(plan["inputs"]["endpoint_config"]["path"]))
    summary = read_json(local_path(plan["inputs"]["endpoint_summary"]["path"]))
    case = index["cases"][scope["case"]]
    require(index["step"] == 10000 and case["file"] == "case_000.npz" and
            case["group"] == "train" and case["nodes"] == 179316, "Endpoint index scope mismatch")
    train = [c for c in config["cases"]["train"] if c["sample_key"] == scope["case"]]
    require(config["groups"] == {"train": ["101_479"], "validation": []} and
            len(train) == 1 and train[0]["alpha"] == [1.0, 0.0, 0.0] and
            train[0]["nodes"] == 179316 and not config["cases"]["validation"],
            "TRAIN membership mismatch")
    require(config["scaler"]["center_mm"] == 0.0 and
            config["scaler"]["transform"] == "linear_zero_center_common_scalar" and
            config["scaler"]["channels"] == plan["inputs"]["endpoint"]["channels"],
            "Signed physical displacement policy mismatch")
    require(summary["status"] == "completed" and summary["step"] == 10000 and
            summary["checkpoint_selection"] == "fixed10000_endpoint_no_selection",
            "Saved specialist endpoint is not completed fixed10000")
    prior = read_json(local_path(plan["parent"]["geometry_result"]))
    require(prior["status"] == "completed" and prior["gate_pass"] is True and
            prior["geometry_gate"]["selected_original_rows"] == SELECTED_ROWS,
            "Prior geometry gate/selection mismatch")
    records = prior["geometry_gate"]["regularity_selected_cells_11_samples"]
    require([r["original_row"] for r in records] == SELECTED_ROWS and
            all(r["valid"] and r["status"] == "passed" and r["orientation_sign"] in (-1, 1)
                for r in records), "Prior orientation records invalid")
    tool_sha = sha_file(Path(__file__))
    ready = read_json(local_path(plan["execution"]["ready"]))
    require(ready["status"] == "ready_for_source_review" and
            ready["plan_sha256"] == PLAN_SHA and ready["tool_sha256"] == tool_sha and
            ready["geometry_module_sha256"] == MODULE_SHA and
            ready["toy"]["passed"] is True and ready["toy"]["fixture_runs"] == 1,
            "Author toy/source readiness mismatch")
    identities["ready"] = identity(plan["execution"]["ready"], sha_file(local_path(plan["execution"]["ready"])))
    return {"schema_version": 1, "stage": plan["stage"], "plan_sha256": PLAN_SHA,
            "tool_sha256": tool_sha, "identities": identities,
            "fixed_scope": scope, "endpoint_schema": headers,
            "connectivity_member_names": connectivity_names,
            "allowed_array_reads": {"geometry": ["vertices"], "connectivity": ["cells10"],
                                    "endpoint": list(ENDPOINT_SCHEMA)},
            "prior_orientation": {str(r["original_row"]): r["orientation_sign"] for r in records},
            "u_scale_mm": config["scaler"]["scale_mm"],
            "quadrature": {"barycentric_a": (5 + 3 * math.sqrt(5)) / 20,
                           "barycentric_b": (5 - math.sqrt(5)) / 20,
                           "reference_weight": 1 / 24, "points_per_cell": 4},
            "geometry_thresholds": {"abs_det_J_over_L3_strict_gt": 1e-18,
                                    "sigma_min_J_over_L_strict_gt": 1e-12,
                                    "condition_strict_lt": 1e12},
            "direct_math_relative_tolerance": 1e-10,
            "signed_energy_identity_relative_tolerance": 1e-10,
            "scientific_imports_in_request_contract": 0,
            "model_calls": 0, "CUDA_initialized": False, "optimizer_updates": 0}


def science():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from geobc.continuous_query_geometry import shape_gradient, REFERENCE_POINTS
    return np, shape_gradient, REFERENCE_POINTS


def quadrature(np):
    a, b = (5 + 3 * math.sqrt(5)) / 20, (5 - math.sqrt(5)) / 20
    bary = np.full((4, 4), b, dtype=np.float64)
    np.fill_diagonal(bary, a)
    return bary


def relative_block(np, actual, expected):
    return float(np.max(np.abs(actual - expected)) / max(1.0, float(np.max(np.abs(expected)))))


def symmetric(matrix):
    return (matrix + matrix.swapaxes(-1, -2)) * 0.5


def kinematics(np, D, X, u):
    """All arrays FP64. J has physical rows/reference columns; G has u rows/X columns."""
    A = (X[1:4] - X[0]).T
    U = (u[1:4] - u[0]).T
    G1 = np.linalg.solve(A.T, U.T).T
    J = np.einsum("ic,qir->qcr", X, D)
    U2 = np.einsum("ic,qir->qcr", u, D)
    G2 = np.linalg.solve(J.swapaxes(-1, -2), U2.swapaxes(-1, -2)).swapaxes(-1, -2)
    affine = u[0] + (X - X[0]) @ G1.T
    delta = u - affine
    Umid = np.einsum("ic,qir->qcr", delta, D)
    Gmid = np.linalg.solve(J.swapaxes(-1, -2), Umid.swapaxes(-1, -2)).swapaxes(-1, -2)
    return {"G1": G1, "G2": G2, "eps1": symmetric(G1),
            "eps2": symmetric(G2), "epsmid": symmetric(Gmid), "delta": delta}


def energy(np, w, tensor):
    # Full symmetric 3x3 inner product counts each off-diagonal shear twice.
    return float(np.sum(w * np.sum(tensor * tensor, axis=(-2, -1))))


def energy_parts(np, w, e1, emid, e2):
    E1, Emid, E2 = energy(np, w, e1), energy(np, w, emid), energy(np, w, e2)
    cross = float(2 * np.sum(w * np.sum(e1 * emid, axis=(-2, -1))))
    denominator = abs(E2) + abs(E1) + abs(Emid) + abs(cross)
    residual = 0.0 if denominator == 0.0 else abs(E2 - E1 - Emid - cross) / denominator
    return {"E1": E1, "Emid": Emid, "signed_cross": cross, "E2": E2,
            "identity_relative_residual": residual}


def ratio(numerator, denominator):
    if denominator == 0.0:
        return {"value": None, "reason": "exact_zero_reference_energy"}
    return {"value": math.sqrt(numerator / denominator), "reason": None}


def metrics(np, w, truth, pred):
    volume = float(np.sum(w))
    e1 = pred["eps1"] - truth["eps1"]
    emid = pred["epsmid"] - truth["epsmid"]
    e2 = pred["eps2"] - truth["eps2"]
    result = {"error_decomposition": energy_parts(np, w, e1, emid, e2)}
    for name in ("P1", "P2"):
        key = "eps1" if name == "P1" else "eps2"
        reference = energy(np, w, truth[key])
        prediction = energy(np, w, pred[key])
        error = energy(np, w, pred[key] - truth[key])
        result[name] = {"truth_reference_energy_mm3": reference,
                        "prediction_reference_energy_mm3": prediction,
                        "error_energy_mm3": error,
                        "truth_reference_Frobenius_RMS": math.sqrt(reference / volume),
                        "prediction_reference_Frobenius_RMS": math.sqrt(prediction / volume),
                        "error_Frobenius_RMS": math.sqrt(error / volume),
                        "relative_L2_to_truth": ratio(error, reference)}
    for label, value in (("truth", truth), ("prediction", pred)):
        mismatch = energy(np, w, value["eps2"] - value["eps1"])
        result[label + "_P2_minus_P1"] = {
            "energy_mm3": mismatch, "Frobenius_RMS": math.sqrt(mismatch / volume),
            "relative_L2_to_same_P2": ratio(mismatch, result["P2"][label + "_reference_energy_mm3"])}
        delta = value["delta"][..., 4:10, :]
        result[label + "_midside_residual_displacement_vector_RMS_mm"] = float(
            np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))
    return result


def toy():
    """One deterministic suite; reads no scientific source files or saved fields."""
    started = time.perf_counter()
    require(sha_file(local_path(PLAN)) == PLAN_SHA, "Toy plan SHA mismatch")
    require(sha_file(local_path("geobc/continuous_query_geometry.py")) == MODULE_SHA,
            "Toy geometry module SHA mismatch")
    ready_path = local_path(read_json(local_path(PLAN))["execution"]["ready"])
    require(not ready_path.exists(), "Unique author toy receipt already exists; do not repeat")
    np, shape_gradient, reference = science()
    bary = quadrature(np)
    D = shape_gradient(bary)
    straight = np.asarray(reference, dtype=np.float64).copy()
    curved = straight.copy()
    curved[4:] += np.array([[.015, .010, -.005], [-.010, .012, .005],
                           [.005, -.008, .006], [.007, .003, .012],
                           [-.006, .004, -.009], [.004, -.005, .007]])
    B = np.array([[.3, -.4, .2], [.7, -.2, -.1], [-.5, .6, .1]])
    rigid_B = (B - B.T) * .5
    translation = np.array([.2, -.3, .4])
    errors = {}
    for name, X in (("straight", straight), ("curved", curved)):
        for mode, matrix in (("affine", B), ("rigid", rigid_B)):
            value = kinematics(np, D, X, translation + X @ matrix.T)
            errors[name + "_" + mode] = {
                "gradient": relative_block(np, value["G2"], matrix),
                "strain": relative_block(np, value["eps2"], symmetric(matrix)),
                "mid_strain": float(np.max(np.abs(value["epsmid"]))),
                "decomposition": relative_block(np, value["eps2"], value["eps1"] + value["epsmid"])}
    spike = np.zeros((10, 3), dtype=np.float64)
    spike[4, 0] = 1.0
    mid = kinematics(np, D, straight, spike)
    expected = np.zeros((4, 3, 3))
    expected[:, 0, 0] = 4 * (bary[:, 0] - bary[:, 1])
    expected[:, 0, 1] = expected[:, 1, 0] = -2 * bary[:, 1]
    expected[:, 0, 2] = expected[:, 2, 0] = -2 * bary[:, 1]
    w = np.full(4, 1 / 24, dtype=np.float64)
    spike_energy = energy(np, w, mid["eps2"])
    affine = np.zeros_like(spike)
    affine[:, 0] = straight[:, 1]
    mixed = kinematics(np, D, straight, affine + spike)
    parts = energy_parts(np, w, mixed["eps1"], mixed["epsmid"], mixed["eps2"])
    errors["spike"] = {"strain": relative_block(np, mid["eps2"], expected),
                       "P1": float(np.max(np.abs(mid["eps1"]))),
                       "energy": abs(spike_energy - 8 / 15),
                       "volume_average_RMS": abs(math.sqrt(spike_energy / (1 / 6)) - 4 / math.sqrt(5))}
    errors["signed_mixed_energy"] = {"E1": abs(parts["E1"] - 1 / 12),
                                     "Emid": abs(parts["Emid"] - 8 / 15),
                                     "signed_cross": abs(parts["signed_cross"] + 1 / 6),
                                     "E2": abs(parts["E2"] - 9 / 20),
                                     "identity": parts["identity_relative_residual"]}
    maximum = max(v for record in errors.values() for v in record.values())
    receipt = {"passed": maximum <= 1e-12, "fixture_runs": 1, "FP64_tolerance": 1e-12,
               "maximum_absolute_or_block_error": maximum, "errors": errors,
               "observed_spike_energy": spike_energy, "observed_mixed_energy": parts,
               "elapsed_seconds": time.perf_counter() - started,
               "real_array_reads": 0, "model_calls": 0, "CUDA_initialized": False,
               "optimizer_updates": 0, "fixture_kind": "fixed_analytic_FP64_only"}
    ready = {"schema_version": 1, "status": "ready_for_source_review" if receipt["passed"] else "toy_failed",
             "plan": PLAN, "plan_sha256": PLAN_SHA,
             "tool": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
             "tool_sha256": sha_file(Path(__file__)), "geometry_module_sha256": MODULE_SHA,
             "API": {"request_contract(planPath)": "purestdlib identity/stat/hash/JSON/NPY-header only",
                     "CLI_toy": "--toy (unique author fixture, no actual arrays)",
                     "CLI_actual": "--plan <frozen plan> --output <frozen output>; root queue release only"},
             "toy": receipt, "actual_scientific_arrays_read_by_author": 0,
             "actual_stage_executed_by_author": False,
             "endpoint_schema_source": "ZIP NPY headers only; payload not read",
             "midside_RMS_definition": "uniform vector RMS over192 local midside incidences; repeats retained",
             "awaiting": "Independent focused source review and root unique CPU actual release",
             "created_at_utc": utc()}
    write_json(ready_path, ready)
    print(json.dumps(ready, allow_nan=False))
    return 0 if receipt["passed"] else 1


def array_sha(np, array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def gauss_geometry(np, D, X, rows, scale, orientation):
    J = np.einsum("nic,qir->nqcr", X, D)
    scaled = J / scale
    records, failures = [], []
    for ci, row in enumerate(rows):
        points = []
        for qi in range(4):
            M = scaled[ci, qi]
            finite = bool(np.all(np.isfinite(M)))
            det = sigma = condition = None
            sign = 0
            if finite:
                det = float(np.linalg.det(M))
                singular = np.linalg.svd(M, compute_uv=False)
                sigma = float(singular[-1])
                condition = None if sigma == 0.0 else float(singular[0] / sigma)
                sign = int(np.sign(det)) if math.isfinite(det) else 0
                det = det if math.isfinite(det) else None
                sigma = sigma if math.isfinite(sigma) else None
                condition = condition if condition is not None and math.isfinite(condition) else None
            valid = (finite and det is not None and math.isfinite(det) and abs(det) > 1e-18 and
                     sigma is not None and math.isfinite(sigma) and sigma > 1e-12 and
                     condition is not None and math.isfinite(condition) and condition < 1e12 and
                     sign == orientation[str(row)])
            point = {"quadrature_index": qi, "valid": valid, "det_J_over_L3": det,
                     "sigma_min_J_over_L": sigma, "condition": condition,
                     "orientation_sign": sign, "prior_orientation_sign": orientation[str(row)]}
            points.append(point)
            if not valid:
                failures.append({"original_row": row, **point})
        records.append({"original_row": row, "passed": all(p["valid"] for p in points), "points": points})
    # No target processing may proceed unless ALL128 points passed.
    return J, records, failures


def actual(contract):
    plan = read_json(local_path(PLAN))
    np, shape_gradient, _ = science()
    import h5py
    accesses = {"H5": [], "connectivity_NPZ": [], "endpoint_NPZ": []}
    geometry = plan["inputs"]["geometry"]
    with h5py.File(local_path(geometry["path"]), "r") as h:
        require(h["vertices"].shape == (179316, 3) and h["vertices"].dtype == np.dtype("float32"),
                "Physical vertices schema mismatch")
        vertices = h["vertices"][...]
        accesses["H5"].append("vertices")
    require(array_sha(np, vertices) == geometry["array_float32_sha256"], "Physical vertices array SHA mismatch")
    require(bool(np.all(np.isfinite(vertices))), "Nonfinite source coordinates")
    connectivity = plan["inputs"]["connectivity"]
    with np.load(local_path(connectivity["path"]), allow_pickle=False) as z:
        cells = z["cells10"]
        accesses["connectivity_NPZ"].append("cells10")
    require(cells.shape == (111754, 10) and cells.dtype == np.dtype("int64"), "cells10 schema mismatch")
    require(array_sha(np, cells) == connectivity["array_int64_sha256"], "cells10 array SHA mismatch")
    require(int(cells.min()) >= 0 and int(cells.max()) < 179316 and
            bool(np.all(np.diff(np.sort(cells, axis=1), axis=1) != 0)), "Invalid cells10 indices")
    endpoint = plan["inputs"]["endpoint"]
    norm = plan["inputs"]["normalization"]
    with np.load(local_path(endpoint["path"]), allow_pickle=False) as z:
        xyz = z["xyz"]
        accesses["endpoint_NPZ"].append("xyz")
        require(array_sha(np, xyz) == endpoint["xyz_float32_sha256"], "Endpoint xyz SHA mismatch")
        canonical = (vertices - np.asarray(norm["bbox_min_mm"], dtype=np.float32)) / np.float32(norm["characteristic_length_mm"])
        require(canonical.dtype == np.dtype("float32") and np.array_equal(canonical, xyz),
                "Original physical/canonical coordinate identity mismatch")
        scalar = {}
        for name in ("alpha", "sample_key", "step", "channels", "u_scale_mm"):
            scalar[name] = z[name]
            accesses["endpoint_NPZ"].append(name)
        require(scalar["alpha"].tolist() == endpoint["alpha"] and
                scalar["sample_key"].item() == endpoint["sample_key"] and
                scalar["step"].item() == endpoint["step"] and
                scalar["channels"].tolist() == endpoint["channels"] and
                scalar["u_scale_mm"].item() == contract["u_scale_mm"], "Endpoint scalar metadata mismatch")
        rows = SELECTED_ROWS
        selected_ids = cells[rows]
        X = vertices[selected_ids].astype(np.float64)
        bary = quadrature(np)
        D = shape_gradient(bary)
        J, regularity, failures = gauss_geometry(np, D, X, rows, norm["characteristic_length_mm"],
                                                 contract["prior_orientation"])
        base = {"checks": {"source_scope_schema_coordinate_identity": True,
                            "all128_Gauss_J_regular_and_prior_orientation": not failures},
                "Gauss_geometry": regularity, "Gauss_points_checked": 128,
                "selected_original_rows": rows, "quadrature_barycentric": bary.tolist(),
                "reference_weight": 1 / 24, "accessed_members": accesses,
                "target_numerical_points_processed": 0, "per_cell": [], "metrics": None}
        if failures:
            return {**base, "gate_pass": False, "scientific_failures": failures,
                    "unexecuted_reason": "At least one Gauss J failed; no target processing or partial target report"}
        # Geometry passes first. Only these two immutable signed physical TRAIN arrays are labels/predictions.
        fields = {}
        for name in ("u_true_mm", "u_pred_mm"):
            value = z[name]
            accesses["endpoint_NPZ"].append(name)
            require(value.shape == (179316, 3) and value.dtype == np.dtype("float64") and
                    bool(np.all(np.isfinite(value))), f"Invalid saved physical field: {name}")
            fields[name] = value[selected_ids].astype(np.float64, copy=True)
    weights = np.abs(np.linalg.det(J)) / 24
    all_values = {"truth": [], "prediction": []}
    per_cell = []
    direct_errors = []
    for ci, row in enumerate(rows):
        pair = {}
        try:
            for label, name in (("truth", "u_true_mm"), ("prediction", "u_pred_mm")):
                value = kinematics(np, D, X[ci], fields[name][ci])
                require(all(bool(np.all(np.isfinite(v))) for v in value.values()), "Nonfinite kinematics")
                pair[label] = value
                all_values[label].append(value)
                direct_errors.append(relative_block(np, value["eps2"], value["eps1"] + value["epsmid"]))
                direct_errors.append(relative_block(np, value["delta"][:4], np.zeros((4, 3))))
        except np.linalg.LinAlgError:
            return {**base, "gate_pass": False,
                    "scientific_failures": [{"original_row": row, "reason": "Singular physical corner-P1 reference"}],
                    "unexecuted_reason": "Full corner-P1 decomposition unavailable; no partial target report"}
        report = metrics(np, weights[ci], pair["truth"], pair["prediction"])
        report.update({"original_row": row, "quadrature_volume_mm3": float(weights[ci].sum()),
                       "corner_volume_mm3": float(abs(np.linalg.det((X[ci, 1:4] - X[ci, 0]).T)) / 6),
                       "direct_truth_decomposition_block_error": relative_block(np, pair["truth"]["eps2"], pair["truth"]["eps1"] + pair["truth"]["epsmid"]),
                       "direct_prediction_decomposition_block_error": relative_block(np, pair["prediction"]["eps2"], pair["prediction"]["eps1"] + pair["prediction"]["epsmid"])})
        per_cell.append(report)
    stacked = {}
    for label in ("truth", "prediction"):
        stacked[label] = {key: np.stack([value[key] for value in all_values[label]])
                          for key in ("eps1", "eps2", "epsmid", "delta")}
        stacked[label]["eps1"] = stacked[label]["eps1"][:, None, :, :]
    combined = metrics(np, weights, stacked["truth"], stacked["prediction"])
    energy_errors = [r["error_decomposition"]["identity_relative_residual"] for r in per_cell]
    energy_errors.append(combined["error_decomposition"]["identity_relative_residual"])
    direct_pass = max(direct_errors) <= 1e-10
    energy_pass = max(energy_errors) <= 1e-10
    base["checks"].update({"physical_affine_decomposition": direct_pass, "signed_energy_identity": energy_pass})
    scientific_failures = []
    if not direct_pass:
        scientific_failures.append({"reason": "physical_affine_decomposition", "max_block_error": max(direct_errors)})
    if not energy_pass:
        scientific_failures.append({"reason": "signed_energy_identity", "max_relative_residual": max(energy_errors)})
    return {**base, "gate_pass": all(base["checks"].values()), "scientific_failures": scientific_failures,
            "target_numerical_points_processed": 128, "per_cell": per_cell, "metrics": combined,
            "max_direct_block_error": max(direct_errors), "max_energy_identity_relative_residual": max(energy_errors),
            "quadrature_volume_mm3": float(weights.sum()),
            "selected_corner_volume_mm3": sum(r["corner_volume_mm3"] for r in per_cell),
            "midside_residual_RMS_scope": {"local_incidences": 192, "unique_node_count": int(np.unique(selected_ids[:, 4:]).size),
                                           "definition": "sqrt(mean over192 local midside incidences of vector norm squared); uniform, repeats retained"},
            "strain_RMS_definition": "sqrt(quadrature weighted full symmetric3x3 Frobenius energy / quadrature volume)",
            "energy_units": "mm^3 (dimensionless strain squared times physical volume)",
            "quad_exactness": "Affine geometry P2 polynomial energy exact; curved geometry fixed4point proxy only",
            "historical_metric_replaced": False, "accuracy_gate_applied": False}


def peak_rss():
    try:
        import resource
        return {"value": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "unit": "KiB_on_Linux"}
    except ImportError:
        return None


def execute(planPath, output):
    started = time.perf_counter()
    contract = request_contract(planPath)
    plan = read_json(local_path(PLAN))
    destination = Path(output).resolve() if Path(output).is_absolute() else local_path(output)
    require(destination == local_path(plan["execution"]["output"]), "Only the frozen output directory is permitted")
    result_path = local_path(plan["execution"]["result"])
    summary_path, config_path, status_path = destination / "summary.json", destination / "config.json", destination / "status.json"
    if destination.exists() and any(destination.iterdir()):
        require(summary_path.is_file() and config_path.is_file() and status_path.is_file(), "Incomplete output; automatic rerun forbidden")
        old = read_json(summary_path)
        require(old["status"] == "completed" and read_json(status_path)["status"] == "completed" and
                read_json(config_path)["contract"] == contract and old["config"]["contract"] == contract and
                result_path.is_file() and read_json(result_path) == old, "Completed reuse contract mismatch")
        print(json.dumps({"status": "completed", "reused": True, "gate_pass": old["gate_pass"],
                          "result": str(result_path)}, allow_nan=False))
        return 0
    require(not result_path.exists(), "Formal result already exists; automatic replacement forbidden")
    config = {"contract": contract, "method": plan["mathematics"], "limits": plan["limits"],
              "started_at_utc": utc(), "CPU_workers": 1, "BLAS_threads": 1,
              "no_array_writes": True, "scope": "Fixed saved TRAIN101_479/ver endpoint10000; selected32 cells only"}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    write_json(status_path, {"status": "running", "started_at_utc": config["started_at_utc"]})
    try:
        gate = actual(contract)
        # Recheck every source identity after the read; an input change is corrupt, never a scientific negative.
        require(request_contract(planPath) == contract, "Source identity changed during execution")
        result = {"schema_version": 1, "status": "completed", "scientific_outcome": "passed" if gate["gate_pass"] else "negative",
                  "gate_pass": gate["gate_pass"], "config": config, "kinematics": gate,
                  "elapsed_seconds": time.perf_counter() - started, "peak_RSS": peak_rss(),
                  "receipt": {"source": "Existing signed physical saved TRAIN truth/pred endpoint, no inference",
                              "labels_read": gate["accessed_members"]["endpoint_NPZ"].count("u_true_mm"),
                              "prediction_arrays_read": gate["accessed_members"]["endpoint_NPZ"].count("u_pred_mm"),
                              "other_labels_read": 0, "weights_read": 0, "model_calls": 0,
                              "CUDA_initialized": False, "optimizer_updates": 0, "scientific_array_writes": 0},
                  "completed_at_utc": utc()}
        write_json(summary_path, result)
        write_json(result_path, result)
        write_json(status_path, {"status": "completed", "gate_pass": result["gate_pass"],
                                 "scientific_outcome": result["scientific_outcome"], "completed_at_utc": result["completed_at_utc"]})
        print(json.dumps({"status": "completed", "gate_pass": result["gate_pass"],
                          "scientific_outcome": result["scientific_outcome"],
                          "elapsed_seconds": result["elapsed_seconds"], "result": str(result_path)}, allow_nan=False))
        return 0
    except Exception as exc:
        failure = {"schema_version": 1, "status": "failed", "gate_pass": False,
                   "config": config, "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_seconds": time.perf_counter() - started, "failed_at_utc": utc()}
        write_json(summary_path, failure)
        write_json(result_path, failure)
        write_json(status_path, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps(failure, allow_nan=False))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--output", default="runs/geobc_exploration/quadratic_kinematics_specialist101_479_ver")
    args = parser.parse_args()
    try:
        if args.toy:
            return toy()
        return execute(args.plan, args.output)
    except Exception as exc:
        # Preflight/source corruption is an execution failure, never a scientific negative.
        print(json.dumps({"status": "failed", "gate_pass": False,
                          "error_type": type(exc).__name__, "error": str(exc),
                          "failed_at_utc": utc()}, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
