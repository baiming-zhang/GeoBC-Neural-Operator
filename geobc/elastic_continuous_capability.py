"""Frozen original physical full-TET10 continuous U/gradient capability screen.

Metadata entry points never import science or read array payloads. The actual
released run persists all five candidate fields before any shared source GT U
read. Derived reference gradients/engineering strains are nodal P2 proxies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "geobc/elastic_continuous_capability.py"
PLAN = "notes/geobc_physical_continuous_plan.json"
DESIGN = "notes/geobc_physical_continuous_candidate_design.json"
KIND = "physical_TET10_continuous_U_gradient_capability"
OUTPUT = "runs/geobc_exploration/elastic_continuous5_ver_capability"
CASES = ("101_428/ver", "101_56/ver", "101_78/ver", "102_249/ver", "102_38/ver")
SCIENCE_DIGEST = "81e86d9e61aadc793134fd95e4047798003a3bfaccd8f47400d767892fd265f2"
ZERO_KEYS = ("stress_source_label_reads", "strain_source_label_reads", "learned_weight_reads", "training_updates",
    "scaler_fits", "new_physical_solves", "new_tensor_recoveries", "K_applications", "old_K_guard_repeats",
    "old_final_reads", "sealed_new_test_reads", "DEV_dia_reads", "GT_reads_before_all_five_prediction_barrier")
ACTUAL_KEYS = ("physical_U_prediction_reads", "physical_U_byte_hash_reads", "GT_U_reads", "input_cache_reads",
    "input_cache_byte_hash_reads", "input_member_reads", "operator_array_reads", "own_prediction_array_reads",
    "own_reference_array_reads", "source_GT_U_read_attempts", "source_GT_U_read_completed")
INPUT_MEMBERS = ("vertices_mm", "cells10", "grid_node_ids", "ctetra_element_ids")
FIELDS = ("U_mm", "gradient_dimensionless", "engineering_strain6")


def path(value):
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/"):
        value = value[5].upper() + ":/" + value[7:]
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def read(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def write(value, data):
    target = path(value); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def sha(value):
    result = hashlib.sha256()
    with path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stat_matches(record):
    info = path(record["path"]).stat()
    return info.st_size == record["bytes"] and info.st_mtime_ns == record["mtime_ns"]


def science_design(design):
    excluded = {"status", "actual_execution_released", "source_preparation_authorized", "created_UTC", "scientific_digest"}
    return {key: value for key, value in design.items() if key not in excluded}


def request_contract(plan_path=PLAN):
    """Source, small JSON and stat only; no array/target/scientific imports."""
    plan, design = read(plan_path), read(DESIGN)
    require(plan["kind"] == KIND and tuple(plan["case_keys"]) == CASES and plan["output"] == OUTPUT,
        "Frozen five-CAD continuous scope differs")
    science = science_design(design)
    require(digest(science) == SCIENCE_DIGEST == design["scientific_digest"] == plan["scientific_digest"]
        and plan["science_recipe"] == science and plan["actual_execution_released"] is False,
        "Frozen continuous scientific recipe changed")
    sources = {name: sha(name) for name in plan["source_sha256"]}
    require(sources == plan["source_sha256"] and SOURCE in sources, "Frozen continuous source identity differs")
    inherited = 0.0
    for key in CASES:
        item = science["records"][key]
        for name in ("input_manifest", "operator_summary", "original_zero_summary"):
            require(sha(item[name]["path"]) == item[name]["sha256"], "Accepted parent metadata differs: " + key)
        manifest, operator, old = (read(item[name]["path"]) for name in ("input_manifest", "operator_summary", "original_zero_summary"))
        require(manifest["status"] == operator["status"] == old["status"] == "completed"
            and manifest["feasibility_pass"] is True and operator["feasibility_pass"] is True
            and old["numerically_converged"] is True and old["true_relative_residual"] <= 1e-8,
            "Accepted original numerical/cache endpoint incomplete")
        require(operator["input_manifest_sha256"] == item["input_manifest"]["sha256"]
            and item["input_members"]["grid_node_ids"]["sha256"] == old["config"]["contract"]["expected_geometry"]["grid_node_ids_int64_sha256"]
            and item["input_members"]["cells10"]["shape"] == [item["cells"], 10], "Full geometry/GRID identity differs")
        require(item["operator_arrays"]["gradients"]["shape"] == [item["cells"], 4, 10, 3]
            and item["operator_arrays"]["weights"]["shape"] == [item["cells"], 4], "Accepted full4Q cache shape differs")
        for record in (item["input_cache"], item["physical_U"], item["deferred_GT_U"], *item["operator_arrays"].values()):
            require(stat_matches(record), "Accepted input/field/target stat differs: " + key)
        inherited += item["inherited_original_GT_free_physical_seconds"]
    require(abs(inherited - 452.954357734) < 1e-8
        and abs(inherited - science["budget"]["inherited_original_zero_GT_free_seconds_once"]) < 1e-8,
        "Inherited original physical accounting differs")
    return {"schema": "geobc_physical_TET10_continuous_v1", "kind": KIND, "case_keys": list(CASES),
        "records": science["records"], "science_recipe": science, "scientific_digest": SCIENCE_DIGEST,
        "science_recipe_sha256": SCIENCE_DIGEST, "source_sha256": sources, "inherited_seconds": inherited,
        "output": str(path(OUTPUT)), "array_or_label_reads_in_request_contract": 0}, plan


class ScientificNegative(RuntimeError):
    pass


class Clock:
    def __init__(self, output, began):
        self.file = output / "clock.json"; previous = read(self.file) if self.file.exists() else {}
        self.prior = float(previous.get("new_actual_wall_seconds", 0)); self.gap = float(previous.get("unobserved_exit_gap_seconds", 0))
        if previous.get("active"):
            extra = max(0.0, time.time() - previous["observed_unix_seconds"])
            self.prior += extra; self.gap += extra
        self.began = began; self.inherited = 452.954357734; self.torch = None
        self.peak_allocated = int(previous.get("peak_GPU_allocated_bytes", 0)); self.peak_reserved = int(previous.get("peak_GPU_reserved_bytes", 0))
        self.attempts = int(previous.get("attempts", 0)) + 1; self.last_persist = began
        self.persist()

    def actual(self):
        return self.prior + time.perf_counter() - self.began

    def persist(self, active=True, phase=None):
        if self.torch is not None:
            self.peak_allocated = max(self.peak_allocated, self.torch.cuda.max_memory_allocated())
            self.peak_reserved = max(self.peak_reserved, self.torch.cuda.max_memory_reserved())
        write(self.file, {"new_actual_wall_seconds": self.actual(), "inherited_original_zero_GT_free_seconds_once": self.inherited,
            "campaign_accounted_wall_seconds": self.inherited + self.actual(), "unobserved_exit_gap_seconds": self.gap,
            "peak_GPU_allocated_bytes": self.peak_allocated, "peak_GPU_reserved_bytes": self.peak_reserved,
            "attempts": self.attempts, "active": active, "observed_unix_seconds": time.time(), "phase": phase})
        self.last_persist = time.perf_counter()

    def guard(self):
        if self.actual() >= 300:
            raise ScientificNegative("Fixed new-actual300s cumulative budget exhausted")
        if self.torch is not None:
            self.peak_allocated = max(self.peak_allocated, self.torch.cuda.max_memory_allocated())
            self.peak_reserved = max(self.peak_reserved, self.torch.cuda.max_memory_reserved())
        if max(self.peak_allocated, self.peak_reserved) > 24 * 1024 ** 3:
            raise ScientificNegative("Fixed cumulative24GiB GPU peak exceeded")
        if time.perf_counter() - self.last_persist >= 60:
            self.persist()


class CaseClock:
    def __init__(self, folder, clock, phase):
        self.file = folder / "costs.json"; self.clock = clock; self.phase = phase
        self.data = read(self.file) if self.file.exists() else {"cumulative_actual_case_seconds": 0.0, "phases": {}, "attempts": 0}
        if self.data.get("active"):
            gap = max(0.0, time.time() - self.data["observed_unix_seconds"])
            self.data["cumulative_actual_case_seconds"] += gap
            self.data["unobserved_exit_gap_seconds"] = self.data.get("unobserved_exit_gap_seconds", 0) + gap
        self.prior = self.data["cumulative_actual_case_seconds"]; self.start = time.perf_counter()
        self.data.update(active=True, phase=phase, observed_unix_seconds=time.time(), attempts=self.data["attempts"] + 1)
        write(self.file, self.data); clock.persist(phase=phase)

    def guard(self):
        self.clock.guard()
        if self.prior + time.perf_counter() - self.start >= 60:
            raise ScientificNegative("Fixed combined candidate/reference60s case budget exhausted")

    def finish(self):
        elapsed = time.perf_counter() - self.start
        self.data["cumulative_actual_case_seconds"] = self.prior + elapsed
        self.data["phases"][self.phase] = self.data["phases"].get(self.phase, 0.0) + elapsed
        self.data.update(active=False, observed_unix_seconds=time.time()); write(self.file, self.data)
        self.clock.persist(phase=self.phase + "_finished")
        if self.data["cumulative_actual_case_seconds"] >= 60:
            raise ScientificNegative("Fixed combined60s case budget exhausted including persistence")
        self.clock.guard()


class Ledger:
    def __init__(self, output):
        self.file = output / "read_receipt.json"
        self.data = read(self.file) if self.file.exists() else {key: 0 for key in ZERO_KEYS + ACTUAL_KEYS}
        self.data.setdefault("attempted_members", {}); self.data.setdefault("completed_members", {})
        require(all(type(self.data.get(key)) is int and self.data[key] == 0 for key in ZERO_KEYS), "Forbidden historical reads")
        self.persist()

    def persist(self):
        write(self.file, self.data)

    def bump(self, key, token=None):
        self.data[key] += 1
        if token:
            self.data["attempted_members"][token] = self.data["attempted_members"].get(token, 0) + 1
        self.persist()

    def complete(self, token):
        self.data["completed_members"][token] = self.data["completed_members"].get(token, 0) + 1; self.persist()


def array_identity(np, value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save_array(np, target, value):
    target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_name(target.name + ".tmp")
    array = np.ascontiguousarray(value)
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, target); info = target.stat()
    return {"path": str(target), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "shape": list(array.shape),
        "dtype": str(array.dtype), "values_sha256": array_identity(np, array)}


def own_array(np, record, ledger, reference=False):
    token = "own:" + record["path"]; ledger.bump("own_reference_array_reads" if reference else "own_prediction_array_reads", token)
    require(stat_matches(record), "Owned array stat changed")
    array = np.load(path(record["path"]), allow_pickle=False, mmap_mode="r")
    require(list(array.shape) == record["shape"] and str(array.dtype) == record["dtype"]
        and array_identity(np, array) == record["values_sha256"], "Owned array schema/value identity changed")
    ledger.complete(token); return array


def save_small(np, target, arrays):
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, target); info = target.stat()
    return {"path": str(target), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "file_sha256": sha(target),
        "members": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "values_sha256": array_identity(np, value)} for name, value in arrays.items()}}


def own_small(np, record, ledger, reference=False):
    token = "own:" + record["path"]; ledger.bump("own_reference_array_reads" if reference else "own_prediction_array_reads", token)
    require(stat_matches(record) and sha(record["path"]) == record["file_sha256"], "Owned small query identity changed")
    with np.load(path(record["path"]), allow_pickle=False) as stored:
        require(set(stored.files) == set(record["members"]), "Small query member set differs")
        result = {name: stored[name].copy() for name in stored.files}
    for name, value in result.items():
        expected = record["members"][name]
        require(list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"]
            and array_identity(np, value) == expected["values_sha256"], "Small query schema/value differs")
    ledger.complete(token); return result


def load_inputs(np, torch, record, ledger):
    from .elastic_pilot import _load_array
    manifest = read(record["input_manifest"]["path"]); cache = record["input_cache"]
    ledger.bump("input_cache_byte_hash_reads")
    require(stat_matches(cache) and sha(cache["path"]) == cache["sha256"], "Accepted target-free input bytes changed")
    ledger.bump("input_cache_reads", cache["path"])
    result = {}
    with np.load(path(cache["path"]), allow_pickle=False) as stored:
        for name in INPUT_MEMBERS:
            token = cache["path"] + ":" + name; ledger.bump("input_member_reads", token)
            value = np.array(stored[name], copy=True); expected = record["input_members"][name]
            require(list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"]
                and array_identity(np, value) == expected["sha256"], "Accepted full input member differs: " + name)
            result[name] = value; ledger.complete(token)
    ledger.complete(cache["path"])
    result["X"] = torch.as_tensor(result["vertices_mm"], device="cuda", dtype=torch.float64)
    result["cells"] = torch.as_tensor(result["cells10"], device="cuda", dtype=torch.int64)
    for name in ("gradients", "weights"):
        value = record["operator_arrays"][name]; ledger.bump("operator_array_reads", value["path"])
        result[name] = _load_array(np, torch, value, torch.device("cuda:0")); ledger.complete(value["path"])
    result["L_mm"] = manifest["contract"]["expected_geometry"]["characteristic_length_mm"]
    return result


def load_physical_U(np, record, ledger):
    field = record["physical_U"]; ledger.bump("physical_U_byte_hash_reads")
    require(stat_matches(field) and sha(field["path"]) == field["file_sha256"], "Original immutable zero-start physical U differs")
    ledger.bump("physical_U_prediction_reads", field["path"])
    value = np.load(path(field["path"]), allow_pickle=False)
    require(value.shape == (record["nodes"], 3) and value.dtype == np.float64 and np.isfinite(value).all(), "Original FP64 mm U invalid")
    ledger.complete(field["path"]); return value


def quadrature_basis(np, torch, geometry):
    a, b = (5 + 3 * math.sqrt(5)) / 20, (5 - math.sqrt(5)) / 20
    bary = np.array([[a if i == q else b for i in range(4)] for q in range(4)], dtype=np.float64)
    return torch.as_tensor(geometry.quadratic_shape(bary), device="cuda", dtype=torch.float64)


def engineering(torch, gradient):
    return torch.stack((gradient[..., 0, 0], gradient[..., 1, 1], gradient[..., 2, 2],
        gradient[..., 0, 1] + gradient[..., 1, 0], gradient[..., 0, 2] + gradient[..., 2, 0],
        gradient[..., 1, 2] + gradient[..., 2, 1]), dim=-1)


def query_full(np, torch, geometry, inputs, nodal, timer, include_X=False):
    """Every cell/fourQ, original rows, original accepted physical gradients."""
    count = len(inputs["cells"]); basis = quadrature_basis(np, torch, geometry)
    nodal_gpu = torch.as_tensor(nodal, device="cuda", dtype=torch.float64)
    result = {"U_mm": np.empty((count, 4, 3), dtype=np.float64),
        "gradient_dimensionless": np.empty((count, 4, 3, 3), dtype=np.float64),
        "engineering_strain6": np.empty((count, 4, 6), dtype=np.float64)}
    if include_X:
        result["X_mm"] = np.empty((count, 4, 3), dtype=np.float64)
    for start in range(0, count, 8192):
        timer.guard(); stop = min(start + 8192, count); cells = inputs["cells"][start:stop]
        grad = inputs["gradients"][start:stop]; weights = inputs["weights"][start:stop]
        if not (torch.isfinite(grad).all() and torch.isfinite(weights).all() and (weights > 0).all()):
            raise ScientificNegative("Nonfinite accepted4Q gradient or nonpositive physical volume weight")
        local = nodal_gpu[cells]
        uq = torch.einsum("qi,tia->tqa", basis, local)
        gq = torch.einsum("tia,tqib->tqab", local, grad)
        eq = engineering(torch, gq)
        require(torch.isfinite(uq).all().item() and torch.isfinite(gq).all().item() and torch.isfinite(eq).all().item(),
            "Actual full4Q continuous field nonfinite")
        for name, value in (("U_mm", uq), ("gradient_dimensionless", gq), ("engineering_strain6", eq)):
            result[name][start:stop] = value.cpu().numpy()
        if include_X:
            result["X_mm"][start:stop] = torch.einsum("qi,tia->tqa", basis, inputs["X"][cells]).cpu().numpy()
    return result


def numpy_engineering(np, gradient):
    return np.stack((gradient[..., 0, 0], gradient[..., 1, 1], gradient[..., 2, 2], gradient[..., 0, 1] + gradient[..., 1, 0],
        gradient[..., 0, 2] + gradient[..., 2, 0], gradient[..., 1, 2] + gradient[..., 2, 1]), axis=-1)


def native_query(np, geometry, vertices_mm, cells10, nodal_U_mm, element_rows, barycentric):
    """Known-cell arbitrary P2 query: physical X/U/G/engineering strain, FP64.

    Physical-coordinate callers first obtain element/barycentric roots from the
    explicit bounded locator. Shared-face roots have one-sided gradients.
    """
    rows = np.asarray(element_rows, dtype=np.int64); bary = np.asarray(barycentric, dtype=np.float64)
    require(rows.ndim == 1 and bary.shape == (len(rows), 4) and len(rows) > 0
        and rows.min() >= 0 and rows.max() < len(cells10) and np.isfinite(bary).all()
        and (bary >= -1e-10).all() and np.max(np.abs(bary.sum(1)-1)) <= 1e-12,
        "Native known-cell physical query shape/containment differs")
    ids = cells10[rows]; nodes = np.asarray(vertices_mm[ids], dtype=np.float64)
    basis = geometry.quadratic_shape(bary); D = geometry.shape_gradient(bary); J = geometry.jacobian(nodes, bary)
    try:
        gradient = np.linalg.solve(J.transpose(0, 2, 1), D.transpose(0, 2, 1)).transpose(0, 2, 1)
    except np.linalg.LinAlgError as error:
        raise ScientificNegative("Actual native physical query J inverse failed") from error
    if not np.isfinite(gradient).all():
        raise ScientificNegative("Actual native physical query gradient nonfinite")
    local = np.asarray(nodal_U_mm[ids], dtype=np.float64)
    X = np.einsum("qi,qia->qa", basis, nodes); U = np.einsum("qi,qia->qa", basis, local)
    G = np.einsum("qia,qib->qab", local, gradient)
    return {"rows": rows, "node_ids": ids, "barycentric": bary, "basis": basis, "physical_gradients": gradient,
        "X_mm": X, "U_mm": U, "gradient_dimensionless": G, "engineering_strain6": numpy_engineering(np, G)}


def small_queries(np, geometry, inputs, nodal, timer):
    cells, vertices = inputs["cells10"], inputs["vertices_mm"]
    scan = geometry.scan_face_conformity(cells, len(vertices), max_records=8)
    if scan["status"] != "passed":
        raise ScientificNegative("Actual native face-query topology unsupported: " + scan["status"])
    selected, faces = geometry.select_cells_and_faces(scan, len(cells), max_cells=32, max_faces=8)
    rows, barys, face_pairs, normals = [], [], [], []
    for row in selected:
        for bary in ([.1, .2, .3, .4], [.4, .3, .2, .1]):
            rows.append(int(row)); barys.append(bary)
    interior_count = len(rows)
    for face in faces:
        corners = sorted(face["corner_ids"])
        for face_bary in ([1/3, 1/3, 1/3], [.2, .3, .5]):
            pair = []
            for row in face["cell_rows"]:
                local = cells[row, :4].tolist(); bary = [0.0] * 4
                for corner, value in zip(corners, face_bary):
                    bary[local.index(corner)] = value
                pair.append(len(rows)); rows.append(int(row)); barys.append(bary)
            face_pairs.append(pair)
            row = face["cell_rows"][0]; local = cells[row, :4].tolist()
            ds, dt = np.zeros(4), np.zeros(4)
            for corner, s, t in zip(corners, (-1, 1, 0), (-1, 0, 1)):
                ds[local.index(corner)] = s; dt[local.index(corner)] = t
            J = geometry.jacobian(vertices[cells[row]], np.array(barys[pair[0]], dtype=np.float64))
            normal = np.cross(J @ ds[1:], J @ dt[1:]); norm = float(np.linalg.norm(normal))
            if not math.isfinite(norm) or norm == 0:
                raise ScientificNegative("Actual shared P2 face has invalid tangent normal")
            normals.append(normal / norm)
    timer.guard(); row_array = np.array(rows, dtype=np.int64); bary = np.array(barys, dtype=np.float64)
    state = native_query(np, geometry, vertices, cells, nodal, row_array, bary)
    X, U, G = state["X_mm"], state["U_mm"], state["gradient_dimensionless"]
    pairs = np.asarray(face_pairs, dtype=np.int64).reshape(-1, 2); normal = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    diagnostics = {"status": "completed", "selected_original_cells": selected.tolist(), "selected_cells": len(selected),
        "interior_query_occurrences": interior_count, "shared_face_physical_queries": len(pairs), "shared_face_sided_occurrences": 2 * len(pairs),
        "total_query_occurrences": len(rows), "whole_CAD_fast_locator_claim": False,
        "face_topology": {name: scan[name] for name in ("cell_count", "interior_face_count", "boundary_face_count", "midside_mismatch_count", "nonmanifold_face_count")},
        "C1_required": False, "normal_derivative_jump_is_diagnostic": True}
    if len(pairs):
        left, right = pairs[:, 0], pairs[:, 1]
        x_gap = np.linalg.norm(X[left] - X[right], axis=1) / inputs["L_mm"]
        u_den = np.maximum(1.0, np.maximum(np.abs(U[left]).max(axis=1), np.abs(U[right]).max(axis=1)))
        u_gap = np.abs(U[left] - U[right]).max(axis=1) / u_den
        jump = np.einsum("qab,qb->qa", G[left] - G[right], normal)
        diagnostics.update(C0_max_position_gap_over_L=float(x_gap.max()), C0_max_U_trace_scaled_gap=float(u_gap.max()),
            C0_passed=bool((x_gap <= 1e-9).all() and (u_gap <= 1e-9).all()),
            normal_derivative_jump_RMS_dimensionless=float(np.sqrt(np.square(jump).sum(axis=1).mean())),
            normal_derivative_jump_max_dimensionless=float(np.linalg.norm(jump, axis=1).max()))
    else:
        diagnostics.update(C0_passed=False, C0_missing_reason="No shared face in fixed selected scope")
    # Physical inversion is bounded to these selected cells; no full-CAD search claim.
    locator = geometry.GeometryLocator(vertices, cells[selected], selected, inputs["L_mm"], descriptor_blocks={"U_mm": nodal})
    located = locator.locate(X)
    statuses = {}; max_gap = 0.0; valid = 0; interior_gradient_gap = 0.0; interior_gradient_count = 0; one_sided_roots = []
    for index, item in enumerate(located):
        status = item.get("status"); statuses[status] = statuses.get(status, 0) + 1
        if status == "inside_accepted":
            interpolated = locator.interpolate_result(item)["U_mm"]
            gap = float(np.max(np.abs(interpolated - U[index])) / max(1.0, float(np.max(np.abs(U[index])))))
            max_gap = max(max_gap, gap); valid += 1
            recovered = native_query(np, geometry, vertices, cells, nodal, [item["element_row"]], [item["barycentric"]])
            if index < interior_count:
                interior_gradient_count += 1
                interior_gradient_gap = max(interior_gradient_gap,
                    float(np.max(np.abs(recovered["gradient_dimensionless"][0] - G[index])) / max(1.0, float(np.max(np.abs(G[index]))))))
            else:
                one_sided_roots.append({"query_occurrence": index, "accepted_element_row": int(item["element_row"]),
                    "direct_side_element_row": int(row_array[index])})
    diagnostics.update(inverse_status_counts=statuses, inverse_valid_queries=valid, inverse_max_direct_U_scaled_gap=max_gap,
        inverse_replay_passed=valid == len(rows) and max_gap <= 1e-7, inverse_counters=locator.counters,
        inverse_interior_gradient_scaled_gap_diagnostic=interior_gradient_gap if interior_gradient_count else None,
        inverse_interior_gradient_evaluated_queries=interior_gradient_count,
        inverse_shared_face_accepted_sides=one_sided_roots,
        inverse_sampled_regularity_not_global_uniqueness_proof=True)
    state.update(face_pairs=pairs, common_face_normals=normal)
    return state, diagnostics


def predict_case(np, torch, geometry, key, contract, output, ledger, clock):
    folder = output / key.replace("/", "_"); folder.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "case_key": key}; write(folder / "config.json", config)
    manifest_file = folder / "prediction_manifest.json"
    if manifest_file.exists():
        saved = read(manifest_file); require(saved["config"] == config and saved["status"] == "completed", "Committed candidate identity differs")
        require(all(stat_matches(value) for value in saved["arrays"].values()) and stat_matches(saved["small_queries"]),
            "Committed native candidate arrays changed")
        return saved
    timer = CaseClock(folder, clock, "candidate_query_and_persist")
    try:
        timer.guard(); record = contract["records"][key]; inputs = load_inputs(np, torch, record, ledger)
        nodal = load_physical_U(np, record, ledger)
        fields = query_full(np, torch, geometry, inputs, nodal, timer, include_X=True)
        arrays = {name: save_array(np, folder / ("candidate_" + name + ".npy"), value) for name, value in fields.items()}
        small, diagnostics = small_queries(np, geometry, inputs, nodal, timer)
        small_record = save_small(np, folder / "candidate_small_queries.npz", small)
        write(folder / "geometry_diagnostics.json", diagnostics)
        write(folder / "query_manifest.json", {"config": config, "all_original_cells": record["cells"], "fourQ_queries": 4 * record["cells"],
            "order": "original cell row then accepted Q index", "dtype": "float64", "U_units": "mm", "gradient_units": "dimensionless",
            "engineering_strain_channels": ["xx", "yy", "zz", "gamma_xy", "gamma_xz", "gamma_yz"], "small_query_state": small_record,
            "original_U_file_sha256": record["physical_U"]["file_sha256"], "no_new_solver_or_K_guard": True})
        timer.guard()
        saved = {"status": "completed", "config": config, "full_Q": {"cells": record["cells"], "quadrature_points": 4 * record["cells"], "complete": True},
            "arrays": arrays, "small_queries": small_record, "geometry_diagnostics": diagnostics, "persisted_unix_seconds": time.time(),
            "original_physical_U_file_sha256": record["physical_U"]["file_sha256"], "GT_U_reads_at_prediction_commit": ledger.data["GT_U_reads"],
            "read_receipt": dict(ledger.data)}
        require(ledger.data["GT_U_reads"] == 0, "Candidate commit reached after new GT source read")
        write(manifest_file, saved); write(folder / "read_receipt.json", ledger.data)
        write(folder / "status.json", {"status": "predicted_pending_shared_reference", "full_Q": saved["full_Q"]})
        del inputs, nodal, fields, small; torch.cuda.empty_cache(); return saved
    finally:
        timer.finish()


def commit_barrier(output, contract, predictions, ledger):
    file = output / "prediction_barrier.json"
    inventory = {key: sha(output / key.replace("/", "_") / "prediction_manifest.json") for key in CASES}
    for key in CASES:
        require(predictions[key]["config"] == {"contract": contract, "case_key": key}
            and predictions[key]["full_Q"]["complete"] is True
            and predictions[key]["GT_U_reads_at_prediction_commit"] == 0, "Full-five prelabel candidate barrier invalid")
    if file.exists():
        require(read(file)["prediction_manifest_sha256"] == inventory, "Persisted all-five barrier inventory changed")
    else:
        require(ledger.data["GT_U_reads"] == 0, "Source GT U read before shared all-five barrier")
        write(file, {"passed": True, "case_keys": list(CASES), "prediction_manifest_sha256": inventory,
            "all_five_candidates_persisted_before_any_new_GT_U": True, "committed_unix_seconds": time.time()})


def read_GT_U(np, record, folder, config, ledger, output):
    require(read(output / "prediction_barrier.json")["passed"] is True, "Shared GT U read lacks all-five prediction barrier")
    claim = folder / "source_GT_U_read_started.json"
    if claim.exists():
        raise RuntimeError("Uncommitted prior source GT U read cannot be silently repeated")
    source = record["deferred_GT_U"]; require(stat_matches(source), "Original canonical GT displacement stat changed")
    write(claim, {"config": config, "member": "displacement", "maximum_source_reads": 1, "attempted": True})
    ledger.bump("source_GT_U_read_attempts", source["path"] + ":displacement")
    with np.load(path(source["path"]), allow_pickle=False) as archive:
        ledger.bump("GT_U_reads"); value = np.array(archive["displacement"], dtype=np.float64)
    require(value.shape == (record["nodes"], 3) and np.isfinite(value).all(), "Original mm GT U shape/finite mismatch")
    ledger.bump("source_GT_U_read_completed"); ledger.complete(source["path"] + ":displacement")
    write(claim, {"config": config, "member": "displacement", "maximum_source_reads": 1, "attempted": True, "completed": True})
    return value


def proxy_small(np, state, truth):
    local = truth[state["node_ids"]]
    U = np.einsum("qi,qia->qa", state["basis"], local)
    G = np.einsum("qia,qib->qab", local, state["physical_gradients"])
    return {"U_mm": U, "gradient_dimensionless": G, "engineering_strain6": numpy_engineering(np, G)}


def weighted_metrics(np, torch, prediction, reference, weights, timer):
    sums = {name: {"error": 0.0, "reference": 0.0, "maximum": 0.0} for name in FIELDS + ("symmetric_strain_Frobenius",)}
    volume = 0.0; eng_weights = torch.tensor([1, 1, 1, .5, .5, .5], device="cuda", dtype=torch.float64)
    count = len(weights)
    for start in range(0, count, 8192):
        timer.guard(); stop = min(start + 8192, count); w = weights[start:stop]; volume += float(w.sum().item())
        require(torch.isfinite(w).all().item() and (w > 0).all().item(), "Accepted physical comparison weights invalid")
        for name in FIELDS:
            p = torch.as_tensor(np.array(prediction[name][start:stop], copy=True), device="cuda", dtype=torch.float64)
            r = torch.as_tensor(np.array(reference[name][start:stop], copy=True), device="cuda", dtype=torch.float64)
            require(torch.isfinite(p).all().item() and torch.isfinite(r).all().item(), "Nonfinite native/proxy comparison")
            dimensions = tuple(range(2, p.ndim)); e2 = (p-r).square().sum(dim=dimensions); r2 = r.square().sum(dim=dimensions)
            sums[name]["error"] += float((w * e2).sum().item()); sums[name]["reference"] += float((w * r2).sum().item())
            sums[name]["maximum"] = max(sums[name]["maximum"], float(e2.max().sqrt().item()))
            if name == "engineering_strain6":
                e2 = ((p-r).square() * eng_weights).sum(-1); r2 = (r.square() * eng_weights).sum(-1)
                sums["symmetric_strain_Frobenius"]["error"] += float((w * e2).sum().item())
                sums["symmetric_strain_Frobenius"]["reference"] += float((w * r2).sum().item())
                sums["symmetric_strain_Frobenius"]["maximum"] = max(sums["symmetric_strain_Frobenius"]["maximum"], float(e2.max().sqrt().item()))
    require(math.isfinite(volume) and volume > 0, "Positive physical quadrature volume missing")
    require(all(all(math.isfinite(value) and value >= 0 for value in item.values()) for item in sums.values()),
        "Nonfinite full4Q proxy energy or maximum")
    return {name: {"volume_weighted_relative_l2": None if value["reference"] == 0 else math.sqrt(value["error"] / value["reference"]),
        "zero_reference": value["reference"] == 0, "volume_weighted_RMS_error": math.sqrt(value["error"] / volume),
        "max_query_error_norm": value["maximum"], "reference_weighted_squared_energy": value["reference"],
        "error_weighted_squared_energy": value["error"], "quadrature_volume_mm3": volume,
        "error_units": "mm" if name == "U_mm" else "dimensionless", "reference_is_nodal_P2_proxy": True} for name, value in sums.items()}


def small_proxy_diagnostics(np, candidate, reference):
    result = {"reference_is_nodal_P2_proxy": True, "not_source_strain_GT": True}
    for name in FIELDS:
        delta = candidate[name] - reference[name]
        error = float(np.square(delta).sum()); energy = float(np.square(reference[name]).sum())
        result[name] = {"relative_l2": None if energy == 0 else math.sqrt(error / energy), "zero_reference": energy == 0,
            "max_abs_error": float(np.abs(delta).max())}
    pairs, normal = candidate["face_pairs"], candidate["common_face_normals"]
    if len(pairs):
        left, right = pairs[:, 0], pairs[:, 1]
        jump = np.einsum("qab,qb->qa", reference["gradient_dimensionless"][left] - reference["gradient_dimensionless"][right], normal)
        result["GT_nodal_proxy_normal_derivative_jump_RMS"] = float(np.sqrt(np.square(jump).sum(axis=1).mean()))
        result["GT_nodal_proxy_normal_derivative_jump_max"] = float(np.linalg.norm(jump, axis=1).max())
        result["normal_jump_is_C0_FE_diagnostic_not_C1_gate"] = True
    return result


def reference_case(np, torch, geometry, key, contract, prediction, output, ledger, clock):
    folder = output / key.replace("/", "_"); config = {"contract": contract, "case_key": key}
    summary_file = folder / "summary.json"
    if summary_file.exists():
        summary = read(summary_file); require(summary["config"] == config and summary["status"] == "completed", "Committed proxy metric identity differs")
        return summary
    timer = CaseClock(folder, clock, "shared_reference_proxy_and_metrics")
    try:
        timer.guard(); record = contract["records"][key]; inputs = load_inputs(np, torch, record, ledger)
        predicted = {name: own_array(np, prediction["arrays"][name], ledger) for name in FIELDS}
        small_candidate = own_small(np, prediction["small_queries"], ledger)
        ref_file = folder / "reference_manifest.json"
        if ref_file.exists():
            manifest = read(ref_file); require(manifest["config"] == config, "Committed reference proxy identity differs")
            reference = {name: own_array(np, manifest["arrays"][name], ledger, True) for name in FIELDS}
            small_reference = own_small(np, manifest["small_queries"], ledger, True)
        else:
            truth = read_GT_U(np, record, folder, config, ledger, output)
            reference = query_full(np, torch, geometry, inputs, truth, timer)
            small_reference = proxy_small(np, small_candidate, truth)
            arrays = {name: save_array(np, folder / ("GT_nodal_P2_proxy_" + name + ".npy"), value) for name, value in reference.items()}
            small_record = save_small(np, folder / "GT_nodal_P2_proxy_small.npz", small_reference)
            manifest = {"status": "completed", "config": config, "arrays": arrays, "small_queries": small_record,
                "reference_source": record["deferred_GT_U"], "source_GT_U_reads": 1,
                "definition": "same full10 P2 basis/physical gradients applied to original nodal GT U mm; not source strain GT",
                "all_five_prediction_barrier_sha256": sha(output / "prediction_barrier.json")}
            write(ref_file, manifest); del truth
        metrics = weighted_metrics(np, torch, predicted, reference, inputs["weights"], timer)
        small_metrics = small_proxy_diagnostics(np, small_candidate, small_reference)
        timer.guard(); diagnostics = prediction["geometry_diagnostics"]
        complete = prediction["full_Q"]["complete"] and diagnostics["C0_passed"] and diagnostics["inverse_replay_passed"]
        write(folder / "metrics.json", {"config": config, "full4Q": metrics, "small_queries": small_metrics})
        summary = {"status": "completed", "config": config, "full_Q": prediction["full_Q"], "metrics": metrics,
            "geometry_diagnostics": diagnostics, "small_proxy_diagnostics": small_metrics,
            "capability_integrity_passed": bool(complete), "GT_accuracy_is_proxy_only": True, "source_strain_GT_claim": False,
            "old_node_U_GT_metric_recomputed": False, "costs": {}, "read_receipt": dict(ledger.data)}
        write(folder / "read_receipt.json", ledger.data)
        del inputs, predicted, reference, small_candidate, small_reference; torch.cuda.empty_cache()
    finally:
        timer.finish()
    summary["costs"] = read(folder / "costs.json")
    query_seconds = summary["costs"]["phases"].get("candidate_query_and_persist", 0.0)
    summary["costs"].update(cached_continuous_candidate_query_wall_seconds=query_seconds,
        composed_original_physics_plus_continuous_query_seconds=record["inherited_original_GT_free_physical_seconds"] + query_seconds,
        fresh_combined_cold_measurement=False, source_GT_proxy_metrics_excluded_from_deployment_cost=True,
        query_cost_scope="all original4Q field persistence and fixed bounded inverse/face diagnostics; not single-point latency",
        inherited_original_zero_GT_free_physical_seconds=record["inherited_original_GT_free_physical_seconds"])
    write(summary_file, summary); write(folder / "status.json", {"status": "completed", "capability_integrity_passed": summary["capability_integrity_passed"],
        "costs": summary["costs"]}); return summary


def completion_gate(cases, negative=None):
    complete = len(cases) == len(CASES) and all(cases.get(key, {}).get("capability_integrity_passed") is True for key in CASES)
    return {"passed": negative is None and complete, "fixed_denominator": 5, "completed_cases": len(cases),
        "all_five_full4Q_and_bounded_query_integrity": complete, "GT_accuracy_is_nodal_P2_proxy": True,
        "normal_derivative_continuity_required": False, "source_strain_GT_certified": False,
        "final_solver_selection": False, "terminal_negative_reason": negative}


def brief(output, summary):
    lines = ["# Physical native TET10 continuous U / gradient", "", "Every original cell/fourQ uses FP64 mm geometry and original zero-start U.",
        "The reference is the same P2 evaluation of GT nodal U; strain is a derived proxy.", "",
        "| CAD/ver | FullQ U L2 | Strain Frobenius proxy L2 | Cached query s |", "|---|---:|---:|---:|"]
    fmt = lambda value: "zero-ref/missing" if not finite(value) else f"{100*value:.4f}%"
    for key in CASES:
        case = summary["cases"].get(key, {}); metrics = case.get("metrics", {}); costs = case.get("costs", {})
        query = costs.get("cached_continuous_candidate_query_wall_seconds")
        lines.append(f"| {key} | {fmt(metrics.get('U_mm', {}).get('volume_weighted_relative_l2'))} | {fmt(metrics.get('symmetric_strain_Frobenius', {}).get('volume_weighted_relative_l2'))} | {'missing' if not finite(query) else f'{query:.3f}'} |")
    lines += ["", f"Coverage/integrity gate: {summary['completion_gate']['passed']}; fixed denominator5.",
        "Normal derivative jumps are expected C0 finite-element diagnostics; C1 is not required.",
        "Native all-cell barycentric evaluation and a bounded32-cell inverse were checked; a fast whole-CAD locator is not claimed.",
        f"New actual wall {summary['costs']['new_actual_wall_seconds']:.3f}s; original GT-free physical452.954s inherited once.",
        "Deployment cold costs compose original physics and measured query costs; source GT metrics are separate."]
    if summary["completion_gate"]["terminal_negative_reason"]:
        lines.append("Fixed candidate stopped: " + summary["completion_gate"]["terminal_negative_reason"])
    require(len(lines) <= 25, "Small report line limit exceeded")
    (output / "BRIEF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "metrics.md").write_text("\n".join(lines[5:12]) + "\n", encoding="utf-8")


def run(plan_path=PLAN, output=None):
    began = time.perf_counter(); initial_plan = read(plan_path)
    output = path(output or initial_plan["output"]); require(output == path(OUTPUT), "Frozen continuous output differs")
    output.mkdir(parents=True, exist_ok=True); clock = Clock(output, began); ledger = Ledger(output)
    contract = None; cases = {}; predictions = {}; negative = None; technical = None
    try:
        contract, plan = request_contract(plan_path); config = {"contract": contract}
        marker = read(plan["root_review_marker"])
        require(marker.get("passed") is True and marker.get("source_sha256") == contract["source_sha256"][SOURCE]
            and marker.get("scientific_digest") == SCIENCE_DIGEST
            and marker.get("decision") == "proceed_frozen_full_TET10_continuous_U_gradient_capability", "Real root execution release missing")
        if (output / "config.json").exists():
            require(read(output / "config.json") == config, "Persistent capability contract differs")
        write(output / "config.json", config)
        if (output / "summary.json").exists():
            old = read(output / "summary.json")
            if old.get("status") == "completed":
                require(old["config"] == config, "Completed capability identity differs"); return old
        clock.guard()
        import numpy as np
        import torch
        from . import continuous_query_geometry as geometry
        require(torch.cuda.is_available(), "Frozen GPU backend unavailable")
        clock.torch = torch; torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(); torch.set_num_threads(8)
        for key in CASES:
            clock.guard(); predictions[key] = predict_case(np, torch, geometry, key, contract, output, ledger, clock)
        commit_barrier(output, contract, predictions, ledger); clock.persist(phase="all_five_candidates_committed")
        for key in CASES:
            clock.guard(); cases[key] = reference_case(np, torch, geometry, key, contract, predictions[key], output, ledger, clock)
        clock.guard()
    except ScientificNegative as error:
        negative = str(error)
    except Exception as error:
        if "out of memory" in str(error).lower():
            negative = "Fixed GPU backend memory negative: " + str(error)
        else:
            technical = type(error).__name__ + ": " + str(error)
    finally:
        clock.persist(False, "exit"); ledger.persist()
    summary = {"status": "failed" if technical else "completed", "kind": KIND, "config": {"contract": contract},
        "case_keys": list(CASES), "cases": {key: cases.get(key, {"status": "missing", "capability_integrity_passed": False}) for key in CASES},
        "predictions": predictions, "completion_gate": completion_gate(cases, negative or technical), "costs": read(clock.file),
        "read_receipt": ledger.data, "technical_error": technical, "old_node_U_GT_metric_recomputed": False}
    write(output / "summary.json", summary); brief(output, summary); clock.persist(False, "report_persisted")
    summary["costs"] = read(clock.file)
    if summary["costs"]["new_actual_wall_seconds"] >= 300 and technical is None:
        summary["completion_gate"] = completion_gate(cases, "Fixed new-actual300s budget exhausted including report persistence")
    write(output / "summary.json", summary); write(output / "status.json", {"status": summary["status"], "completion_gate": summary["completion_gate"],
        "costs": summary["costs"], "technical_error": technical})
    if technical:
        raise RuntimeError(technical)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", default=PLAN); parser.add_argument("--output")
    args = parser.parse_args(); result = run(args.plan, args.output)
    print(json.dumps({"status": result["status"], "completion_gate": result["completion_gate"], "costs": result["costs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
