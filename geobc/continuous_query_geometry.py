"""Isolated NumPy geometry for the frozen bounded continuous-query gate.

No file, model, target, Torch or CUDA access occurs in this module.  The
locator works on its explicitly supplied submesh, not an implicit whole CAD.
Regularity samples and two seeds do not certify a globally unique inverse.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import numpy as np


PLAN_SHA256 = "16169bbd20fc1d746cc8ccf88fa9d4c287fa410744d4ee8b6f0bb210543bb71f"
POLICY_SHA256 = "13dd1a57dd368a9f44e74841408a30d6ac95da335fcba97a9cc1825e4eaf446f"
EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], dtype=np.int64)
FACES = np.array([[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7],
                  [0, 2, 3, 6, 9, 7], [1, 2, 3, 5, 9, 8]], dtype=np.int64)
REFERENCE_BARYCENTRIC = np.concatenate((np.eye(4), (np.eye(4)[EDGES[:, 0]]
                                                   + np.eye(4)[EDGES[:, 1]]) / 2))
REFERENCE_POINTS = REFERENCE_BARYCENTRIC[:, 1:]
REGULARITY_BARYCENTRIC = np.concatenate((REFERENCE_BARYCENTRIC, np.full((1, 4), .25)))
_DL = np.array([[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def _bary(values):
    result = np.asarray(values)
    if result.shape[-1:] != (4,) or not np.isfinite(result).all():
        raise ValueError("Expected finite barycentric coordinates [...,4]")
    return result if result.dtype.kind == "f" else result.astype(np.float64)


def quadratic_shape(barycentric):
    """All ten signed VTK-order quadratic Lagrange weights, preserving FP32."""
    b = _bary(barycentric)
    return np.concatenate((b * (2 * b - 1),
                           4 * b[..., EDGES[:, 0]] * b[..., EDGES[:, 1]]), axis=-1)


def shape_gradient(barycentric):
    """dN/dr for r=(L1,L2,L3), shape [...,10,3]."""
    b = _bary(barycentric)
    d = _DL.astype(b.dtype)
    corners = (4 * b - 1)[..., :, None] * d
    mids = 4 * (b[..., EDGES[:, 1], None] * d[EDGES[:, 0]]
                + b[..., EDGES[:, 0], None] * d[EDGES[:, 1]])
    return np.concatenate((corners, mids), axis=-2)


def map_points(nodes, barycentric):
    return np.einsum("...i,...ic->...c", quadratic_shape(barycentric), np.asarray(nodes))


def jacobian(nodes, barycentric):
    """Physical J[c,r]=dX_c/dr_r, not its transpose."""
    return np.einsum("...ir,...ic->...cr", shape_gradient(barycentric), np.asarray(nodes))


def bezier_bounds(nodes):
    """Conservative degree-two control-point AABB; no corner-only surrogate."""
    x = np.asarray(nodes, dtype=np.float64)
    if x.shape[-2:] != (10, 3) or not np.isfinite(x).all():
        raise ValueError("Expected finite element nodes [...,10,3]")
    controls = 2 * x[..., 4:, :] - .5 * (x[..., EDGES[:, 0], :] + x[..., EDGES[:, 1], :])
    all_controls = np.concatenate((x[..., :4, :], controls), axis=-2)
    return all_controls.min(axis=-2), all_controls.max(axis=-2)


def scan_face_conformity(cells10, node_count, max_records=8):
    """Scan all supplied topology; retain only the first ordered face records.

    A face record's local_faces are indices into FACES, not corner indices.
    Edge-associated midside IDs are checked, not merely unordered six-ID sets.
    This function performs no coordinate or Jacobian calculation.
    """
    cells = np.asarray(cells10)
    if cells.ndim != 2 or cells.shape[1] != 10 or cells.dtype.kind not in "iu" or not len(cells):
        raise ValueError("Expected nonempty integer cells10[T,10]")
    if node_count < 1 or max_records < 0:
        raise ValueError("Invalid topology scan bounds")
    if cells.min() < 0 or cells.max() >= node_count:
        return {"status": "invalid_geometry", "reason": "node_index_out_of_bounds", "interior_faces": []}
    duplicates = np.flatnonzero(np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0, axis=1))
    if len(duplicates):
        return {"status": "invalid_geometry", "reason": "repeated_local_node_ID",
                "first_mismatches": duplicates[:10].tolist(), "interior_faces": []}
    facets = cells[:, FACES].reshape(-1, 6)
    keys = np.sort(facets[:, :3], axis=1)
    order = np.lexsort((np.arange(len(keys)), keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    bad_multiplicity = np.flatnonzero(counts > 2)
    pair_groups = np.flatnonzero(counts == 2)
    left = order[starts[pair_groups]]
    right = order[starts[pair_groups] + 1]

    def canonical_mid_ids(indices):
        face = facets[indices]
        corners = np.sort(face[:, :3], axis=1)
        expected_lo = corners[:, [0, 0, 1]]
        expected_hi = corners[:, [1, 2, 2]]
        edge_lo = np.minimum(face[:, [0, 1, 2]], face[:, [1, 2, 0]])
        edge_hi = np.maximum(face[:, [0, 1, 2]], face[:, [1, 2, 0]])
        match = ((expected_lo[:, :, None] == edge_lo[:, None, :])
                 & (expected_hi[:, :, None] == edge_hi[:, None, :]))
        return np.take_along_axis(face[:, 3:], match.argmax(axis=-1), axis=1)

    mismatches = np.flatnonzero(np.any(canonical_mid_ids(left) != canonical_mid_ids(right), axis=1))
    records = [{"corner_ids": keys[a].tolist(), "cell_rows": [int(a // 4), int(b // 4)],
                "local_faces": [int(a % 4), int(b % 4)]}
               for a, b in zip(left[:max_records], right[:max_records])]
    first_mismatches = [{"corner_ids": keys[left[i]].tolist(),
                         "cell_rows": [int(left[i] // 4), int(right[i] // 4)]}
                        for i in mismatches[:10]]
    return {"status": "passed" if not len(bad_multiplicity) and not len(mismatches) else "invalid_geometry",
            "cell_count": int(len(cells)), "facet_occurrences": int(len(facets)),
            "boundary_face_count": int(np.count_nonzero(counts == 1)),
            "interior_face_count": int(len(pair_groups)),
            "nonmanifold_face_count": int(len(bad_multiplicity)),
            "midside_mismatch_count": int(len(mismatches)),
            "first_nonmanifold_corner_ids": sorted_keys[starts[bad_multiplicity[:10]]].tolist(),
            "first_mismatches": first_mismatches, "interior_faces": records,
            "retained_face_records": len(records), "all_paired_faces_checked": True}


def select_cells_and_faces(scan, total_cells, max_cells=32, max_faces=8):
    if scan.get("status") != "passed":
        # A reached topology failure is a completed scientific negative.
        # Do not select a repaired subset or execute numeric queries afterward.
        return np.empty(0, dtype=np.int64), []
    selected_faces = scan["interior_faces"][:max_faces]
    chosen = {int(row) for item in selected_faces for row in item["cell_rows"]}
    if len(chosen) > max_cells:
        raise ValueError("Face incident cells exceed the fixed numeric cell bound")
    for row in range(total_cells):
        if len(chosen) >= min(max_cells, total_cells):
            break
        chosen.add(row)
    return np.asarray(sorted(chosen), dtype=np.int64), selected_faces


def interpolate_nodal(values, cells10, element_rows, barycentric, dtype=np.float64):
    """Signed full10 gather/sum; element_rows index the provided connectivity."""
    nodes = np.asarray(values, dtype=dtype)
    cells = np.asarray(cells10, dtype=np.int64)
    rows = np.asarray(element_rows, dtype=np.int64)
    weights = np.asarray(quadratic_shape(np.asarray(barycentric, dtype=np.float64)), dtype=dtype)
    return (nodes[cells[rows]] * weights[..., :, None]).sum(axis=-2, dtype=dtype)


def post_ln_surrogate(node_ids):
    i = np.asarray(node_ids, dtype=np.float64)[:, None]
    k = np.arange(32, dtype=np.float64)[None, :]
    s = np.sin((i + 1) * (k + 1) * .17) + .1 * np.cos((i + 1 + k) * .07)
    centered = s - s.mean(axis=1, keepdims=True)
    return np.tanh(centered / np.sqrt((centered ** 2).mean(axis=1, keepdims=True) + 1e-5))


@dataclass(frozen=True)
class LocatorConfig:
    residual_tolerance: float = 1e-10
    stop_tolerance: float = 1e-13
    reference_tolerance: float = 1e-10
    partition_tolerance: float = 1e-12
    determinant_minimum: float = 1e-18
    sigma_minimum: float = 1e-12
    condition_maximum: float = 1e12
    tie_tolerance: float = 1e-7
    root_equivalence_tolerance: float = 1e-7
    iterations: int = 24
    backtracks: int = 8
    maximum_cells: int = 32
    maximum_queries: int = 512

    @classmethod
    def from_mapping(cls, config):
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        alias = {"physical_inverse_final_residual_over_L_maximum": "residual_tolerance",
                 "Newton_stop_residual_over_L_target": "stop_tolerance",
                 "reference_containment_tolerance": "reference_tolerance",
                 "shape_partition_and_direct_replay_relative_tolerance": "partition_tolerance",
                 "jacobian_sigma_min_over_L_minimum": "sigma_minimum",
                 "jacobian_condition_maximum": "condition_maximum",
                 "inverse_tie_trace_relative_tolerance": "tie_tolerance",
                 "same_element_reference_root_equivalence_tolerance": "root_equivalence_tolerance",
                 "iterations_per_seed_maximum": "iterations", "backtracks_per_iteration_maximum": "backtracks",
                 "maximum_original_cells": "maximum_cells", "maximum_query_occurrences": "maximum_queries"}
        names = {item.name for item in fields(cls)}
        values = {alias.get(name, name): value for name, value in config.items()
                  if alias.get(name, name) in names}
        return cls(**values)


def _number(value):
    return float(value) if np.isfinite(value) else None


def _spectral(matrix, config):
    if not np.isfinite(matrix).all():
        return {"regular": False, "determinant": None, "sigma_min": None,
                "condition": None, "sign": 0}
    try:
        determinant = float(np.linalg.det(matrix))
        singular = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"regular": False, "determinant": None, "sigma_min": None,
                "condition": None, "sign": 0}
    minimum = float(singular[-1])
    condition = float(singular[0] / minimum) if minimum > 0 else np.inf
    regular = (np.isfinite(determinant) and abs(determinant) > config.determinant_minimum
               and minimum > config.sigma_minimum and condition < config.condition_maximum)
    return {"regular": bool(regular), "determinant": _number(determinant),
            "sigma_min": _number(minimum), "condition": _number(condition),
            "sign": int(np.sign(determinant)) if abs(determinant) > config.determinant_minimum else 0}


class GeometryLocator:
    """Bounded physical quadratic locator with explicit failure diagnostics.

    cells10 are already selected rows; original_rows names those rows in the
    original mesh. Descriptor blocks use the global node indices of cells10.
    global_bounds, if supplied, is the whole-CAD conservative bound, not the
    selected-cell bounding box. Every invalid result has no field payload.
    """
    def __init__(self, xyz_mm, cells10, original_rows, L_scale, config=None,
                 global_bounds=None, descriptor_blocks: Mapping | None = None):
        self.config = LocatorConfig.from_mapping(config)
        self.xyz_mm = np.asarray(xyz_mm, dtype=np.float64)
        source_cells = np.asarray(cells10)
        source_rows = np.asarray(original_rows)
        if source_cells.dtype.kind not in "iu" or source_rows.dtype.kind not in "iu":
            raise ValueError("Connectivity and original row IDs must be integer inputs")
        self.cells10 = source_cells.astype(np.int64, copy=False)
        self.original_rows = source_rows.astype(np.int64, copy=False)
        self.L_scale = float(L_scale)
        if (self.xyz_mm.ndim != 2 or self.xyz_mm.shape[1] != 3 or not np.isfinite(self.xyz_mm).all()
                or self.cells10.ndim != 2 or self.cells10.shape[1] != 10 or not len(self.cells10)
                or self.original_rows.shape != (len(self.cells10),)
                or len(np.unique(self.original_rows)) != len(self.original_rows)
                or (self.original_rows < 0).any() or self.cells10.min() < 0
                or self.cells10.max() >= len(self.xyz_mm) or not np.isfinite(self.L_scale) or self.L_scale <= 0):
            raise ValueError("Invalid explicit source geometry, rows or characteristic scale")
        if len(self.cells10) > self.config.maximum_cells:
            raise ValueError("Selected numeric cell count exceeds the fixed scope")
        if np.any(np.diff(np.sort(self.cells10, axis=1), axis=1) == 0):
            raise ValueError("Repeated original node ID within a cell")
        self.nodes = self.xyz_mm[self.cells10]
        self.lower, self.upper = bezier_bounds(self.nodes)
        self.global_bounds = None
        if global_bounds is not None:
            lo, hi = (np.asarray(value, dtype=np.float64) for value in global_bounds)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or (lo > hi).any():
                raise ValueError("Invalid explicitly supplied global conservative bounds")
            self.global_bounds = (lo, hi)
        self.descriptor_blocks = {}
        for name, values in (descriptor_blocks or {}).items():
            array = np.asarray(values)
            if array.ndim != 2 or len(array) != len(self.xyz_mm) or not np.isfinite(array).all():
                raise ValueError("Descriptor blocks must be finite globally node-indexed [N,C] arrays")
            self.descriptor_blocks[str(name)] = array
        self.counters = {name: 0 for name in ("logical_queries", "candidate_evaluations", "newton_attempts",
                         "newton_iterations", "backtrack_evaluations", "jacobian_evaluations",
                         "map_evaluations", "descriptor_evaluations", "affine_seed_checks")}
        self.regularity_records = [self._regularity(index) for index in range(len(self.nodes))]

    def _regularity(self, index):
        spectra = []
        for b in REGULARITY_BARYCENTRIC:
            self.counters["jacobian_evaluations"] += 1
            spectra.append(_spectral(jacobian(self.nodes[index], b) / self.L_scale, self.config))
        signs = {item["sign"] for item in spectra if item["sign"]}
        # Aggregate all eleven first: an evidenced fold outranks singular center.
        status = "invalid_geometry" if len(signs) > 1 else (
            "passed" if all(item["regular"] for item in spectra) else "invalid_jacobian")
        determinants = [abs(item["determinant"]) for item in spectra if item["determinant"] is not None]
        minima = [item["sigma_min"] for item in spectra if item["sigma_min"] is not None]
        conditions = [item["condition"] for item in spectra if item["condition"] is not None]
        return {"element_row": int(self.original_rows[index]), "original_row": int(self.original_rows[index]),
                "status": status, "valid": status == "passed",
                "orientation_sign": int(spectra[-1]["sign"]), "sample_count": 11,
                "sample_signs": [item["sign"] for item in spectra], "samples": spectra,
                "minimum_abs_det_over_L3": min(determinants) if determinants else None,
                "minimum_sigma_over_L": min(minima) if minima else None,
                "maximum_condition": max(conditions) if conditions else None,
                "min_abs_det_over_L3": min(determinants) if determinants else None,
                "min_sigma_min_over_L": min(minima) if minima else None,
                "max_condition": max(conditions) if conditions else None,
                "global_validity_proven": False}

    def _map(self, index, bary):
        self.counters["map_evaluations"] += 1
        return map_points(self.nodes[index], bary)

    def _jac(self, index, bary):
        self.counters["jacobian_evaluations"] += 1
        matrix = jacobian(self.nodes[index], bary)
        return matrix, _spectral(matrix / self.L_scale, self.config)

    def _attempt(self, index, point, seed_name, reference):
        self.counters["newton_attempts"] += 1
        r = np.array(reference, dtype=np.float64, copy=True)
        reason = "iteration_bound"
        iterations = 0
        for _ in range(self.config.iterations):
            bary = np.r_[1 - r.sum(), r]
            value = self._map(index, bary)
            residual = value - point
            norm = float(np.max(np.abs(residual)) / self.L_scale)
            if not np.isfinite(norm):
                reason = "nonfinite_residual"
                break
            if norm <= self.config.stop_tolerance:
                reason = "stop_residual_target"
                break
            matrix, spectrum = self._jac(index, bary)
            if not spectrum["regular"]:
                reason = "invalid_iteration_jacobian"
                break
            try:
                delta = np.linalg.solve(matrix, residual)
            except np.linalg.LinAlgError:
                reason = "singular_iteration_solve"
                break
            if not np.isfinite(delta).all():
                reason = "nonfinite_newton_step"
                break
            self.counters["newton_iterations"] += 1
            iterations += 1
            moved = False
            for backtrack in range(self.config.backtracks):
                proposal = r - (2. ** -backtrack) * delta
                self.counters["backtrack_evaluations"] += 1
                if not np.isfinite(proposal).all():
                    continue
                proposed_bary = np.r_[1 - proposal.sum(), proposal]
                proposed_residual = self._map(index, proposed_bary) - point
                proposed_norm = float(np.max(np.abs(proposed_residual)) / self.L_scale)
                if np.isfinite(proposed_norm) and (proposed_norm < norm or proposed_norm <= self.config.stop_tolerance):
                    r = proposal
                    moved = True
                    break
            if not moved:
                reason = "no_residual_decreasing_backtrack"
                break
        bary = np.r_[1 - r.sum(), r]
        residual = float(np.max(np.abs(self._map(index, bary) - point)) / self.L_scale)
        _, spectrum = self._jac(index, bary)
        contained = bool((bary >= -self.config.reference_tolerance).all()
                         and (bary <= 1 + self.config.reference_tolerance).all())
        orientation = spectrum["sign"] == self.regularity_records[index]["orientation_sign"]
        partition_error = float(abs(quadratic_shape(bary).sum() - 1))
        accepted = bool(np.isfinite(bary).all() and np.isfinite(residual)
                        and residual <= self.config.residual_tolerance and contained
                        and partition_error <= self.config.partition_tolerance and spectrum["regular"] and orientation)
        return {"element_row": int(self.original_rows[index]), "local_element_index": int(index),
                "seed": seed_name, "accepted": accepted, "barycentric": bary.tolist(),
                "residual_over_L": _number(residual), "iterations": iterations, "termination": reason,
                "contained": contained, "orientation_consistent": bool(orientation),
                "evidenced_domain_sign_reversal": bool(contained and spectrum["sign"]
                                                       and not orientation),
                "invalid_domain_jacobian": bool(contained and not spectrum["regular"]),
                "partition_error": partition_error, "final_jacobian": spectrum}

    def _blocks(self, root):
        index = root["local_element_index"]
        b = np.asarray(root["barycentric"], dtype=np.float64)
        self.counters["descriptor_evaluations"] += len(self.descriptor_blocks)
        return {name: interpolate_nodal(values, self.cells10, np.asarray(index), b)
                for name, values in self.descriptor_blocks.items()}

    def _trace_difference(self, first, second):
        a, b = self._blocks(first), self._blocks(second)
        largest = 0.
        for name in a:
            largest = max(largest, float(np.max(np.abs(a[name] - b[name])) / max(1., np.max(np.abs(b[name])))))
        x_a = self._map(first["local_element_index"], first["barycentric"])
        x_b = self._map(second["local_element_index"], second["barycentric"])
        return max(largest, float(np.max(np.abs(x_a - x_b)) / self.L_scale))

    def _compatible_entity(self, first, second):
        ia, ib = first["local_element_index"], second["local_element_index"]
        ba, bb = (np.asarray(item["barycentric"]) for item in (first, second))
        active_a = set(self.cells10[ia, :4][np.abs(ba) > self.config.tie_tolerance].tolist())
        active_b = set(self.cells10[ib, :4][np.abs(bb) > self.config.tie_tolerance].tolist())
        if active_a != active_b or not 1 <= len(active_a) <= 3:
            return False
        def mids(cell):
            return {tuple(sorted((int(cell[a]), int(cell[b])))): int(cell[4 + j])
                    for j, (a, b) in enumerate(EDGES) if int(cell[a]) in active_a and int(cell[b]) in active_a}
        return mids(self.cells10[ia]) == mids(self.cells10[ib])

    @staticmethod
    def _failure(status, diagnostics):
        return {"status": status, "element_row": None, "local_element_index": None,
                "barycentric": None, "weights": None, "residual_over_L": None,
                "field_blocks": None, "diagnostics": diagnostics}

    def _locate_one(self, point):
        padding = self.L_scale * self.config.reference_tolerance
        if not np.isfinite(point).all():
            return self._failure("invalid_geometry", {"reason": "nonfinite_physical_query"})
        if self.global_bounds is not None:
            lo, hi = self.global_bounds
            if ((point < lo - padding) | (point > hi + padding)).any():
                return self._failure("outside_certified", {"reason": "excluded_by_global_conservative_bounds",
                                      "candidate_rows": [], "accepted_candidate_rows": []})
        candidates = np.flatnonzero(np.all((point >= self.lower - padding) & (point <= self.upper + padding), axis=1))
        diagnostics = {"candidate_rows": self.original_rows[candidates].tolist(), "seed_attempts": [],
                       "accepted_candidate_rows": [], "compatible_tie_trace_max": 0.,
                       "same_element_reference_root_equivalence_tolerance": self.config.root_equivalence_tolerance,
                       "global_inverse_uniqueness_proven": False}
        if not len(candidates):
            status = "outside_certified" if self.global_bounds is None else "unresolved_inverse"
            diagnostics["reason"] = "excluded_by_declared_submesh_bounds" if self.global_bounds is None else "no_selected_candidate_inside_global_bounds"
            return self._failure(status, diagnostics)
        self.counters["candidate_evaluations"] += len(candidates)
        invalid = [self.regularity_records[i] for i in candidates if self.regularity_records[i]["status"] != "passed"]
        if invalid:
            diagnostics["invalid_candidate_regularity"] = invalid
            status = "invalid_geometry" if any(item["status"] == "invalid_geometry" for item in invalid) else "invalid_jacobian"
            return self._failure(status, diagnostics)
        accepted = []
        for index in candidates:
            affine = (self.nodes[index, 1:4] - self.nodes[index, 0]).T
            seeds = []
            self.counters["affine_seed_checks"] += 1
            if _spectral(affine / self.L_scale, self.config)["regular"]:
                seeds.append(("affine_corner_if_regular", np.linalg.solve(affine, point - self.nodes[index, 0])))
            seeds.append(("reference_center", np.full(3, .25)))
            for seed_name, seed in seeds:
                attempt = self._attempt(index, point, seed_name, seed)
                diagnostics["seed_attempts"].append(attempt)
                if attempt["accepted"]:
                    accepted.append(attempt)
        diagnostics["accepted_candidate_rows"] = sorted({item["element_row"] for item in accepted})
        if any(item["evidenced_domain_sign_reversal"] for item in diagnostics["seed_attempts"]):
            diagnostics["reason"] = "evidenced_contained_reference_sign_reversal"
            return self._failure("invalid_geometry", diagnostics)
        if not accepted:
            # A bad Jacobian along an outside Newton path is no inside/fold certificate.
            invalid_jac = any(item["invalid_domain_jacobian"] for item in diagnostics["seed_attempts"])
            return self._failure("invalid_jacobian" if invalid_jac else "unresolved_inverse", diagnostics)
        # Keep every seed root in diagnostics before any numerical equivalence.
        for a_index, a in enumerate(accepted):
            for b in accepted[a_index + 1:]:
                if a["local_element_index"] == b["local_element_index"]:
                    difference = float(np.max(np.abs(np.asarray(a["barycentric"]) - b["barycentric"])))
                    if difference > self.config.root_equivalence_tolerance:
                        diagnostics["reason"] = "distinct_accepted_reference_roots_in_same_element"
                        diagnostics["root_reference_separation"] = difference
                        return self._failure("ambiguous_geometry", diagnostics)
                elif not self._compatible_entity(a, b):
                    diagnostics["reason"] = "accepted_roots_not_on_identical_global_shared_entity"
                    return self._failure("ambiguous_geometry", diagnostics)
                trace = self._trace_difference(a, b)
                diagnostics["compatible_tie_trace_max"] = max(diagnostics["compatible_tie_trace_max"], trace)
                if trace > self.config.tie_tolerance:
                    diagnostics["reason"] = "accepted_root_descriptor_traces_disagree"
                    return self._failure("ambiguous_geometry", diagnostics)
        chosen = min(accepted, key=lambda item: (item["element_row"], item["residual_over_L"], item["seed"]))
        bary = np.asarray(chosen["barycentric"], dtype=np.float64)
        return {"status": "inside_accepted", "element_row": chosen["element_row"],
                "local_element_index": chosen["local_element_index"], "barycentric": bary,
                "weights": quadratic_shape(bary), "residual_over_L": chosen["residual_over_L"],
                "field_blocks": self._blocks(chosen), "diagnostics": diagnostics}

    def locate(self, points):
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Physical queries must be [Q,3]")
        if self.counters["logical_queries"] + len(values) > self.config.maximum_queries:
            return [self._failure("resource_limit", {"reason": "logical_query_bound"}) for _ in values]
        results = []
        for point in values:
            before = self.counters.copy()
            self.counters["logical_queries"] += 1
            result = self._locate_one(point)
            result["diagnostics"]["counters"] = {key: self.counters[key] - before[key] for key in self.counters}
            results.append(result)
        return results

    def interpolate_result(self, result):
        """No field is supplied for failure; accepted fields are already computed."""
        return result["field_blocks"] if result["status"] == "inside_accepted" else None


def synthetic_fixture():
    """One author-run CPU-only fixture; never opens scientific source arrays."""
    straight = REFERENCE_POINTS.copy()
    shape = quadratic_shape(REFERENCE_BARYCENTRIC)
    cardinal_error = float(np.max(np.abs(shape - np.eye(10))))
    partition_error = float(np.max(np.abs(quadratic_shape(np.array([[.25] * 4, [.1, .2, .3, .4]])).sum(1) - 1)))
    gradient_sum_error = float(np.max(np.abs(shape_gradient(REGULARITY_BARYCENTRIC).sum(axis=1))))
    affine_jacobian_error = float(np.max(np.abs(jacobian(straight, REGULARITY_BARYCENTRIC) - np.eye(3))))
    xyz = np.concatenate((straight, [[0., 0., -1.], [0., 0., -.5], [.5, 0., -.5], [0., .5, -.5]]))
    cells = np.array([list(range(10)), [0, 1, 2, 10, 4, 5, 6, 11, 12, 13]], dtype=np.int64)
    conformity = scan_face_conformity(cells, len(xyz))
    blocks = {"raw43": np.repeat(np.arange(len(xyz))[:, None] / 10., 43, axis=1),
              "Q32": post_ln_surrogate(np.arange(len(xyz))), "cachedxyz": xyz.astype(np.float32)}
    locator = GeometryLocator(xyz, cells, [9, 3], 1., descriptor_blocks=blocks)
    face = locator.locate(np.array([[.3, .5, 0.]]))[0]
    nodal = locator.locate(straight)
    replay_error = max(float(np.max(np.abs(item["field_blocks"]["raw43"] - blocks["raw43"][i])))
                       for i, item in enumerate(nodal) if item["status"] == "inside_accepted")
    singular = straight.copy(); singular[:, 2] = 0
    folded = straight.copy(); folded[:, 0] = straight[:, 0] - 2 * straight[:, 0] ** 2
    failure_cases = [
        ("singular", singular, np.arange(10)[None], [0], np.array([[.2, .2, 0.]]), "invalid_jacobian"),
        ("evidenced_fold", folded, np.arange(10)[None], [0], np.array([[.08, .1, .1]]), "invalid_geometry"),
        ("overlapping_interiors", np.concatenate((straight, straight)), np.arange(20).reshape(2, 10),
         [0, 1], np.array([[.25, .25, .25]]), "ambiguous_geometry"),
        ("unaccepted_conservative_candidate", straight, np.arange(10)[None], [0],
         np.array([[.8, .8, .8]]), "unresolved_inverse")]
    failures = []
    for name, vertices, connectivity, rows, query, expected in failure_cases:
        failure_locator = GeometryLocator(vertices, connectivity, rows, 1.)
        result = failure_locator.locate(query)[0]
        failures.append({"name": name, "expected": expected, "actual": result["status"],
                         "no_field": failure_locator.interpolate_result(result) is None,
                         "passed": result["status"] == expected and result["field_blocks"] is None,
                         "counters": failure_locator.counters})
    curved = straight.copy(); curved[4] += [.02, .01, .01]
    curved_locator = GeometryLocator(curved, np.arange(10)[None], [4], 1.)
    b = np.array([.1, .2, .3, .4])
    curved_result = curved_locator.locate(map_points(curved, b)[None])[0]
    curved_inverse_error = (float(np.max(np.abs(curved_result["barycentric"] - b)))
                            if curved_result["status"] == "inside_accepted" else None)
    spike = np.r_[np.zeros(4), np.ones(6)]
    spike_error = float(np.max(np.abs(shape @ spike - spike)))
    fp32_error = float(np.max(np.abs(interpolate_nodal(blocks["raw43"], cells,
                                      np.zeros(10, np.int64), REFERENCE_BARYCENTRIC, np.float32)
                                   - blocks["raw43"][:10])))
    passed = (max(cardinal_error, partition_error, gradient_sum_error, affine_jacobian_error, replay_error, spike_error) <= 1e-12
              and conformity["status"] == "passed" and face["status"] == "inside_accepted" and face["element_row"] == 3
              and locator.regularity_records[1]["orientation_sign"] == -1
              and all(item["status"] == "inside_accepted" for item in nodal)
              and all(item["passed"] for item in failures) and fp32_error <= 2e-6
              and curved_inverse_error is not None and curved_inverse_error <= 1e-7)
    receipt = {"status": "passed" if passed else "failed", "source_arrays_read": False, "CUDA_imported": False,
               "cardinal_error": cardinal_error, "partition_error": partition_error,
               "gradient_sum_error": gradient_sum_error, "affine_J_error": affine_jacobian_error,
               "raw43_nodal_replay_error": replay_error, "midside_spike_error": spike_error,
               "FP32_replay_error": fp32_error, "curved_inverse_reference_error": curved_inverse_error,
               "negative_orientation_accepted": locator.regularity_records[1]["orientation_sign"] == -1,
               "shared_face_smallest_original_row": face["element_row"], "four_failure_queries": failures,
               "limits": "Small mathematical fixture only; no real CAD or global inverse/native model proof."}
    if not passed:
        raise AssertionError(receipt)
    return receipt


__all__ = ["PLAN_SHA256", "POLICY_SHA256", "EDGES", "FACES", "REFERENCE_BARYCENTRIC", "REFERENCE_POINTS",
           "REGULARITY_BARYCENTRIC", "quadratic_shape", "shape_gradient", "map_points", "jacobian",
           "bezier_bounds", "scan_face_conformity", "select_cells_and_faces", "interpolate_nodal",
           "post_ln_surrogate", "LocatorConfig", "GeometryLocator", "synthetic_fixture"]
