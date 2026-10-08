"""Map coordinate-aligned original DeepJEB labels onto a quadratic FEM grid.

Topology comes from original H5 TETRA10 cells. Curved isoparametric geometry
is inverted by Newton iterations inside conservative quadratic Bezier bounds.
No Delaunay reconstruction, point-cloud nearest-label assignment, or linear
tetrahedral surrogate defines occupancy or targets.

The original slide experiment's 18-channel schema is unavailable. This tool
records an explicit replacement: occupancy + prepared features[0:16] +
features[24]. Targets stay in source physical units. Geometry caches are
shared across load conditions of the same CAD and grid resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch

from geobc.fno import FNO_SOURCES, query_grid
from tools.mesh_boundary import TETRA10_EDGES, midpoint_order_report


ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"
VTK_SOURCE = "https://github.com/Kitware/VTK/blob/517e98882892a079f639d40fb70cb76a6cf95a0e/Common/DataModel/vtkQuadraticTetra.cxx"
FEATURE_NAMES = (
    ["quadratic_FEM_occupancy"] + [f"support_reference_constrained_dof_{i}" for i in range(1, 7)]
    + ["loaded_interface_mask"] + [f"total_force_{axis}_over_42300N" for axis in "xyz"]
    + [f"total_moment_{axis}_over_565000Nmm" for axis in "xyz"]
    + ["E_over_113800MPa", "poisson_ratio", "density_over_4.47e-9_t_per_mm3",
       "characteristic_length_over_200mm"]
)
GRID_NPZ_SCHEMA = {
    "inputs": "float32 [18,R,R,R], channel-first; zero outside mask; no xyz channels",
    "targets": "float32 [C,R,R,R], physical; C=4 uXYZ(mm),signedstress(MPa), or C=1 signedstress(MPa)",
    "mask": "bool [R,R,R], inverse quadratic isoparametric FEM occupancy",
    "occupied_flat_index": "int32 [Q], C-order flat index: (ix*R+iy)*R+iz",
    "occupied_xyz": "float32 [Q,3], normalized physical coordinates",
    "tetra_index": "int32 [Q], original H5 cell row",
    "barycentric": "float32 [Q,4], quadratic reference coordinates L0,L1,L2,L3",
    "shape_weights": "float32 [Q,10], N_i=L_i(2L_i-1), N_ij=4L_iL_j in VTK edge order",
    "node_xyz": "float32 [N,3], normalized original H5 coordinates",
    "node_xyz_mm": "float32 [N,3], source coordinates in mm",
    "node_targets": "float32 [N,C], original coordinate-aligned physical CSV labels",
    "grid_node_ids": "int64 [N], original CSV GRID IDs matching H5 vertex rows",
    "query_indices": "int32 [N,8], regular grid corners for every original FEM node",
    "query_weights": "float32 [N,8], trilinear weights times corner occupancy, renormalized",
    "query_coverage": "bool [N], at least one nonzero occupied corner weight",
    "target_mean": "float64 [C], population mean of selected occupied physical grid labels",
    "target_std": "float64 [C], population standard deviation, floor 1e-8",
    "bbox_min_mm": "float64 [3], unpadded source bbox minimum",
    "characteristic_length_mm": "float64 scalar, largest unpadded source bbox extent",
    "domain_lower_mm": "float64 [3], physical lower cube corner; centered padding is an explicit origin assumption",
    "domain_length_mm": "float64 scalar, physical cube length used by BOTH occupied_xyz and node_xyz",
    "grid_spacing_mm": "float64 scalar, domain_length_mm/(R-1)",
    "field_names": "unicode [C], displacement_x/y/z then signed_stress, or signed_stress",
    "field_units": "unicode [C], mm/mm/mm/MPa or MPa",
    "sample_key": "unicode scalar, CAD/load",
    "metadata_json": "unicode scalar, provenance, actual algorithm, schema caveat, roundtrip metrics",
}


def quadratic_shape(barycentric: torch.Tensor) -> torch.Tensor:
    edges = torch.as_tensor(TETRA10_EDGES, device=barycentric.device)
    corners = barycentric * (2 * barycentric - 1)
    midsides = 4 * barycentric[:, edges[:, 0]] * barycentric[:, edges[:, 1]]
    return torch.cat((corners, midsides), dim=1)


def _shape_gradient(barycentric: torch.Tensor) -> torch.Tensor:
    derivative = barycentric.new_tensor([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    edges = torch.as_tensor(TETRA10_EDGES, device=barycentric.device)
    corners = (4 * barycentric - 1)[:, :, None] * derivative[None]
    midsides = 4 * (barycentric[:, edges[:, 1], None] * derivative[edges[:, 0]][None]
                   + barycentric[:, edges[:, 0], None] * derivative[edges[:, 1]][None])
    return torch.cat((corners, midsides), dim=1)


def _solve_columns(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    first, second, third = matrix.unbind(dim=-1)
    cross23, cross31, cross12 = (torch.linalg.cross(second, third),
                               torch.linalg.cross(third, first), torch.linalg.cross(first, second))
    determinant = (first * cross23).sum(-1)
    safe = torch.where(determinant.abs() > 1e-25, determinant, torch.ones_like(determinant))
    answer = torch.stack(((rhs * cross23).sum(-1), (rhs * cross31).sum(-1),
                          (rhs * cross12).sum(-1)), dim=-1) / safe[:, None]
    return torch.where((determinant.abs() > 1e-25)[:, None], answer, torch.zeros_like(answer))


@torch.no_grad()
def build_geometry_grid(xyz_mm: np.ndarray, cells: np.ndarray, resolution: int,
                        device: torch.device | str = "cuda", pair_batch_size: int = 131072,
                        newton_iterations: int = 10, tolerance: float = 2e-7,
                        domain_length_mm: float | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if resolution < 4 or pair_batch_size < 1 or newton_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid geometry grid configuration")
    xyz_mm = np.asarray(xyz_mm, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    order = midpoint_order_report(cells, xyz_mm)
    if order["midside_deviation_over_edge_p50"] > 0.05 or order["midside_deviation_over_edge_p95"] > 0.25:
        raise ValueError("Source cells do not follow the validated VTK TETRA10 edge order")
    source_lower, source_upper = xyz_mm.min(axis=0), xyz_mm.max(axis=0)
    characteristic_length = float(np.ptp(xyz_mm, axis=0).max())
    if characteristic_length <= 0:
        raise ValueError("Degenerate source bounding box")
    if domain_length_mm is not None:
        if not np.isfinite(domain_length_mm) or domain_length_mm < characteristic_length:
            raise ValueError("Configured cube length must be finite and contain the full source bbox")
        length = float(domain_length_mm)
        lower = 0.5 * (source_lower + source_upper) - 0.5 * length
    else:
        length, lower = characteristic_length, source_lower
    normalized = (xyz_mm - lower) / length
    nodes = normalized[cells]
    corners = nodes[:, :4]
    control = 2 * nodes[:, 4:] - 0.5 * (corners[:, TETRA10_EDGES[:, 0]] + corners[:, TETRA10_EDGES[:, 1]])
    bezier = np.concatenate((corners, control), axis=1)
    starts = np.maximum(np.ceil(bezier.min(1) * (resolution - 1) - 1e-7).astype(np.int64), 0)
    ends = np.minimum(np.floor(bezier.max(1) * (resolution - 1) + 1e-7).astype(np.int64), resolution - 1)
    dimensions = np.maximum(ends - starts + 1, 0)
    counts = dimensions.prod(axis=1)
    cumulative_np = np.cumsum(counts, dtype=np.int64)
    total_pairs = int(cumulative_np[-1])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    began = time.perf_counter()
    nodes_t = torch.as_tensor(nodes, dtype=torch.float64, device=device)
    starts_t = torch.as_tensor(starts, device=device)
    dims_t = torch.as_tensor(dimensions, device=device)
    cumulative = torch.as_tensor(cumulative_np, device=device)
    offsets = torch.cat((cumulative.new_zeros(1), cumulative[:-1]))
    affine = (nodes_t[:, 1:4] - nodes_t[:, 0:1]).transpose(-2, -1)
    inverse, inversion_info = torch.linalg.inv_ex(affine)
    inverse = torch.nan_to_num(inverse)
    best_tetra = torch.full((resolution ** 3,), len(cells), dtype=torch.int64, device=device)
    reference = torch.zeros((resolution ** 3, 4), dtype=torch.float32, device=device)
    for begin in range(0, total_pairs, pair_batch_size):
        pair = torch.arange(begin, min(begin + pair_batch_size, total_pairs), device=device)
        tetra = torch.searchsorted(cumulative, pair, right=True)
        local = pair - offsets[tetra]
        dimensions_b = dims_t[tetra]
        ix = starts_t[tetra, 0] + local // (dimensions_b[:, 1] * dimensions_b[:, 2])
        iy = starts_t[tetra, 1] + (local // dimensions_b[:, 2]) % dimensions_b[:, 1]
        iz = starts_t[tetra, 2] + local % dimensions_b[:, 2]
        flat = (ix * resolution + iy) * resolution + iz
        points = torch.stack((ix, iy, iz), dim=1).to(torch.float64) / (resolution - 1)
        element_nodes = nodes_t[tetra]
        q = torch.einsum("bij,bj->bi", inverse[tetra], points - element_nodes[:, 0])
        q = torch.where((inversion_info[tetra] == 0)[:, None], q, torch.full_like(q, 0.25))
        for _ in range(newton_iterations):
            barycentric = torch.cat((1 - q.sum(1, keepdim=True), q), dim=1)
            mapped = torch.einsum("bn,bnc->bc", quadratic_shape(barycentric), element_nodes)
            jacobian = torch.einsum("bnr,bnc->bcr", _shape_gradient(barycentric), element_nodes)
            delta = _solve_columns(jacobian, mapped - points)
            q = torch.nan_to_num(q - delta.clamp(-1, 1), nan=0.25, posinf=2, neginf=-1)
        barycentric = torch.cat((1 - q.sum(1, keepdim=True), q), dim=1)
        mapped = torch.einsum("bn,bnc->bc", quadratic_shape(barycentric), element_nodes)
        valid = ((mapped - points).abs().amax(1) <= tolerance) & (barycentric.amin(1) >= -5 * tolerance)
        valid &= torch.isfinite(barycentric).all(1) & (barycentric.amax(1) <= 1 + 5 * tolerance)
        found_flat, found_tetra, found_bary = flat[valid], tetra[valid], barycentric[valid]
        best_tetra.scatter_reduce_(0, found_flat, found_tetra, reduce="amin", include_self=True)
        winners = found_tetra == best_tetra[found_flat]
        reference[found_flat[winners]] = found_bary[winners].float()
    occupied = torch.nonzero(best_tetra < len(cells), as_tuple=False).flatten()
    if not len(occupied):
        raise ValueError("No quadratic FEM grid points found; inspect grid spacing and source topology")
    barycentric = reference[occupied]
    indices = best_tetra[occupied]
    grid_xyz = torch.stack((occupied // (resolution ** 2), (occupied // resolution) % resolution,
                            occupied % resolution), dim=1).float() / (resolution - 1)
    shape_weights = quadratic_shape(barycentric)
    reconstructed = torch.einsum("bn,bnc->bc", shape_weights.double(), nodes_t[indices])
    geometry_error = float((reconstructed - grid_xyz.double()).abs().amax())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - began
    cache = {
        "occupied_flat_index": occupied.cpu().numpy().astype(np.int32),
        "occupied_xyz": grid_xyz.cpu().numpy(),
        "tetra_index": indices.cpu().numpy().astype(np.int32),
        "barycentric": barycentric.cpu().numpy(), "shape_weights": shape_weights.cpu().numpy(),
        "bbox_min_mm": source_lower, "source_bbox_max_mm": source_upper,
        "characteristic_length_mm": np.asarray(characteristic_length),
        "domain_lower_mm": lower, "domain_length_mm": np.asarray(length),
        "grid_spacing_mm": np.asarray(length / (resolution - 1)),
        "resolution": np.asarray(resolution, dtype=np.int32),
    }
    report = {
        "resolution": resolution, "source_tetrahedra": len(cells), "candidate_pairs": total_pairs,
        "occupied_grid_points": len(occupied), "spacing_mm": length / (resolution - 1),
        "grid_normalization": "isotropic cube: xyz_mm=domain_lower_mm+domain_length_mm*[ix,iy,iz]/(R-1), endpoints included",
        "domain_lower_mm": lower.tolist(), "domain_length_mm": length,
        "unpadded_source_max_extent_mm": characteristic_length,
        "domain_origin_assumption": "centered padded cube; original experiment origin is unavailable" if domain_length_mm is not None else "unpadded cube anchored at source bbox minimum",
        "geometry": "original curved quadratic isoparametric TETRA10; Newton inverse, not linear corner tetrahedra",
        "broad_phase": "conservative degree-2 Bezier convex-hull AABBs rasterized to candidate grid/tetra pairs",
        "shared_face_tie_break": "smallest original tetrahedron row index",
        "newton_dtype": "float64", "cache_dtype": "float32 barycentric and shape weights",
        "newton_iterations": newton_iterations, "normalized_tolerance": tolerance,
        "maximum_reconstructed_coordinate_error_normalized": geometry_error,
        "corner_singular_tetrahedra": int((inversion_info != 0).sum()),
        "point_location_seconds": seconds, "device": str(device), "VTK_source": VTK_SOURCE,
        "midpoint_order": order,
    }
    return cache, report


def masked_trilinear_cache(node_xyz: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = mask.shape[0]
    if mask.shape != (resolution,) * 3:
        raise ValueError("Expected a cubic occupancy mask")
    positions = np.asarray(node_xyz, dtype=np.float64) * (resolution - 1)
    if positions.min() < -1e-5 or positions.max() > resolution - 1 + 1e-5:
        raise ValueError("Source query nodes are outside the normalized cube")
    positions = np.clip(positions, 0, resolution - 1)
    lower = np.minimum(np.floor(positions).astype(np.int64), resolution - 2)
    fraction = positions - lower
    corners = np.asarray([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
    coordinates = lower[:, None] + corners[None]
    indices = (coordinates[:, :, 0] * resolution + coordinates[:, :, 1]) * resolution + coordinates[:, :, 2]
    weights = np.where(corners[None].astype(bool), fraction[:, None], 1 - fraction[:, None]).prod(2)
    weights *= mask.reshape(-1)[indices]
    denominator = weights.sum(1)
    coverage = denominator > 1e-12
    weights = np.divide(weights, denominator[:, None], out=np.zeros_like(weights), where=coverage[:, None])
    return indices.astype(np.int32), weights.astype(np.float32), coverage


@torch.no_grad()
def interpolate_nodal(values: np.ndarray, cells: np.ndarray, cache: dict[str, np.ndarray],
                      device: torch.device | str, chunk_size: int = 65536) -> np.ndarray:
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device)
    cells_t = torch.as_tensor(cells, dtype=torch.long, device=device)
    rows = torch.as_tensor(cache["tetra_index"], dtype=torch.long, device=device)
    weights = torch.as_tensor(cache["shape_weights"], device=device)
    result = np.empty((len(rows), values_t.shape[1]), dtype=np.float32)
    for begin in range(0, len(rows), chunk_size):
        end = min(begin + chunk_size, len(rows))
        result[begin:end] = (values_t[cells_t[rows[begin:end]]] * weights[begin:end, :, None]).sum(1).cpu().numpy()
    return result


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _geometry_fingerprint(xyz: np.ndarray, cells: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(xyz).tobytes())
    digest.update(np.ascontiguousarray(cells).tobytes())
    return digest.hexdigest()


def prepare_case(prepared: Path, raw: Path, output: Path, cache_dir: Path, sample_key: str,
                 resolution: int = 64, fields: str = "joint", device: str = "cuda",
                 pair_batch_size: int = 131072, domain_length_mm: float | None = None) -> Path:
    started = time.perf_counter()
    cad, load = sample_key.split("/")
    source = prepared / f"{cad}_{load}.npz"
    target = output / f"{cad}_{load}_r{resolution}_{fields}.npz"
    if target.exists():
        raise FileExistsError(f"Prepared grid already exists: {target}")
    with np.load(source, allow_pickle=False) as archive:
        if "label_alignment" not in archive or str(archive["label_alignment"].item()) != ALIGNMENT:
            raise ValueError("Grid preparation requires coordinate-aligned original CSV labels; legacy H5 label order is invalid")
        xyz_mm = np.asarray(archive["xyz_mm"], dtype=np.float32)
        features = np.asarray(archive["features"], dtype=np.float32)
        stress = np.asarray(archive["stress"], dtype=np.float32)
        grid_node_ids = np.asarray(archive["grid_node_ids"], dtype=np.int64)
        if fields == "joint":
            u = np.asarray(archive["u"], dtype=np.float32)
            values = np.column_stack((u, stress))
            field_names, field_units = ["displacement_x", "displacement_y", "displacement_z", "signed_stress"], ["mm", "mm", "mm", "MPa"]
        else:
            values = stress[:, None]
            field_names, field_units = ["signed_stress"], ["MPa"]
    with h5py.File(raw / "FieldMesh" / f"{cad}.h5", "r") as source_h5:
        h5_xyz = source_h5["vertices"][:].astype(np.float32)
        cells = source_h5["cells"][:].astype(np.int64)
    if h5_xyz.shape != xyz_mm.shape or not np.array_equal(h5_xyz, xyz_mm):
        raise ValueError("Aligned label coordinates must retain the original H5 vertex order for TETRA10 cells")
    if features.shape != (len(xyz_mm), 25) or values.shape[0] != len(xyz_mm) or not np.isfinite(values).all():
        raise ValueError("Invalid prepared physical labels or 25-channel conditioning")
    fingerprint = _geometry_fingerprint(xyz_mm, cells)
    domain_tag = "" if domain_length_mm is None else "_d" + f"{domain_length_mm:.9f}".replace(".", "p") + "_centered"
    cache_path = cache_dir / f"{cad}_r{resolution}_quadratic{domain_tag}.npz"
    hit = False
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cache = {key: archive[key] for key in archive.files if key != "metadata_json"}
            geometry_report = json.loads(str(archive["metadata_json"].item()))
        if geometry_report["geometry_sha256"] != fingerprint or int(cache["resolution"]) != resolution:
            raise ValueError(f"Stale geometry cache: {cache_path}")
        hit = True
    else:
        cache, geometry_report = build_geometry_grid(xyz_mm, cells, resolution, device, pair_batch_size,
                                                    domain_length_mm=domain_length_mm)
        geometry_report["geometry_sha256"] = fingerprint
        _atomic_npz(cache_path, {**cache, "metadata_json": np.asarray(json.dumps(geometry_report))})
    flat = cache["occupied_flat_index"]
    occupied_targets = interpolate_nodal(values, cells, cache, device)
    feature_columns = np.r_[np.arange(16), 24]
    occupied_features = interpolate_nodal(features[:, feature_columns], cells, cache, device)
    occupied_features[:, :7] = np.clip(occupied_features[:, :7], 0, 1)
    inputs = np.zeros((18, resolution ** 3), dtype=np.float32)
    inputs[0, flat], inputs[1:, flat] = 1, occupied_features.T
    targets = np.zeros((values.shape[1], resolution ** 3), dtype=np.float32)
    targets[:, flat] = occupied_targets.T
    mask = np.zeros(resolution ** 3, dtype=bool)
    mask[flat] = True
    mask = mask.reshape((resolution,) * 3)
    domain_lower = cache.get("domain_lower_mm", cache["bbox_min_mm"])
    domain_length = float(cache.get("domain_length_mm", cache["characteristic_length_mm"]))
    node_xyz = ((xyz_mm.astype(np.float64) - domain_lower) / domain_length).astype(np.float32)
    query_indices, query_weights, query_coverage = masked_trilinear_cache(node_xyz, mask)
    tensor_grid = torch.as_tensor(targets.reshape(values.shape[1], *((resolution,) * 3)), device=device)
    roundtrip = query_grid(tensor_grid, torch.as_tensor(query_indices, device=device),
                           torch.as_tensor(query_weights, device=device)).cpu().numpy()
    errors = np.linalg.norm(roundtrip.astype(np.float64) - values.astype(np.float64), axis=0) / np.maximum(np.linalg.norm(values.astype(np.float64), axis=0), 1e-12)
    metadata = {
        "sample_key": sample_key, "dataset": "DeepJEB original v1.0", "label_alignment": ALIGNMENT,
        "source_prepared": str(source.resolve()), "source_H5_geometry": str((raw / "FieldMesh" / f"{cad}.h5").resolve()),
        "geometry_cache": str(cache_path.resolve()), "geometry_cache_hit": hit, "geometry": geometry_report,
        "grid_domain": {"lower_mm": domain_lower.tolist(), "length_mm": domain_length,
                        "spacing_mm": domain_length / (resolution - 1), "resolution": resolution,
                        "normalization": "Both occupied_xyz and node_xyz use (xyz_mm-lower_mm)/length_mm"},
        "schema": GRID_NPZ_SCHEMA, "feature_names": FEATURE_NAMES,
        "input_schema_note": "Explicit replacement; original 18-channel definitions unavailable. No xyz channels. support/load indicators are quadratic nodal interpolation clipped to [0,1]; labels are never clipped.",
        "target_scaling": "Physical labels unchanged; per-case occupied-grid population mean/std stored as statistics. Common trainer must fit/record its selected training-case scaler.",
        "field_names": field_names, "field_units": field_units,
        "query": "Masked trilinear weights normalized over occupied corners. No occupied support returns zero, included in all-node metrics; no nearest fallback.",
        "query_covered_nodes": int(query_coverage.sum()), "query_uncovered_nodes": int((~query_coverage).sum()),
        "source_nodes": len(node_xyz), "truth_roundtrip_relative_l2": dict(zip(field_names, errors.tolist())),
        "truth_roundtrip_note": "Observed representation/interpolation loss, not a mathematical lower bound on model error.",
        "FNO_sources": FNO_SOURCES, "preparation_seconds_before_serialization": time.perf_counter() - started,
    }
    if fields == "joint":
        metadata["truth_roundtrip_displacement_vector_relative_l2"] = float(np.linalg.norm(roundtrip[:, :3].astype(np.float64) - values[:, :3]) / max(np.linalg.norm(values[:, :3].astype(np.float64)), 1e-12))
    payload = {
        **cache, "inputs": inputs.reshape(18, *((resolution,) * 3)),
        "targets": targets.reshape(values.shape[1], *((resolution,) * 3)), "mask": mask,
        "node_xyz": node_xyz, "node_xyz_mm": xyz_mm, "node_targets": values,
        "grid_node_ids": grid_node_ids, "query_indices": query_indices, "query_weights": query_weights,
        "query_coverage": query_coverage, "target_mean": occupied_targets.astype(np.float64).mean(0),
        "target_std": np.maximum(occupied_targets.astype(np.float64).std(0), 1e-8),
        "field_names": np.asarray(field_names), "field_units": np.asarray(field_units),
        "feature_names": np.asarray(FEATURE_NAMES), "sample_key": np.asarray(sample_key),
        "label_alignment": np.asarray(ALIGNMENT), "metadata_json": np.asarray(json.dumps(metadata)),
    }
    _atomic_npz(target, payload)
    print(json.dumps({"output": str(target.resolve()), **metadata}), flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/prepared_multifield")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/grids")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/grid_cache")
    parser.add_argument("--cases", nargs="+", default=["10_30/ver"])
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--fields", choices=("joint", "stress"), default="joint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pair-batch-size", type=int, default=131072)
    domain = parser.add_mutually_exclusive_group()
    domain.add_argument("--domain-length-mm", type=float,
                        help="Cube length; centers the padded cube on the source bbox (explicit missing-origin assumption)")
    domain.add_argument("--spacing-mm", type=float,
                        help="Cube length=(R-1)*spacing; use 0.954956 for the slide's 192^3 domain")
    args = parser.parse_args()
    domain_length = args.domain_length_mm
    if args.spacing_mm is not None:
        domain_length = (args.resolution - 1) * args.spacing_mm
    for sample_key in args.cases:
        prepare_case(args.prepared, args.raw, args.output, args.cache_dir, sample_key,
                     args.resolution, args.fields, args.device, args.pair_batch_size, domain_length)


if __name__ == "__main__":
    main()
