"""Exact stored-coordinate radius pairs for a uniform unit-cube lattice.

Imports are lazy. This module performs no model construction, label access,
kernel weighting, FEM integration, or continuous-query acceptance test.
"""
from __future__ import annotations

import math
import time


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_regular_grid_radius(points, *, resolution=64, radius=.033, chunk_points=8192, progress=None):
    """Return both CSR orientations of the same untruncated radius pairs.

    grid_queries has R**3 rows and neighbors that index original points.
    point_queries has N rows and neighbors that index the x-major lattice.
    Input/lattice coordinates are stored float32; the final inclusive squared
    distance predicate operates on those values promoted to float64. Distances
    in both CSR arrays retain that float64 squared-distance representation.
    """
    import torch
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or points.dtype != torch.float32 or points.requires_grad):
        raise ValueError("Expected nonempty target-independent float32 [N,3], without gradients")
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 2:
        raise ValueError("Resolution must be integer >=2")
    if not math.isfinite(radius) or radius <= 0 or not isinstance(chunk_points, int) or chunk_points <= 0:
        raise ValueError("Expected a positive finite radius and chunk size")
    if not bool(torch.isfinite(points).all()) or float(points.min()) < -2e-6 or float(points.max()) > 1 + 2e-6:
        raise ValueError("Stored normalized points leave the unit-cube rounding tolerance; points are never clipped")
    device, count, grid_count = points.device, len(points), resolution**3
    radius_squared = radius * radius
    if count * grid_count >= 2**63:
        raise ValueError("Pair-key integer range exceeds int64")
    _sync(torch, device)
    began = time.monotonic()
    axis = torch.linspace(0., 1., resolution, dtype=torch.float32, device=device)
    axis64 = axis.to(torch.float64)
    ideal_axis = torch.arange(resolution, dtype=torch.float64, device=device) / (resolution - 1)
    lattice_roundoff = float((axis64 - ideal_axis).abs().max())
    # Includes the measured stored-lattice roundoff and a conservative FP64
    # arithmetic pad. This widens enumeration only, never the radius predicate.
    arithmetic_pad = 32 * torch.finfo(torch.float64).eps * (1 + radius)
    bound_pad = lattice_roundoff + arithmetic_pad
    box_side = math.ceil(2 * (radius + bound_pad) * (resolution - 1)) + 1
    offset_axis = torch.arange(box_side, dtype=torch.int64, device=device)
    offsets = torch.stack(torch.meshgrid(offset_axis, offset_axis, offset_axis, indexing="ij"), dim=-1).reshape(-1, 3)
    grid_xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    point_parts, grid_parts, distance_parts = [], [], []
    candidate_slots = valid_box_candidates = accepted_pairs = 0
    with torch.no_grad():
        for chunk_index, start in enumerate(range(0, count, chunk_points)):
            stop = min(start + chunk_points, count)
            point64 = points[start:stop].to(torch.float64)
            low = torch.ceil((point64 - radius - bound_pad) * (resolution - 1)).to(torch.int64).clamp(min=0)
            high = torch.floor((point64 + radius + bound_pad) * (resolution - 1)).to(torch.int64).clamp(max=resolution-1)
            width = high - low + 1
            if bool((width > box_side).any()):
                raise RuntimeError("Conservative enumeration box is unexpectedly too small")
            integer_xyz = low[:, None, :] + offsets[None, :, :]
            valid = (integer_xyz <= high[:, None, :]).all(-1)
            valid &= (integer_xyz >= 0).all(-1) & (integer_xyz < resolution).all(-1)
            rows, slots = torch.nonzero(valid, as_tuple=True)
            selected_xyz = integer_xyz[rows, slots]
            differences = axis64[selected_xyz] - point64[rows]
            squared_distances = (differences * differences).sum(-1)
            inside = squared_distances <= radius_squared
            selected_xyz = selected_xyz[inside]
            original_points = rows[inside] + start
            selected_grid = (selected_xyz[:, 0] * resolution**2
                             + selected_xyz[:, 1] * resolution + selected_xyz[:, 2])
            selected_distance = squared_distances[inside]
            point_parts.append(original_points)
            grid_parts.append(selected_grid)
            distance_parts.append(selected_distance)
            candidate_slots += (stop-start) * len(offsets)
            valid_box_candidates += len(rows)
            accepted_pairs += len(original_points)
            del point64, low, high, width, integer_xyz, valid, rows, slots, selected_xyz, differences, squared_distances, inside
            if progress is not None:
                _sync(torch, device)
                progress({"phase": "enumerating", "chunks_completed": chunk_index+1,
                          "points_processed": stop, "total_points": count,
                          "candidate_slots": candidate_slots, "valid_box_candidates": valid_box_candidates,
                          "accepted_pairs": accepted_pairs, "enumeration_elapsed_seconds": time.monotonic()-began})
        _sync(torch, device)
        enumeration_seconds = time.monotonic() - began
        stage = time.monotonic()
        point_ids = torch.cat(point_parts)
        grid_ids = torch.cat(grid_parts)
        distances = torch.cat(distance_parts)
        point_parts.clear(); grid_parts.clear(); distance_parts.clear()
        del original_points, selected_grid, selected_distance
        point_degree = torch.bincount(point_ids, minlength=count)
        point_row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), point_degree.cumsum(0)))
        _sync(torch, device)
        concatenate_seconds = time.monotonic() - stage
        if progress is not None:
            progress({"phase": "sorting_grid_pairs", "points_processed": count, "accepted_pairs": len(grid_ids)})
        stage = time.monotonic()
        grid_keys = grid_ids * count + point_ids
        permutation = torch.argsort(grid_keys)
        sorted_grid_keys = grid_keys[permutation]
        grid_degree = torch.bincount(grid_ids, minlength=grid_count)
        grid_row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), grid_degree.cumsum(0)))
        grid_neighbors = point_ids[permutation]
        grid_distances = distances[permutation]
        _sync(torch, device)
        sort_seconds = time.monotonic() - stage
        stage = time.monotonic()
        point_keys = point_ids * grid_count + grid_ids
        edges = len(grid_ids)
        checks = {
            "point_pair_keys_strictly_increasing_no_duplicates": bool((point_keys[1:] > point_keys[:-1]).all()),
            "grid_pair_keys_strictly_increasing_no_duplicates": bool((sorted_grid_keys[1:] > sorted_grid_keys[:-1]).all()),
            "point_row_ptr_valid": len(point_row_ptr) == count+1 and int(point_row_ptr[0]) == 0 and int(point_row_ptr[-1]) == edges and bool((point_row_ptr[1:] >= point_row_ptr[:-1]).all()),
            "grid_row_ptr_valid": len(grid_row_ptr) == grid_count+1 and int(grid_row_ptr[0]) == 0 and int(grid_row_ptr[-1]) == edges and bool((grid_row_ptr[1:] >= grid_row_ptr[:-1]).all()),
            "point_neighbor_bounds_valid": bool(((grid_ids >= 0) & (grid_ids < grid_count)).all()),
            "grid_neighbor_bounds_valid": bool(((grid_neighbors >= 0) & (grid_neighbors < count)).all()),
            "finite_inclusive_radius_distances": bool(torch.isfinite(distances).all() & (distances >= 0).all() & (distances <= radius_squared).all()),
            "point_degrees_reproduce_row_ptr": bool(torch.equal(point_row_ptr[1:] - point_row_ptr[:-1], point_degree)),
            "grid_degrees_reproduce_row_ptr": bool(torch.equal(grid_row_ptr[1:] - grid_row_ptr[:-1], grid_degree)),
            "orientations_same_edges_single_permutation": bool(torch.equal(sorted_grid_keys, grid_ids[permutation] * count + grid_neighbors) and torch.equal(grid_distances, distances[permutation])),
        }
        if not all(checks.values()):
            raise RuntimeError("Radius CSR invariant failed: " + repr(checks))
        _sync(torch, device)
        validation_seconds = time.monotonic() - stage
    return {"grid_queries": {"row_ptr": grid_row_ptr, "neighbors_index": grid_neighbors, "squared_distances": grid_distances},
            "point_queries": {"row_ptr": point_row_ptr, "neighbors_index": grid_ids, "squared_distances": distances},
            "grid_axis": axis, "grid_xyz": grid_xyz, "checks": checks,
            "construction": {"N": count, "R": resolution, "grid_rows": grid_count, "edges_per_orientation": edges,
                             "radius": radius, "radius_squared": radius_squared, "chunk_points": chunk_points,
                             "box_side": box_side, "candidate_slots": candidate_slots,
                             "valid_box_candidates": valid_box_candidates,
                             "stored_lattice_max_roundoff_vs_i_over_Rminus1": lattice_roundoff,
                             "enumeration_only_arithmetic_pad": arithmetic_pad,
                             "enumeration_bound_pad": bound_pad,
                             "grid_index_order": "x*R^2+y*R+z; z fastest",
                             "point_order": "unchanged canonical xyz.npy row order",
                             "coordinate_dtype": "float32 stored; promoted to float64 for distance",
                             "distance_dtype": "float64 squared normalized-length distance",
                             "membership": "squared distance of stored float32 values, computed in float64, <= float64(radius)**2; no predicate tolerance",
                             "enumeration": "Conservative integer-box candidate bounds widened for measured lattice rounding and FP64 arithmetic; no neighbor truncation, clipping, sampling, or dense cdist"},
            "costs": {"enumeration_seconds": enumeration_seconds, "concatenate_seconds": concatenate_seconds,
                      "grid_sort_seconds": sort_seconds, "invariant_validation_seconds": validation_seconds}}


def degree_statistics(row_ptr):
    """Actual query-degree distributions; empty rows are retained, never filled."""
    import torch
    degree = row_ptr[1:] - row_ptr[:-1]
    quantiles = (0., .01, .05, .5, .9, .95, .99, 1.)
    values = torch.quantile(degree.to(torch.float64), torch.tensor(quantiles, device=degree.device, dtype=torch.float64)).tolist()
    bins = [(0, 0), (1, 1)] + [(2**power, 2**(power+1)-1) for power in range(1, 17)]
    histogram = [{"min_inclusive": low, "max_inclusive": high, "rows": int(((degree >= low) & (degree <= high)).sum())} for low, high in bins]
    histogram.append({"min_inclusive": 2**17, "max_inclusive": None, "rows": int((degree >= 2**17).sum())})
    empty = torch.nonzero(degree == 0, as_tuple=False).flatten()
    maximum = int(degree.max())
    maximum_rows = torch.nonzero(degree == maximum, as_tuple=False).flatten()
    return {"queries": len(degree), "edges": int(degree.sum()), "mean_degree": float(degree.to(torch.float64).mean()),
            "quantiles": {str(q): value for q, value in zip(quantiles, values)}, "histogram": histogram,
            "empty_rows": int(len(empty)), "empty_fraction": len(empty)/len(degree),
            "query_coverage": 1-len(empty)/len(degree),
            "first_empty_row_indices": empty[:8].tolist(),
            "maximum_degree": maximum, "first_maximum_degree_row_indices": maximum_rows[:8].tolist(),
            "all_degrees_available": "row_ptr[1:] - row_ptr[:-1]; no sparse-row omission"}
