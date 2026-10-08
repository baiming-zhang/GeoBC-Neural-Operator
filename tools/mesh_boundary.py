"""Recover original quadratic tetrahedral surface faces in FEM node ordering.

Some original DeepJEB H5 files store ``faces`` in a separate surface-local
numbering without the surface vertex mapping. Those indices must not be
used against the full FEM ``vertices`` array. This helper extracts boundary
facets from the supplied TETRA10 cells and subdivides each original six-node
facet into its four supplied quadratic subtriangles. It adds no new nodes
and does not resample geometry or stress.
"""

from __future__ import annotations

import numpy as np


# VTK_QUADRATIC_TETRA: corners 0..3 followed by midpoints of these edges.
TETRA10_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
# Corner triangle a,b,c followed by midpoint ab,bc,ca, in FEM global IDs.
TETRA10_FACETS = np.array([[0, 1, 2, 4, 5, 6],
                           [0, 1, 3, 4, 8, 7],
                           [0, 2, 3, 6, 9, 7],
                           [1, 2, 3, 5, 9, 8]])
QUADRATIC_TRI_SPLIT = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [4, 5, 3]])


def _cells_array(cells: np.ndarray, xyz_mm: np.ndarray | None = None) -> np.ndarray:
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != 10 or not len(cells) or cells.dtype.kind not in "iu":
        raise ValueError("Expected nonempty integer TETRA10 cells [T,10].")
    if cells.min() < 0:
        raise ValueError("TETRA10 connectivity must use zero-based, nonnegative FEM node IDs.")
    if xyz_mm is not None:
        xyz_mm = np.asarray(xyz_mm)
        if xyz_mm.ndim != 2 or xyz_mm.shape[1] != 3 or cells.max() >= len(xyz_mm):
            raise ValueError("TETRA10 connectivity does not match xyz_mm [N,3].")
    return cells


def midpoint_order_report(cells: np.ndarray, xyz_mm: np.ndarray) -> dict:
    """Check edge ordering geometrically, allowing curved boundary midsides."""
    cells = _cells_array(cells, xyz_mm)
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    corners = xyz[cells[:, :4]]
    starts, ends = corners[:, TETRA10_EDGES[:, 0]], corners[:, TETRA10_EDGES[:, 1]]
    edge_length = np.linalg.norm(ends - starts, axis=2)
    deviation = np.linalg.norm(xyz[cells[:, 4:]] - (starts + ends) * 0.5, axis=2)
    ratio = deviation / np.maximum(edge_length, 1e-12)
    return {"tetra10_edge_order": TETRA10_EDGES.tolist(),
            "midside_midpoint_deviation_mm_p50": float(np.quantile(deviation, 0.5)),
            "midside_midpoint_deviation_mm_p95": float(np.quantile(deviation, 0.95)),
            "midside_midpoint_deviation_mm_max": float(deviation.max()),
            "midside_deviation_over_edge_p50": float(np.quantile(ratio, 0.5)),
            "midside_deviation_over_edge_p95": float(np.quantile(ratio, 0.95)),
            "note": "Curved quadratic surfaces may have nonzero midpoint deviation; no nodes are moved."}


def quadratic_boundary_triangles(cells: np.ndarray, xyz_mm: np.ndarray | None = None) -> np.ndarray:
    """Return global FEM-node triangles for every original volume boundary facet."""
    cells = _cells_array(cells, xyz_mm)
    if xyz_mm is not None:
        order = midpoint_order_report(cells, xyz_mm)
        if order["midside_deviation_over_edge_p50"] > 0.05 or order["midside_deviation_over_edge_p95"] > 0.25:
            raise ValueError("Cell midside nodes do not follow the VTK TETRA10 edge order.")
    facets = cells[:, TETRA10_FACETS].reshape(-1, 6)
    # Interior facets occur exactly twice; boundary corner triples occur once.
    keys = np.sort(facets[:, :3], axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold tetrahedral connectivity: a corner facet occurs more than twice.")
    boundary = facets[first[counts == 1]]
    if not len(boundary):
        raise ValueError("No volume boundary facets found.")
    triangles = boundary[:, QUADRATIC_TRI_SPLIT].reshape(-1, 3)
    if triangles.max() > np.iinfo(np.int32).max:
        raise ValueError("FEM node IDs exceed int32 surface connectivity capacity.")
    return triangles.astype(np.int32, copy=False)


def boundary_report(xyz_mm: np.ndarray, faces: np.ndarray, cells: np.ndarray | None = None) -> dict:
    """Small physical-scale report to catch surface/FEM numbering mismatches."""
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(xyz):
        raise ValueError("Invalid global FEM surface triangle indices.")
    vertices = xyz[faces]
    edges = np.linalg.norm(vertices - vertices[:, [1, 2, 0]], axis=2).reshape(-1)
    quantiles = np.quantile(edges, [0.5, 0.9, 0.99, 1.0])
    report = {"surface_triangles": len(faces), "quadratic_boundary_facets": len(faces) // 4,
              "surface_FEM_nodes": len(np.unique(faces)),
              "triangle_edge_mm_p50": float(quantiles[0]), "triangle_edge_mm_p90": float(quantiles[1]),
              "triangle_edge_mm_p99": float(quantiles[2]), "triangle_edge_mm_max": float(quantiles[3]),
              "surface_connectivity": "Boundary of original TETRA10 volume; original midside nodes retained; four triangles per six-node facet."}
    if cells is not None:
        report.update(midpoint_order_report(cells, xyz))
    return report


__all__ = ["quadratic_boundary_triangles", "boundary_report", "midpoint_order_report"]
