"""Bounded, target-free connected hierarchy preparation for four fixed TRAIN CADs.

The contract path imports only the standard library. Actual preparation reads
only the predeclared named input/geometry NPZ members, uses CPU SciPy, and never
loads a gradient-cache helper, target, raw H5/deck, model, or CUDA runtime.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _thread in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread] = "1"
ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
CAD_ORDER = ["101_428", "101_479", "101_56", "101_78"]
PLAN_SHA = "a1748a2e9bd02a50d48a57070eaffee6e20325ac51e5f2167e19ce46200aab7d"
EDGE_ORDER = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
ALIGNMENT = "official CSV GRID coordinates -> H5 vertices"


def require(value, message):
    if not value:
        raise ValueError(message)


class ScientificBlock(Exception):
    """A reached predeclared scientific/resource bound, not corrupt input."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and value.startswith("/mnt/"):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def portable(value):
    value = path(value)
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return str(value)


def read_json(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def digest_file(value):
    result = hashlib.sha256()
    with path(value).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(value, sha=False):
    value = path(value)
    stat = value.stat()
    return {"path": portable(value), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest_file(value) if sha else None}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(value, data):
    value = path(value)
    temporary = value.with_suffix(value.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(value)


def match_stat(value, recorded):
    actual = identity(value)
    require(actual["bytes"] == recorded["bytes"] and actual["mtime_ns"] == recorded["mtime_ns"],
            f"Frozen input stat changed: {portable(value)}")


def request_contract(plan_path=ROOT / "notes/geobc_topology_context_plan.json"):
    """Pure stdlib: small JSON/source SHA and selected NPZ stat, no arrays."""
    plan_id = identity(plan_path, sha=True)
    require(plan_id["sha256"] == PLAN_SHA, "The formally frozen topology input plan changed")
    plan = read_json(plan_path)
    scope, data = plan["scope"], plan["data"]
    require(plan["status"] == "predeclared" and scope["CAD_order"] == CAD_ORDER
            and scope["group"] == "train" and scope["canonical_load"] == "ver"
            and scope["workers"] == scope["CPU_numeric_threads"] == 1,
            "Only the fixed serial four-TRAIN input scope is permitted")
    require(data["prepared_members"] == ["xyz", "features", "sample_key", "label_alignment", "bbox_min_mm", "characteristic_length_mm"]
            and data["gradient_members"] == ["cells10", "volumes_mm3"]
            and data["interface_members"] == ["euclidean", "cad_id", "xyz_sha256"], "Input-member recipe differs")
    split_id = identity(scope["split"], sha=True)
    require(split_id["sha256"] == scope["split_sha256"], "Frozen split changed")
    split = read_json(scope["split"])
    require(all(cad in split["groups"]["train"] for cad in CAD_ORDER)
            and not set(CAD_ORDER).intersection(split["groups"]["validation"] + split["groups"]["test"]),
            "Selected CADs must remain exclusively TRAIN")
    grad_path, interface_path = (path(value) for value in data["manifests"])
    grad, interfaces = read_json(grad_path), read_json(interface_path)
    selected = {}
    for cad in CAD_ORDER:
        g, e = grad["CADs"][cad], interfaces["CADs"][cad]
        normalization = {key: g["normalization"][key] for key in ("bbox_min_mm", "characteristic_length_mm")}
        require(g["role"] == "train_loss_and_diagnostic" and g["node_count"] == e["node_count"]
                and g["xyz_sha256"] == e["xyz_sha256"] and normalization == e["normalization"],
                "Selected geometry/input metadata parity differs")
        groups = e["source_groups"]
        require([q["channel"] for q in groups] == list(range(5))
                and [q["role"] for q in groups] == ["support"] * 4 + ["load"]
                and [q["master_node_id"] for q in groups[:4]] == sorted(q["master_node_id"] for q in groups[:4])
                and all(q["card_type"] == "RBE2" for q in groups[:4])
                and groups[4]["card_type"] in ("RBE2", "RBE3"), "Original five-interface identity/order differs")
        prepared = path(split["case_paths"][f"{cad}/ver"])
        require(prepared.stem == f"{cad}_ver" and prepared.parent == path(data["prepared"]), "Only canonical prepared ver allowed")
        gradient = path(data["gradient_geometry"]) / g["file"]
        interface = path(data["interface_inputs"]) / e["file"]
        match_stat(gradient, g)
        match_stat(interface, e)
        selected[cad] = {"prepared": identity(prepared), "gradient_geometry": identity(gradient),
            "interface_inputs": identity(interface), "node_count": g["node_count"], "cell_count": g["cell_count"],
            "xyz_sha256": g["xyz_sha256"], "normalization": normalization, "sum_volume_mm3": g["sum_volume_mm3"],
            "cells_sha256": e["cells_sha256"], "edges_sha256": e["graph_topology"]["edges_sha256"],
            "source_groups": [{k: q[k] for k in ("channel", "role", "master_node_id", "card_type", "member_count",
                "member_GRID_IDs_sha256", "member_node_indices_sha256")} for q in groups]}
    return {"schema_version": 1, "plan": plan_id, "split": split_id, "CAD_order": CAD_ORDER,
        "manifests": [identity(grad_path, sha=True), identity(interface_path, sha=True)],
        "source_sha256": {"tools/prepare_topology_context.py": identity(__file__, sha=True)["sha256"]},
        "inputs": selected, "cache": plan["output"]["cache"], "recipe": {"soft_targets": [2048, 128],
            "hard_maxima": [4096, 512], "omega": 2 / 3, "passes": 2, "P_average_nnz_limit": 64,
            "cache_bytes_limit": 1073741824, "restriction": "hard mass", "W0": "binary",
            "coarse_W": "actual cut multiplicities", "CSR_storage": "int64 indices; float32 P/R; float64 W"},
        "named_member_access": {"prepared": data["prepared_members"], "gradient": data["gradient_members"],
            "interface": data["interface_members"]}, "labels_read": 0, "raw_sources_opened": False,
        "CPU_only": True, "CUDA_initialized": False, "updates": 0}


def array_sha(array, dtype):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def peak_rss():
    try:
        import resource
        return {"bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
                "method": "process lifetime ru_maxrss"}
    except ImportError:
        return {"bytes": None, "method": "resource.getrusage unavailable on this platform"}


def clean_csr(matrix):
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()  # Exact zeros only; never threshold a positive path.
    matrix.sort_indices()
    return matrix


def graph_from_cells(xyz, cells):
    import numpy as np
    from scipy.sparse import csr_matrix
    n = len(xyz)
    require(cells.ndim == 2 and cells.shape[1] == 10 and np.issubdtype(cells.dtype, np.integer)
            and len(cells) and cells.min() >= 0 and cells.max() < n
            and (np.diff(np.sort(cells, axis=1), axis=1) != 0).all(), "Invalid or duplicate TETRA10 node IDs")
    mids = cells[:, 4:]
    pieces = [np.stack((cells[:, a], mids[:, k]), axis=1) for k, (a, b) in enumerate(EDGE_ORDER)]
    pieces += [np.stack((mids[:, k], cells[:, b]), axis=1) for k, (a, b) in enumerate(EDGE_ORDER)]
    edges = np.unique(np.sort(np.concatenate(pieces), axis=1), axis=0)
    vectors = xyz[edges[:, 1]].astype(np.float64) - xyz[edges[:, 0]].astype(np.float64)
    lengths = np.linalg.norm(vectors, axis=1)
    require(np.isfinite(lengths).all() and (lengths > 0).all(), "Nonpositive original subedge length")
    src, dst = edges[:, 0], edges[:, 1]
    W = clean_csr(csr_matrix((np.ones(2 * len(edges)), (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)))
    require(W.nnz == 2 * len(edges) and (W.data == 1).all(), "W0 must be binary after unique edge construction")
    return W, edges, vectors, lengths


def role_components(W, roles):
    import numpy as np
    from scipy.sparse.csgraph import connected_components
    counts = {}
    for role in np.unique(roles):
        ids = np.flatnonzero(roles == role)
        counts[str(int(role))] = int(connected_components(W[ids][:, ids], directed=False, return_labels=False))
    return {"by_exact_signature": counts, "lower_bound": sum(counts.values())}


def connected_owners(W, roles, descendant_min, soft_target, hard_maximum, level):
    import numpy as np
    lower = role_components(W, roles)
    if lower["lower_bound"] > hard_maximum:
        raise ScientificBlock("Exact-role component lower bound exceeds aggregate cap",
            {"level": level, "role_components": lower, "hard_maximum": hard_maximum})
    n = W.shape[0]
    growth = max(1, math.ceil(n / soft_target))
    owner = np.full(n, -1, np.int64)
    parent = np.full(n, -1, np.int64)
    admission = np.full(n, -1, np.int64)
    seeds = []
    for seed in np.argsort(descendant_min, kind="stable"):
        if owner[seed] >= 0:
            continue
        group = len(seeds)
        seeds.append(int(seed))
        owner[seed], parent[seed], admission[seed] = group, seed, 0
        queue, admitted = deque([int(seed)]), 1
        while queue and admitted < growth:
            current = queue.popleft()
            for neighbor in W.indices[W.indptr[current]:W.indptr[current + 1]]:
                if owner[neighbor] < 0 and roles[neighbor] == roles[seed]:
                    owner[neighbor], parent[neighbor], admission[neighbor] = group, current, admitted
                    admitted += 1
                    queue.append(int(neighbor))
                    if admitted == growth:
                        break
        if len(seeds) > hard_maximum:
            raise ScientificBlock("Actual bounded BFS aggregate count exceeds cap", {"level": level,
                "observed_aggregate_count": len(seeds), "assigned_vertices": int((owner >= 0).sum()),
                "vertices": n, "growth_size": growth, "role_components": lower, "hard_maximum": hard_maximum})
    seeds = np.asarray(seeds, np.int64)
    require((owner >= 0).all() and (roles == roles[seeds[owner]]).all(), "Ownership coverage/homogeneity failed")
    mins = np.full(len(seeds), np.iinfo(np.int64).max, np.int64)
    np.minimum.at(mins, owner, descendant_min)
    order = np.argsort(mins, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    owner, seeds, mins = inverse[owner], seeds[order], mins[order]
    require((owner[parent] == owner).all(), "BFS parent belongs to another aggregate")
    nonseed = np.flatnonzero(parent != np.arange(n))
    require((admission[parent[nonseed]] < admission[nonseed]).all()
            and (roles[parent] == roles).all(), "BFS parent is not an earlier same-role member")
    # An actual stored edge to an earlier admitted parent proves connectivity.
    for vertex in nonseed:
        neighbors = W.indices[W.indptr[vertex]:W.indptr[vertex + 1]]
        at = np.searchsorted(neighbors, parent[vertex])
        require(at < len(neighbors) and neighbors[at] == parent[vertex], "BFS parent is not an actual graph edge")
    sizes = np.bincount(owner, minlength=len(seeds))
    facts = {"vertices": n, "aggregate_count": len(seeds), "growth_size": growth,
        "soft_target": soft_target, "hard_maximum": hard_maximum, "role_components": lower,
        "size_min": int(sizes.min()), "size_max": int(sizes.max()), "size_mean": float(sizes.mean()),
        "coverage_complete": True, "disjoint": True, "homogeneous": True,
        "parent_connectivity_witness": True, "renumbered_by_min_original_descendant": True}
    return owner, parent, admission, roles[seeds], mins, facts


def aggregate(W, owner, mass, xyz, named_counts, raw_counts):
    import numpy as np
    from scipy.sparse import csr_matrix
    count = int(owner.max()) + 1
    G = csr_matrix((np.ones(len(owner)), (np.arange(len(owner)), owner)), shape=(len(owner), count))
    coarse = clean_csr(G.T @ W @ G)
    coarse.setdiag(0)
    coarse = clean_csr(coarse)
    masses = np.bincount(owner, weights=mass, minlength=count)
    require((masses > 0).all() and np.isfinite(masses).all(), "Coarse mass not positive")
    centers = np.stack([np.bincount(owner, weights=mass * xyz[:, j], minlength=count) / masses for j in range(3)], axis=1)
    named = np.zeros((count, 5), np.int64)
    raw = np.zeros((count, 7), np.int64)
    np.add.at(named, owner, named_counts)
    np.add.at(raw, owner, raw_counts)
    R = clean_csr(csr_matrix((mass / masses[owner], (owner, np.arange(len(owner)))), shape=(count, len(owner))))
    return coarse, G, R, masses, centers, named, raw


def graph_facts(W):
    import numpy as np
    from scipy.sparse.csgraph import connected_components
    require(W.shape[0] == W.shape[1] and np.isfinite(W.data).all() and (W.data > 0).all()
            and (W.diagonal() == 0).all() and (W - W.T).nnz == 0, "Invalid quotient adjacency")
    components, ids = connected_components(W, directed=False)
    degree = np.asarray(W.sum(axis=1)).ravel()
    return {"nodes": W.shape[0], "directed_entries": W.nnz, "components": int(components),
        "component_sizes": np.bincount(ids).tolist(), "isolated_nodes": int((degree == 0).sum()),
        "symmetric": True, "positive_integer_cut_weights": bool((W.data == np.round(W.data)).all()),
        "diagonal_zero": True}, degree


def mapping_facts(matrix, fp64_tolerance=1e-12, fp32_tolerance=1e-6):
    import numpy as np
    require(np.isfinite(matrix.data).all() and (matrix.data > 0).all(), "Nonpositive/nonfinite sparse map")
    row_error64 = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)))
    saved = matrix.astype(np.float32)
    row_error32 = float(np.max(np.abs(np.asarray(saved.astype(np.float64).sum(axis=1)).ravel() - 1)))
    require(row_error64 <= fp64_tolerance and row_error32 <= fp32_tolerance, "Map constant/row-sum tolerance failed")
    support = np.diff(matrix.indptr)
    require((support > 0).all(), "Map has uncovered rows")
    facts = {"shape": list(matrix.shape), "nnz": matrix.nnz, "nnz_per_row_mean": float(matrix.nnz / matrix.shape[0]),
        "support_min": int(support.min()), "support_median": float(np.median(support)),
        "support_p95": float(np.quantile(support, .95)), "support_max": int(support.max()),
        "FP64_row_sum_max_abs_error": row_error64, "FP32_row_sum_max_abs_error": row_error32,
        "constant_reproduction_max_abs_error": row_error32, "finite_positive_complete_support": True}
    return saved, facts


def prolongation(W, G, roles, coarse_roles, level):
    import numpy as np
    from scipy.sparse import diags
    degree = np.asarray(W.sum(axis=1)).ravel()
    positive = degree > 0
    inverse = np.zeros_like(degree)
    inverse[positive] = 1 / degree[positive]
    J = clean_csr(diags(np.where(positive, 1 / 3, 1.0)) + (2 / 3) * diags(inverse) @ W)
    P = G.astype(np.float64)
    passes = []
    for step in range(2):
        P = clean_csr(J @ P)
        passes.append({"pass": step + 1, "nnz": P.nnz})
        if P.nnz > 64 * W.shape[0]:
            raise ScientificBlock("Complete prolongation exceeds average support cap", {"level": level,
                "pass": step + 1, "nnz": P.nnz, "vertices": W.shape[0], "average_nnz": P.nnz / W.shape[0], "limit": 64})
    saved, facts = mapping_facts(P)
    rows = np.repeat(np.arange(len(roles)), np.diff(P.indptr))
    crossed = roles[rows] != coarse_roles[P.indices]
    crossing_by_row = np.zeros(len(roles), np.float64)
    np.add.at(crossing_by_row, rows, P.data * crossed)
    facts.update(passes=passes, isolated_identity_rows=int((~positive).sum()),
        role_mixing_rows=int((crossing_by_row > 0).sum()), role_mixing_row_fraction=float((crossing_by_row > 0).mean()),
        role_cross_weight_mean=float(crossing_by_row.mean()), role_cross_weight_max=float(crossing_by_row.max()))
    return saved, facts


def pack_csr(arrays, name, matrix, dtype):
    import numpy as np
    arrays[name + "_indptr"] = np.asarray(matrix.indptr, np.int64)
    arrays[name + "_indices"] = np.asarray(matrix.indices, np.int64)
    arrays[name + "_data"] = np.asarray(matrix.data, dtype)
    arrays[name + "_shape"] = np.asarray(matrix.shape, np.int64)


def build_hierarchy(xyz, cells, volumes, role_bits, named, soft_targets=(2048, 128), hard_maxima=(4096, 512)):
    import numpy as np
    began, timings = time.monotonic(), {}
    W0, edges, vectors, lengths = graph_from_cells(xyz, cells)
    mass0 = np.zeros(len(xyz), np.float64)
    np.add.at(mass0, cells.ravel(), np.repeat(volumes / 10, 10))
    require(np.isfinite(mass0).all() and (mass0 > 0).all(), "Uncovered/zero-mass source node")
    require(math.isclose(float(mass0.sum()), float(volumes.sum()), rel_tol=1e-12, abs_tol=1e-12), "Source mass sum differs from cell volumes")
    timings["graph_and_mass_seconds"] = time.monotonic() - began
    roles0 = np.asarray(role_bits, np.uint8)
    raw0 = ((roles0[:, None] >> np.arange(7)) & 1).astype(np.int64)
    arrays = {"xyz0_normalized": np.asarray(xyz, np.float32), "role_bits0": roles0,
        "mass0_mm3": mass0, "named5_counts0": named.astype(np.int64), "role_member_counts0": raw0,
        "edge_endpoints": edges.astype(np.int64), "edge_vectors_normalized": vectors,
        "edge_lengths_normalized": lengths}
    source = {"W": W0, "mass": mass0, "xyz": xyz, "roles": roles0, "named": named, "raw": raw0,
        "descendant": np.arange(len(xyz), dtype=np.int64)}
    graph_records, bfs_records, maps = [], [], {}
    for level in range(2):
        tick = time.monotonic()
        graph_record, degree = graph_facts(source["W"])
        graph_records.append(graph_record)
        arrays[f"degree{level}"] = degree
        pack_csr(arrays, f"W{level}", source["W"], np.float64)
        owner, parent, admission, roles, descendant, bfs = connected_owners(source["W"], source["roles"],
            source["descendant"], soft_targets[level], hard_maxima[level], level)
        bfs_records.append(bfs)
        arrays[f"aggregate{level}{level + 1}"] = owner
        arrays[f"bfs_parent{level}"] = parent
        arrays[f"bfs_admission_rank{level}"] = admission
        arrays[f"descendant_min{level}"] = source["descendant"]
        W, G, R, mass, centers, named_counts, raw_counts = aggregate(source["W"], owner,
            source["mass"], source["xyz"], source["named"], source["raw"])
        require(np.array_equal(named_counts.sum(0), named.sum(0)) and np.array_equal(raw_counts.sum(0), raw0.sum(0))
                and math.isclose(float(mass.sum()), float(mass0.sum()), rel_tol=1e-12, abs_tol=1e-12), "Hard source/mass counts not conserved")
        saved_R, r_facts = mapping_facts(R)
        P, p_facts = prolongation(source["W"], G, source["roles"], roles, level)
        maps[f"P{level}{level + 1}"], maps[f"R{level}{level + 1}"] = p_facts, r_facts
        pack_csr(arrays, f"P{level}{level + 1}", P, np.float32)
        pack_csr(arrays, f"R{level}{level + 1}", saved_R, np.float32)
        arrays[f"mass{level + 1}_mm3"] = mass
        arrays[f"xyz{level + 1}_normalized"] = centers.astype(np.float32)
        arrays[f"role_bits{level + 1}"] = roles
        arrays[f"named5_counts{level + 1}"] = named_counts
        arrays[f"role_member_counts{level + 1}"] = raw_counts
        timings[f"level{level}_hierarchy_and_maps_seconds"] = time.monotonic() - tick
        source = {"W": W, "mass": mass, "xyz": centers, "roles": roles, "named": named_counts,
                  "raw": raw_counts, "descendant": descendant}
    last_graph, last_degree = graph_facts(source["W"])
    graph_records.append(last_graph)
    pack_csr(arrays, "W2", source["W"], np.float64)
    arrays["degree2"], arrays["descendant_min2"] = last_degree, source["descendant"]
    return arrays, {"N0": len(xyz), "N1": len(arrays["mass1_mm3"]), "N2": len(arrays["mass2_mm3"]),
        "graphs": graph_records, "BFS": bfs_records, "maps": maps, "timings": timings,
        "input_volume_proxy_sum_mm3": float(mass0.sum()), "named5_member_counts": named.sum(0).astype(int).tolist(),
        "mass_and_source_count_conservation": True, "labels_read": 0, "CUDA_initialized": False}


def read_inputs(cad, source):
    import numpy as np
    for key in ("prepared", "gradient_geometry", "interface_inputs"):
        match_stat(source[key]["path"], source[key])
    with np.load(path(source["prepared"]["path"]), allow_pickle=False) as stored:
        xyz, features = stored["xyz"], stored["features"]
        sample_key, alignment = str(stored["sample_key"].item()), str(stored["label_alignment"].item())
        lower, length = stored["bbox_min_mm"], float(stored["characteristic_length_mm"].item())
    require(xyz.dtype == np.float32 and xyz.shape == (source["node_count"], 3) and np.isfinite(xyz).all()
            and features.shape == (len(xyz), 25) and np.isfinite(features).all()
            and sample_key == f"{cad}/ver" and alignment == ALIGNMENT, "Canonical input/alignment guard failed")
    require(array_sha(xyz, "<f4") == source["xyz_sha256"]
            and np.array_equal(lower, np.asarray(source["normalization"]["bbox_min_mm"], np.float32))
            and length == source["normalization"]["characteristic_length_mm"]
            and math.isfinite(length) and length > 0, "XYZ order/normalization differs")
    flags = features[:, :7]
    require(((flags == 0) | (flags == 1)).all(), "Original flags are not exact binary values")
    roles = (flags.astype(np.uint8) * (2 ** np.arange(7, dtype=np.uint8))).sum(1).astype(np.uint8)
    del features
    # Do not call load_gradient_cache, list/read other NPZ members, or touch reference values.
    with np.load(path(source["gradient_geometry"]["path"]), allow_pickle=False) as stored:
        cells, volumes = stored["cells10"], stored["volumes_mm3"]
    require(cells.dtype == np.int64 and cells.shape == (source["cell_count"], 10)
            and volumes.dtype == np.float64 and volumes.shape == (len(cells),)
            and np.isfinite(volumes).all() and (volumes > 0).all()
            and array_sha(cells, "<i8") == source["cells_sha256"]
            and math.isclose(float(volumes.sum()), source["sum_volume_mm3"], rel_tol=1e-12, abs_tol=1e-12), "Invalid cached geometry/volumes")
    with np.load(path(source["interface_inputs"]["path"]), allow_pickle=False) as stored:
        distances, cache_cad = stored["euclidean"], str(stored["cad_id"].item())
        cache_xyz = str(stored["xyz_sha256"].item())
    require(cache_cad == cad and cache_xyz == source["xyz_sha256"] and distances.dtype == np.float32
            and distances.shape == (len(xyz), 5) and np.isfinite(distances).all() and (distances >= 0).all(), "Invalid named interface inputs")
    named = distances == 0.0
    require(named.sum(0).tolist() == [q["member_count"] for q in source["source_groups"]]
            and np.array_equal(named[:, :4].any(1), flags[:, :6].any(1))
            and np.array_equal(named[:, 4], flags[:, 6] == 1), "Exactzero named groups/counts/BC union mismatch")
    del distances
    return xyz, cells, volumes, roles, named.astype(np.int64)


def prepare_one(cad, source, cache, contract_sha, current_bytes):
    import numpy as np
    began = time.monotonic()
    xyz, cells, volumes, roles, named = read_inputs(cad, source)
    input_elapsed = time.monotonic() - began
    arrays, facts = build_hierarchy(xyz, cells, volumes, roles, named)
    require(array_sha(arrays["edge_endpoints"], "<i8") == source["edges_sha256"], "Original edge structural hash differs")
    facts["timings"]["input_reads_seconds"] = input_elapsed
    record = {"status": "completed", "cad_id": cad, "canonical_case": f"{cad}/ver", "contract_sha256": contract_sha,
        "source": source, "xyz_sha256": source["xyz_sha256"], "normalization": source["normalization"],
        "source_groups": source["source_groups"], "facts": facts, "plain_array_bytes": sum(x.nbytes for x in arrays.values()),
        "units": {"xyz_and_edges": "mm / characteristic_length_mm", "mass": "corner P1 cell mm3 / 10 per original incident node", "W": "binary or cut multiplicity", "P_R": "dimensionless"},
        "accessed_members": {"prepared": ["xyz", "features", "sample_key", "label_alignment", "bbox_min_mm", "characteristic_length_mm"],
            "gradient": ["cells10", "volumes_mm3"], "interface": ["euclidean", "cad_id", "xyz_sha256"]},
        "labels_read": 0, "raw_sources_opened": False, "CUDA_initialized": False, "target_independent": True}
    record["elapsed_seconds"] = time.monotonic() - began
    record["peak_RSS"] = peak_rss()
    if current_bytes + record["plain_array_bytes"] > 1073741824:
        raise ScientificBlock("Total cache size lower bound exceeds 1GiB", {"completed_bytes": current_bytes,
            "candidate_array_bytes": record["plain_array_bytes"], "limit": 1073741824, "facts": facts})
    arrays.update(cad_id=np.asarray(cad), xyz_sha256=np.asarray(source["xyz_sha256"]),
        contract_sha256=np.asarray(contract_sha), record_json=np.asarray(json.dumps(record, ensure_ascii=False, allow_nan=False)))
    target, temporary = cache / f"{cad}.npz", cache / f"{cad}.tmp.npz"
    require(not target.exists(), "Will not overwrite an existing completed CAD cache")
    tick = time.monotonic()
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    candidate_bytes = temporary.stat().st_size
    if current_bytes + candidate_bytes > 1073741824:
        temporary.unlink()  # Only this tool's uncommitted temporary file.
        raise ScientificBlock("Actual total cache bytes exceeds 1GiB", {"completed_bytes": current_bytes,
            "candidate_bytes": candidate_bytes, "limit": 1073741824, "facts": facts})
    for key in ("prepared", "gradient_geometry", "interface_inputs"):
        match_stat(source[key]["path"], source[key])
    temporary.replace(target)
    record.update(file=target.name, file_identity=identity(target, sha=True))
    record["facts"]["timings"]["cache_write_and_hash_seconds"] = time.monotonic() - tick
    record["elapsed_seconds"] = time.monotonic() - began
    return record


def recover_cache(cad, cache, contract_sha, source, saved=None):
    """Completed reuse verifies identity; orphan atomic files expose only record JSON."""
    target = cache / f"{cad}.npz"
    if saved is not None:
        require(saved["status"] == "completed" and saved["contract_sha256"] == contract_sha
                and saved["source"] == source, "Completed CAD contract differs")
        match_stat(target, saved["file_identity"])
        require(digest_file(target) == saved["file_identity"]["sha256"], "Completed CAD cache SHA changed")
        return saved
    if not target.exists():
        return None
    import numpy as np
    with np.load(target, allow_pickle=False) as stored:
        record = json.loads(str(stored["record_json"].item()))
    require(record["status"] == "completed" and record["contract_sha256"] == contract_sha
            and record["source"] == source and record["labels_read"] == 0 and record["target_independent"],
            "Orphan atomic CAD cache is not this completed contract")
    record.update(file=target.name, file_identity=identity(target, sha=True), atomic_cache_recovered=True)
    return record


def run(args):
    began = time.monotonic()
    contract = request_contract(args.plan)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "arrays_opened": False}
    output, cache = path(args.output), path(contract["cache"])
    require(output != cache, "Run artifacts and immutable caches must use distinct directories")
    config = {"contract": contract, "contract_sha256": fingerprint(contract),
        "scope": "four fixed TRAIN canonical ver inputs only", "optimizer_updates": 0}
    summary_path, config_path, manifest_path = output / "summary.json", output / "config.json", cache / "manifest.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Nonempty output config differs; no silent overwrite")
        if summary_path.exists() and read_json(summary_path).get("status") == "completed":
            old = read_json(summary_path)
            require(old["config"]["contract"] == contract, "Completed summary contract differs")
            for cad, record in old["CADs"].items():
                recover_cache(cad, cache, config["contract_sha256"], contract["inputs"][cad], record)
            return old
    else:
        require(not output.exists() or not any(output.iterdir()), "Nonempty run directory has no matching config")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        require(manifest["contract"] == contract, "Existing topology cache manifest contract differs")
    else:
        require(not any(p.name not in [f"{cad}.npz" for cad in CAD_ORDER] + [f"{cad}.tmp.npz" for cad in CAD_ORDER]
                        for p in cache.iterdir()), "Cache directory contains unrelated files")
        manifest = {"schema_version": 1, "status": "running", "contract": contract, "CADs": {}}
    completed, blocked, failed, untouched = {}, {}, {}, list(CAD_ORDER)
    status = {"status": "running", "config": config, "CAD_order": CAD_ORDER, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        for cad in CAD_ORDER:
            source = contract["inputs"][cad]
            record = recover_cache(cad, cache, config["contract_sha256"], source, manifest["CADs"].get(cad))
            current_bytes = sum(r["file_identity"]["bytes"] for r in completed.values())
            status.update(current_CAD=cad, completed_CADs=list(completed), elapsed_seconds=time.monotonic() - began)
            write_json(output / "status.json", status)
            if record is None:
                try:
                    record = prepare_one(cad, source, cache, config["contract_sha256"], current_bytes)
                except ScientificBlock as exc:
                    blocked[cad] = {"reason": str(exc), "evidence": exc.evidence, "labels_read": 0}
                    untouched.remove(cad)
                    break
            require(current_bytes + record["file_identity"]["bytes"] <= 1073741824, "Existing cache bundle exceeds frozen total bytes cap")
            completed[cad], manifest["CADs"][cad] = record, record
            untouched.remove(cad)
            manifest.update(status="running", cache_bytes=sum(r["file_identity"]["bytes"] for r in completed.values()))
            write_json(manifest_path, manifest)
        passed = len(completed) == len(CAD_ORDER)
        summary = {**status, "status": "completed", "outcome": "feasible" if passed else "infeasible",
            "feasibility_pass": passed, "CADs": completed, "completed_CADs": list(completed), "blocked_CADs": blocked,
            "untouched_CADs": untouched, "failed_CADs": {}, "cache_bytes": sum(r["file_identity"]["bytes"] for r in completed.values()),
            "elapsed_seconds": time.monotonic() - began, "peak_RSS": peak_rss(), "labels_read": 0,
            "CUDA_initialized": False, "no_automatic_training": True}
        manifest.update(status="completed", feasibility_pass=passed, outcome=summary["outcome"],
                        blocked_CADs=blocked, untouched_CADs=untouched)
        write_json(manifest_path, manifest)
        write_json(summary_path, summary)
        write_json(output / "status.json", summary)
        write_json(ROOT / "notes/geobc_topology_context_feasibility_result.json", summary)
        return summary
    except Exception as exc:
        current = status.get("current_CAD")
        if current:
            failed[current] = {"error": f"{type(exc).__name__}: {exc}"}
        failure = {**status, "status": "failed", "outcome": "invalid_input_or_execution_error",
            "error": f"{type(exc).__name__}: {exc}", "CADs": completed, "completed_CADs": list(completed),
            "blocked_CADs": blocked, "failed_CADs": failed, "untouched_CADs": untouched,
            "elapsed_seconds": time.monotonic() - began, "peak_RSS": peak_rss(), "labels_read": 0,
            "CUDA_initialized": False, "completed_files_preserved": True}
        write_json(summary_path, failure)
        write_json(output / "status.json", failure)
        raise


def synthetic_fixture():
    """One CPU fixture; it never opens any actual field/cache or Torch/CUDA."""
    import numpy as np
    from scipy.sparse import csr_matrix, block_diag
    xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1], [.5,0,0],[.5,.5,0],[0,.5,0],
        [0,0,.5],[.5,0,.5],[0,.5,.5]], np.float32)
    cells = np.arange(10, dtype=np.int64)[None, :]
    W, edges, vectors, lengths = graph_from_cells(xyz, cells)
    expected = sorted(tuple(sorted((c, 4 + k))) for k, edge in enumerate(EDGE_ORDER) for c in edge)
    require(edges.tolist() == [list(p) for p in expected] and len(edges) == 12, "Synthetic VTK midside edge ordering failed")
    roles = np.zeros(10, np.uint8)
    roles[0], roles[4] = 1, 64
    named = np.zeros((10,5), np.int64)
    named[0,0], named[4,4] = 1,1
    arrays, facts = build_hierarchy(xyz, cells, np.array([1/6], np.float64), roles, named,
                                   soft_targets=(4,2), hard_maxima=(16,16))
    require(facts["mass_and_source_count_conservation"] and (arrays["named5_counts2"].sum(0) == named.sum(0)).all(), "Fixture conservation failed")
    # The same fixture includes one genuine isolated positive-mass abstract row.
    Wi = clean_csr(block_diag((W, csr_matrix((1,1)))))
    ri = np.r_[np.zeros(10,np.uint8), np.uint8(64)]
    oi, pi, ai, ci, mi, bi = connected_owners(Wi, ri, np.arange(11), 4,16,0)
    from scipy.sparse import csr_matrix as csr
    G = csr((np.ones(11),(np.arange(11),oi)),shape=(11,int(oi.max())+1))
    P, p_fact = prolongation(Wi,G,ri,ci,0)
    isolated = P.getrow(10)
    require(pi[10] == 10 and isolated.nnz == 1 and isolated.indices[0] == oi[10]
            and isolated.data[0] == 1 and p_fact["isolated_identity_rows"] == 1, "Isolated singleton/identity policy failed")
    return {"status":"passed", "fixture":"one synthetic TETRA10 hierarchy plus isolated abstract positive-mass row",
        "edge_count":len(edges),"N0_N1_N2":[facts[k] for k in ("N0","N1","N2")],
        "checks":["VTK 12 subedge ordering","exact role BFS parent connectivity","quotient symmetry/cut counts",
            "hard mass/named5 conservation","nonnegative complete P/R and row sums/constants",
            "isolated singleton identity row"],"actual_arrays_read":False,"labels_read":0,"CUDA_initialized":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "notes/geobc_topology_context_plan.json"))
    parser.add_argument("--output", default=str(ROOT / "runs/geobc_exploration/topology_context_input_train4"))
    parser.add_argument("--dry-run", action="store_true", help="Small JSON/stat/source contract only")
    parser.add_argument("--self-test", action="store_true", help="Only the single synthetic CPU fixture")
    args = parser.parse_args()
    result = synthetic_fixture() if args.self_test else run(args)
    if args.dry_run or args.self_test:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps({k:result[k] for k in ("status","outcome","feasibility_pass","completed_CADs","cache_bytes","elapsed_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
