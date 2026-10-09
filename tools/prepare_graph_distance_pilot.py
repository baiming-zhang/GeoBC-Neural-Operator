"""One fixed TRAIN CAD's label-free TETRA10/interface mesh-edge distances.

request_contract/--dry-run read only small JSON/source and file stat. Runtime
is CPU only, opens geometry and input members, and never opens u/stress, a
model, or validation/dia/test arrays. Missing geometry/source/path distances
produce a blocked summary rather than a fabricated finite feature.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
CAD = "101_479"
EDGE_ORDER = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
MAP_TOL_MM = 2e-4
SOURCE_FILES = ("tools/parse_deepjeb_deck.py", "tools/prepare_deepjeb.py", "tools/mesh_boundary.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def host_path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable(path):
    path = host_path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path):
    return json.loads(host_path(path).read_text(encoding="utf-8"))


def file_identity(path, sha=False):
    path = host_path(path)
    stat = path.stat()
    return {"path": portable(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if sha else None}


def write_json(path, value):
    path = host_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_contract(plan_path=ROOT / "notes/geobc_graph_distance_pilot_plan.json"):
    """Metadata only: no scientific imports, array files, or deck parsing."""
    plan = read_json(plan_path)
    require(plan["status"] == "predeclared" and plan["CAD"] == CAD, "Requires this predeclared fixed TRAIN pilot")
    access = plan["data_access"]
    require(access["H5_members"] == ["vertices", "cells"] and access["prepared_members"] ==
            ["xyz", "features", "sample_key", "bbox_min_mm", "characteristic_length_mm"]
            and not access["u_or_stress_members"] and not access["validation_or_dia_or_final_test_arrays"]
            and not access["weights_or_models"], "Only declared geometry/input members are permitted")
    graph = plan["graph"]
    require(graph["midside_edges_in_slots_4_to_9"] == [list(edge) for edge in EDGE_ORDER]
            and graph["edges_per_tet"] == 12 and not graph["RBE_rigid_shortcuts"]
            and not graph["corner_clique_or_face_diagonal_edges"] and not graph["directed"],
            "The graph must use only unique undirected corner-mid subedges")
    execution = plan["execution"]
    require(execution["CPU_only"] and execution["training_updates"] == 0 and execution["GPU_calls"] == 0
            and not execution["download"], "This pilot cannot become training or a GPU job")
    split_path = host_path(plan["split"])
    split = read_json(split_path)
    require(CAD in split["groups"]["train"] and CAD not in split["groups"]["validation"] + split["groups"]["test"],
            "Fixed CAD must remain exclusively TRAIN")
    inputs = {name: file_identity(path) for name, path in plan["inputs"].items()}
    require(host_path(inputs["canonical_prepared"]["path"]) == host_path(split["case_paths"][f"{CAD}/ver"])
            and host_path(inputs["H5"]["path"]) == ROOT / f"data/raw/FieldMesh/{CAD}.h5"
            and host_path(inputs["SolverDeck"]["path"]) == ROOT / f"data/raw/SolverDeck/{CAD}.fem", "Source paths differ")
    return {"schema_version": 1, "plan": file_identity(plan_path, sha=True), "CAD": CAD,
            "split": file_identity(split_path, sha=True), "inputs": inputs,
            "sources_sha256": {name: file_identity(ROOT / name, sha=True)["sha256"] for name in SOURCE_FILES},
            "tool_sha256": file_identity(__file__, sha=True)["sha256"], "edge_order": [list(e) for e in EDGE_ORDER],
            "mapping_tolerance_mm": MAP_TOL_MM, "normalization": "original float32 vertices; xyz=(xyz_mm-min)/max_bbox_extent",
            "CPU_only": True, "fields_read": [], "model_or_weights_loaded": False}


def array_sha(array, dtype):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def map_members(ids, node_coordinates, tree):
    """Original nearest-coordinate mapping plus a strict candidate-ambiguity guard."""
    import numpy as np
    ids = np.asarray(ids, np.int64)
    require(len(ids) > 0 and len(np.unique(ids)) == len(ids), "Interface member IDs must be nonempty and unique")
    query = np.asarray([node_coordinates[int(node)] for node in ids], np.float64)
    require(np.isfinite(query).all(), "Nonfinite interface coordinates")
    distances, candidates = tree.query(query, k=2, workers=1)
    require(np.isfinite(distances[:, 0]).all() and distances[:, 0].max() <= MAP_TOL_MM,
            "Interface GRID coordinates cannot map to H5 vertices within2e-4mm")
    require(not (distances[:, 1] <= MAP_TOL_MM).any(), "Ambiguous multiple H5 vertices within interface mapping tolerance")
    indices = candidates[:, 0].astype(np.int64)
    require(len(np.unique(indices)) == len(ids), "Distinct interface GRID IDs map to the same H5 node")
    return indices, float(distances[:, 0].max())


def build_graph(xyz_mm, cells, length):
    import numpy as np
    from scipy.sparse import csr_matrix
    require(cells.ndim == 2 and cells.shape[1] == 10 and len(cells) > 0
            and cells.min() >= 0 and cells.max() < len(xyz_mm), "Invalid original TETRA10 connectivity")
    require(not (np.diff(np.sort(cells, axis=1), axis=1) == 0).any(), "Repeated node within a TETRA10 cell")
    edges = np.asarray(EDGE_ORDER, np.int64)
    mid = cells[:, 4:10]
    subedges = np.stack((np.stack((cells[:, edges[:, 0]], mid), axis=-1),
                         np.stack((mid, cells[:, edges[:, 1]]), axis=-1)), axis=2).reshape(-1, 2)
    unique = np.unique(np.sort(subedges, axis=1), axis=0)
    require(not (unique[:, 0] == unique[:, 1]).any(), "Self edges are not allowed")
    weights = np.linalg.norm(xyz_mm[unique[:, 1]] - xyz_mm[unique[:, 0]], axis=1) / length
    require(np.isfinite(weights).all() and (weights > 0).all(), "Mesh edges must have finite positive physical length")
    rows = np.concatenate((unique[:, 0], unique[:, 1]))
    columns = np.concatenate((unique[:, 1], unique[:, 0]))
    graph = csr_matrix((np.tile(weights, 2), (rows, columns)), shape=(len(xyz_mm), len(xyz_mm)))
    require(graph.nnz == 2 * len(unique), "CSR duplicate edge accumulation changed physical weights")
    return graph, unique, weights, len(subedges)


def distribution(values):
    import numpy as np
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    require(np.isfinite(values).all(), "Distribution received a nonfinite value")
    return {"count": len(values), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95))}


def run(args):
    output = host_path(args.output)
    try:
        contract = request_contract(args.plan)
    except Exception as exc:
        if args.dry_run:
            raise
        require(not output.exists() or not any(output.iterdir()), "Nonempty output is never silently overwritten")
        output.mkdir(parents=True, exist_ok=True)
        blocked = {"status": "blocked", "CAD": CAD, "error": f"{type(exc).__name__}: {exc}",
                   "phase": "metadata_preflight", "fields_read": [], "GPU_used": False}
        write_json(output / "summary.json", blocked)
        write_json(output / "status.json", blocked)
        return blocked
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "arrays_opened": False, "GPU_used": False}
    if output.exists() and any(output.iterdir()):
        previous = read_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        if previous.get("status") == "completed" and previous["config"]["contract"] == contract:
            require(file_identity(output / previous["cache"]["path"], sha=True)["sha256"] == previous["cache"]["sha256"],
                    "Completed input cache changed")
            return {"status": "completed_reused", "summary": portable(output / "summary.json")}
        raise FileExistsError("Nonempty incomplete/blocked output is preserved, not silently overwritten")
    output.mkdir(parents=True, exist_ok=True)
    config = {"contract": contract, "scope": "one fixed TRAIN geometry, no fields/weights/fit",
              "algorithm": "unique undirected12 corner-mid subedges/tet; five member-set multi-source Dijkstra distances/L",
              "comparison": "nearestEuclid to the identical member set, not master reference distance",
              "limitation": read_json(args.plan)["limits"]}
    write_json(output / "config.json", config)
    began = time.monotonic()
    timings, facts = {}, {}
    def status(phase):
        write_json(output / "status.json", {"status": phase, "elapsed_seconds": time.monotonic() - began,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "CAD": CAD})
    try:
        import h5py
        import numpy as np
        import scipy
        from scipy.spatial import cKDTree
        from scipy.sparse.csgraph import connected_components, dijkstra
        from tools.parse_deepjeb_deck import parse_deck
        from tools.mesh_boundary import midpoint_order_report
        start = time.monotonic()
        status("reading_geometry_and_input_parity")
        with h5py.File(host_path(contract["inputs"]["H5"]["path"]), "r") as stored:
            raw_dtype = str(stored["vertices"].dtype)
            xyz_mm = np.asarray(stored["vertices"], np.float32)
            cells = np.asarray(stored["cells"], np.int64)
        require(xyz_mm.ndim == 2 and xyz_mm.shape[1] == 3 and len(xyz_mm) > 1 and np.isfinite(xyz_mm).all(), "Invalid original geometry")
        lower = xyz_mm.min(0)
        length = float((xyz_mm.max(0) - lower).max())
        require(np.isfinite(length) and length > 0, "Invalid canonical isotropic length")
        xyz = ((xyz_mm - lower) / length).astype(np.float32)
        with np.load(host_path(contract["inputs"]["canonical_prepared"]["path"]), allow_pickle=False) as stored:
            prepared_xyz = np.asarray(stored["xyz"], np.float32)
            features = np.asarray(stored["features"], np.float32)
            sample_key = str(stored["sample_key"].item())
            prepared_lower = np.asarray(stored["bbox_min_mm"], np.float32)
            prepared_length = float(stored["characteristic_length_mm"].item())
        require(sample_key == f"{CAD}/ver" and np.array_equal(prepared_xyz, xyz)
                and np.array_equal(prepared_lower, lower) and prepared_length == length,
                "Prepared and original H5 geometry/node order/normalization differ")
        require(features.shape == (len(xyz), 25) and np.isfinite(features).all(), "Invalid original input25")
        order = midpoint_order_report(cells, xyz_mm)
        require(order["midside_deviation_over_edge_p50"] <= .05 and order["midside_deviation_over_edge_p95"] <= .25,
                "Original TETRA10 midsides do not match required edge order")
        facts.update(node_count=len(xyz), cell_count=len(cells), xyz_sha256=array_sha(xyz, "<f4"),
            vertices_mm_sha256=array_sha(xyz_mm, "<f4"), cells_sha256=array_sha(cells, "<i8"),
            raw_vertices_dtype=raw_dtype, geometry_policy="source vertices castfloat32 as originalprepare, promoteFP64 for physical edge weights",
            normalization={"bbox_min_mm": lower.tolist(), "characteristic_length_mm": length}, tetra10_order=order)
        timings["geometry_and_input_read_seconds"] = time.monotonic() - start
        start = time.monotonic()
        status("mapping_actual_interface_members")
        deck = parse_deck(host_path(contract["inputs"]["SolverDeck"]["path"]))
        supports = sorted(deck["fixed_groups"], key=lambda group: group["master_node_id"])
        require(len(supports) == 4 and len({s["master_node_id"] for s in supports}) == 4, "Need exactly four unambiguous support interfaces")
        groups = supports + [deck["loaded_group"]]
        require(len({g["master_node_id"] for g in groups}) == 5, "Support/load reference groups are ambiguous")
        tree = cKDTree(xyz_mm.astype(np.float64))
        records, mapped, all_ids = [], [], []
        expected_flags = np.zeros((len(xyz), 7), np.float32)
        for channel, group in enumerate(groups):
            indices, error = map_members(group["member_node_ids"], deck["node_coordinates"], tree)
            role = "support" if channel < 4 else "load"
            name = f"support_{group['master_node_id']}" if channel < 4 else f"load_{group['master_node_id']}"
            records.append({"channel": channel, "name": name, "role": role, "card_type": "RBE2" if channel < 4 else "RBE3",
                "element_id": int(group["element_id"]), "master_node_id": int(group["master_node_id"]),
                "members": len(indices), "member_GRID_IDs_sha256": array_sha(group["member_node_ids"], "<i8"),
                "source_node_indices_sha256": array_sha(indices, "<i8"), "max_coordinate_mapping_error_mm": error})
            if channel < 4:
                for dof in group["dofs"]:
                    require(dof in "123456", "Unsupported support DOF")
                    expected_flags[indices, int(dof)-1] = 1
            else:
                expected_flags[indices, 6] = 1
            mapped.append(indices)
            all_ids.append(np.asarray(group["member_node_ids"], np.int64))
        require(np.array_equal(expected_flags, features[:, :7]), "Actual interface members disagree with original input flags0:7")
        require(np.allclose(features[:, 24], length/200, rtol=2e-7, atol=1e-8), "Original input length slot differs")
        facts["source_groups"] = records
        facts["input_parity"] = {"normalized_xyz_bbox_length_exact": True, "original_flags_0_to6_exact": True,
                                  "length_feature_24_matches": True}
        timings["deck_parse_and_member_mapping_seconds"] = time.monotonic() - start
        start = time.monotonic()
        status("constructing_unique_mesh_subedge_graph")
        graph, unique_edges, edge_weights, raw_edges = build_graph(xyz_mm.astype(np.float64), cells, length)
        degrees = np.diff(graph.indptr)
        n_components, component_ids = connected_components(graph, directed=False)
        isolated = np.flatnonzero(degrees == 0)
        facts["graph"] = {"raw_subedge_instances": raw_edges, "unique_undirected_edges": len(unique_edges),
            "CSR_directed_entries": int(graph.nnz), "edge_pairs_sha256": array_sha(unique_edges, "<i8"),
            "normalized_edge_weights_sha256": array_sha(edge_weights, "<f8"), "edge_weights": distribution(edge_weights),
            "isolated_nodes": len(isolated), "connected_components": n_components,
            "component_node_counts": np.bincount(component_ids).tolist(), "duplicates_removed_before_CSR": True,
            "rigid_shortcuts": False, "all_original_nodes_checked": len(xyz)}
        timings["graph_construction_seconds"] = time.monotonic() - start
        start = time.monotonic()
        status("computing_five_source_set_distances")
        distances = np.empty((len(xyz), 5), np.float64)
        reachable = np.zeros((len(xyz), 5), bool)
        blocked = []
        for channel, indices in enumerate(mapped):
            graph_distance = dijkstra(graph, directed=False, indices=indices, min_only=True)
            require(graph_distance.shape == (len(xyz),) and not np.isnan(graph_distance).any(), "Invalid multi-source graph output")
            distances[:, channel] = graph_distance
            finite = np.isfinite(graph_distance)
            reachable[:, channel] = finite
            source_mask = np.zeros(len(xyz), bool)
            source_mask[indices] = True
            require((graph_distance[indices] == 0).all(), "Actual source members must have graph distance0")
            euclidean = cKDTree(xyz_mm[indices].astype(np.float64)).query(xyz_mm.astype(np.float64), workers=1)[0] / length
            require(np.isfinite(euclidean).all(), "Nonfinite same-source Euclidean distance")
            if channel == 4:
                require(np.allclose(euclidean, features[:, 23], rtol=3e-7, atol=3e-8), "Loaded-member Euclidean slot23 differs")
                facts["input_parity"]["loaded_member_Euclidean_feature_23_matches"] = True
            eligible = finite & ~source_mask
            require((euclidean[eligible] > 0).all(), "Non-source node coincides with source members; ratio undefined")
            ratio = graph_distance[eligible] / euclidean[eligible]
            require(not len(ratio) or ratio.min() >= 1-1e-10, "Graph distance is shorter than same-source Euclidean lower bound")
            g, e = graph_distance[eligible], euclidean[eligible]
            corr = float(np.corrcoef(g, e)[0,1]) if len(g) > 1 and g.std() > 0 and e.std() > 0 else None
            records[channel].update(unreachable_nodes=int((~finite).sum()), finite_nodes=int(finite.sum()),
                graph_distance=distribution(graph_distance[finite]), nearest_Euclidean_distance=distribution(euclidean),
                non_source_reachable_nodes=int(eligible.sum()), detour_ratio=distribution(ratio),
                graph_Euclidean_correlation=corr,
                excess_distance_RMS=float(np.sqrt(np.mean((g-e)**2))) if len(g) else None,
                detour_fraction_gt_1p01=float((ratio > 1.01).mean()) if len(ratio) else None,
                detour_fraction_gt_1p1=float((ratio > 1.1).mean()) if len(ratio) else None,
                detour_fraction_gt_1p5=float((ratio > 1.5).mean()) if len(ratio) else None,
                detour_fraction_gt_2=float((ratio > 2).mean()) if len(ratio) else None)
            if not finite.all():
                blocked.append(f"{records[channel]['name']}: {(~finite).sum()} originalnodes unreachable")
        timings["Dijkstra_and_Euclidean_comparison_seconds"] = time.monotonic() - start
        np.savez(output / "reachability.npz", reachable=reachable, isolated_nodes=isolated,
                 component_id=component_ids, xyz_sha256=np.asarray(facts["xyz_sha256"]))
        facts["reachability"] = file_identity(output / "reachability.npz", sha=True)
        require(request_contract(args.plan) == contract, "Source/plan identities changed during pilot")
        if blocked:
            raise ValueError("; ".join(blocked))
        start = time.monotonic()
        feature_names = [r["name"] + "_mesh_edge_distance_over_length" for r in records]
        cached = distances.astype(np.float32)
        require(np.isfinite(cached).all(), "Float32 input cache is nonfinite; do not clip/infill")
        offsets = np.concatenate(([0], np.cumsum([len(s) for s in mapped])))
        cache_path = output / "input_graph_distance.npz"
        np.savez(cache_path, features=cached, source_node_indices=np.concatenate(mapped),
            source_GRID_IDs=np.concatenate(all_ids), source_offsets=offsets,
            feature_names=np.asarray(feature_names), node_count=np.asarray(len(xyz)), CAD=np.asarray(CAD),
            xyz_sha256=np.asarray(facts["xyz_sha256"]), edges_sha256=np.asarray(facts["graph"]["edge_pairs_sha256"]),
            characteristic_length_mm=np.asarray(length), target_independent=np.asarray(True))
        timings["cache_write_seconds"] = time.monotonic() - start
        cache = file_identity(cache_path, sha=True)
        cache["path"] = cache_path.name
        facts["cache_features_sha256"] = array_sha(cached, "<f4")
        summary = {"status": "completed", "config": config, **facts, "cache": cache,
            "timings": timings, "elapsed_seconds": time.monotonic() - began,
            "software": {"numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__},
            "fields_read": [], "GPU_used": False, "model_or_weights_loaded": False, "training_updates": 0,
            "interpretation": "Source-set mesh-edge detour can demonstrate input non-redundancy, not predictive benefit, exact geodesics, stiffness or elasticity."}
        write_json(output / "summary.json", summary)
        status("completed")
        return {"status": "completed", "summary": portable(output / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}
    except Exception as exc:
        summary = {"status": "blocked", "config": config, **facts, "timings": timings,
            "elapsed_seconds": time.monotonic() - began, "error": f"{type(exc).__name__}: {exc}",
            "fields_read": [], "GPU_used": False, "training_updates": 0, "finite_features_fabricated": False}
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "blocked", "error": summary["error"], "elapsed_seconds": summary["elapsed_seconds"]})
        return {"status": "blocked", "summary": portable(output / "summary.json"), "error": summary["error"]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default="notes/geobc_graph_distance_pilot_plan.json")
    result.add_argument("--output", default="runs/geobc_exploration/graph_distance_pilot_101_479")
    result.add_argument("--dry-run", action="store_true", help="smallJSON/source/stat only; no arrays, weights or scientific imports")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, allow_nan=False))
