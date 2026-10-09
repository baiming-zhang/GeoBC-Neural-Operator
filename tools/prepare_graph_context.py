"""Cache both same-interface Euclidean and mesh-edge distances for45 CADs.

Only canonical ver geometry/input members of train40+validation5 are opened.
Exact metadata/source contracts permit reuse of completed CADs without reading
their arrays again. No labels, scalers, models, GPU, test, or dia fields.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == "nt" and (ROOT / ".local_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local_deps"))
from tools.prepare_graph_distance_pilot import (EDGE_ORDER, MAP_TOL_MM, array_sha, build_graph,
    distribution, file_identity, host_path, map_members, portable, read_json, require, write_json)

FEATURE_NAMES = [f"support_member_{index}_distance_over_length" for index in range(1, 5)] + ["loaded_member_distance_over_length"]
SOURCE_FILES = ("tools/prepare_graph_distance_pilot.py", "tools/parse_deepjeb_deck.py", "tools/mesh_boundary.py", "tools/prepare_deepjeb.py")


def request_contract(plan_path=ROOT / "notes/geobc_graph_context_data_plan.json"):
    """Small JSON/source/stat only; no scientific imports or array members."""
    plan = read_json(plan_path)
    require(plan["status"] == "predeclared" and plan["groups"] == ["train", "validation"]
            and plan["canonical_load"] == "ver" and all(plan["excluded"].values()), "Must remain the45 CAD input-only scope")
    require(plan["distance_arrays"] == ["euclidean", "graph"] and plan["tree_workers"] == 1 and plan["workers"] == 4
            and plan["midside_edges_in_slots_4_to_9"] == [list(edge) for edge in EDGE_ORDER]
            and plan["data_members"]["H5"] == ["vertices", "cells"]
            and plan["data_members"]["prepared"] == ["xyz", "features", "sample_key", "bbox_min_mm", "characteristic_length_mm"],
            "Declared arrays/geometry/mapping recipe changed")
    pilot_sha = file_identity(plan["pilot_source"], sha=True)["sha256"]
    require(pilot_sha == plan["pilot_source_sha256"], "Frozen pilot source changed")
    split = read_json(plan["split"])
    groups = split["groups"]
    cads = groups["train"] + groups["validation"]
    require(len(groups["train"]) == 40 and len(groups["validation"]) == 5 and len(cads) == len(set(cads)) == 45
            and not (set(cads) & set(groups["test"])), "Exactly train40+validation5, no final test, is required")
    inputs = {}
    for cad in cads:
        canonical = split["case_paths"][f"{cad}/ver"]
        require(host_path(canonical).stem == f"{cad}_ver", "Only canonicalver prepared inputs allowed")
        inputs[cad] = {"H5": file_identity(host_path(plan["raw_geometry_directory"]) / f"{cad}.h5"),
                       "SolverDeck": file_identity(host_path(plan["SolverDeck_directory"]) / f"{cad}.fem"),
                       "canonical_prepared": file_identity(canonical)}
    return {"schema_version": 1, "plan": file_identity(plan_path, sha=True), "split": file_identity(plan["split"], sha=True),
            "groups": groups, "CAD_order": cads, "canonical_load": "ver", "inputs": inputs,
            "source_sha256": {name: file_identity(name, sha=True)["sha256"] for name in SOURCE_FILES},
            "tool_sha256": file_identity(__file__, sha=True)["sha256"], "feature_names": FEATURE_NAMES,
            "array_keys": ["euclidean", "graph"], "units": "mm/characteristic_length_mm",
            "mapping_tolerance_mm": MAP_TOL_MM, "target_members_read": [], "scalers_fitted": False,
            "models_or_weights_loaded": False, "CPU_only": True}


def same_stat(path, record):
    identity = file_identity(path)
    require(identity["bytes"] == record["bytes"] and identity["mtime_ns"] == record["mtime_ns"], "Cached input identity changed")


def prepare_one(task):
    """One graph/input read perCAD produces BOTH5-column distance arrays."""
    cad, source, output_name, contract_sha = task
    output = host_path(output_name)
    began = time.monotonic()
    facts, timings = {}, {}
    try:
        import h5py
        import numpy as np
        from scipy.spatial import cKDTree
        from scipy.sparse.csgraph import connected_components, dijkstra
        from tools.parse_deepjeb_deck import parse_deck
        from tools.mesh_boundary import midpoint_order_report
        tick = time.monotonic()
        for identity in source.values():
            same_stat(identity["path"], identity)
        with h5py.File(host_path(source["H5"]["path"]), "r") as stored:
            raw_dtype = str(stored["vertices"].dtype)
            xyz_mm = np.asarray(stored["vertices"], np.float32)
            cells = np.asarray(stored["cells"], np.int64)
        require(xyz_mm.ndim == 2 and xyz_mm.shape[1] == 3 and len(xyz_mm) > 1 and np.isfinite(xyz_mm).all(), "Invalid geometry")
        lower = xyz_mm.min(0)
        length = float((xyz_mm.max(0) - lower).max())
        require(math.isfinite(length) and length > 0, "Invalid canonical isotropic length")
        xyz = ((xyz_mm-lower)/length).astype(np.float32)
        with np.load(host_path(source["canonical_prepared"]["path"]), allow_pickle=False) as stored:
            prepared_xyz = np.asarray(stored["xyz"], np.float32)
            features = np.asarray(stored["features"], np.float32)
            sample_key = str(stored["sample_key"].item())
            prepared_lower = np.asarray(stored["bbox_min_mm"], np.float32)
            prepared_length = float(stored["characteristic_length_mm"].item())
        require(sample_key == f"{cad}/ver" and np.array_equal(xyz, prepared_xyz)
                and np.array_equal(lower, prepared_lower) and length == prepared_length,
                "Original/prepared XYZ node order or normalization differs")
        require(features.shape == (len(xyz), 25) and np.isfinite(features).all(), "Invalid input25")
        order = midpoint_order_report(cells, xyz_mm)
        require(order["midside_deviation_over_edge_p50"] <= .05 and order["midside_deviation_over_edge_p95"] <= .25,
                "Original TETRA10 midside ordering failed")
        facts.update(node_count=len(xyz), cell_count=len(cells), xyz_sha256=array_sha(xyz, "<f4"),
            vertices_mm_sha256=array_sha(xyz_mm, "<f4"), cells_sha256=array_sha(cells, "<i8"),
            source_vertices_dtype=raw_dtype, normalization={"bbox_min_mm": lower.tolist(), "characteristic_length_mm": length},
            geometry_policy="originalprepare float32 vertices promoted toFP64 for physical edge lengths", tetra10_order=order)
        timings["input_read_geometry_parity_seconds"] = time.monotonic()-tick
        tick = time.monotonic()
        deck = parse_deck(host_path(source["SolverDeck"]["path"]))
        supports = sorted(deck["fixed_groups"], key=lambda group: group["master_node_id"])
        require(len(supports) == 4 and len({g["master_node_id"] for g in supports}) == 4, "Need exactly four support member groups")
        groups = supports + [deck["loaded_group"]]
        require(len({g["master_node_id"] for g in groups}) == 5, "Ambiguous source groups")
        tree = cKDTree(xyz_mm.astype(np.float64))
        flags = np.zeros((len(xyz), 7), np.float32)
        mapped, member_ids, records = [], [], []
        for index, group in enumerate(groups):
            indices, error = map_members(group["member_node_ids"], deck["node_coordinates"], tree)
            source_type = "RBE3" if "node_dofs" in group and "reference_dofs" in group else "RBE2"
            records.append({"channel": index, "role": "support" if index < 4 else "load",
                "master_node_id": int(group["master_node_id"]), "master_coords_mm": list(group["master_coords"]),
                "element_id": int(group["element_id"]), "card_type": source_type, "member_count": len(indices),
                "member_GRID_IDs_sha256": array_sha(group["member_node_ids"], "<i8"),
                "member_node_indices_sha256": array_sha(indices, "<i8"), "max_mapping_error_mm": error})
            if index < 4:
                require(source_type == "RBE2", "Support source must be the actual RBE2 member group")
                for dof in group["dofs"]:
                    require(dof in "123456", "Unknown support DOF")
                    flags[indices, int(dof)-1] = 1
            else:
                flags[indices, 6] = 1
            mapped.append(indices)
            member_ids.append(np.asarray(group["member_node_ids"], np.int64))
        require(np.array_equal(flags, features[:, :7]) and np.allclose(features[:, 24], length/200, rtol=2e-7, atol=1e-8),
                "Original interface flags/length differ from source membership")
        facts["source_groups"] = records
        timings["deck_member_mapping_seconds"] = time.monotonic()-tick
        tick = time.monotonic()
        graph, edges, weights, raw_edges = build_graph(xyz_mm.astype(np.float64), cells, length)
        components, component_ids = connected_components(graph, directed=False)
        isolated = np.flatnonzero(np.diff(graph.indptr) == 0)
        facts["graph_topology"] = {"raw_subedge_instances": raw_edges, "unique_undirected_edges": len(edges),
            "CSR_entries": int(graph.nnz), "edges_sha256": array_sha(edges, "<i8"),
            "normalized_weights_sha256": array_sha(weights, "<f8"), "connected_components": components,
            "component_node_counts": np.bincount(component_ids).tolist(), "isolated_nodes": len(isolated),
            "isolated_first20_node_indices": isolated[:20].tolist(), "all_original_nodes_checked": len(xyz),
            "edges": "12 corner-mid subedges per originalTETRA10; unique; no RBE shortcuts"}
        timings["graph_construction_seconds"] = time.monotonic()-tick
        tick = time.monotonic()
        euclidean = np.empty((len(xyz), 5), np.float64)
        distances = np.empty_like(euclidean)
        blocked = []
        for index, indices in enumerate(mapped):
            g = dijkstra(graph, directed=False, indices=indices, min_only=True)
            e = cKDTree(xyz_mm[indices].astype(np.float64)).query(xyz_mm.astype(np.float64), workers=1)[0]/length
            require(g.shape == (len(xyz),) and np.isfinite(e).all() and not np.isnan(g).any(), "Invalid distance result")
            require((g[indices] == 0).all() and (e[indices] == 0).all(), "Source members must have distance0")
            finite = np.isfinite(g)
            source_mask = np.zeros(len(xyz), bool); source_mask[indices] = True
            eligible = finite & ~source_mask
            require((e[eligible] > 0).all(), "Ambiguous coincident non-source nodes")
            ratio = g[eligible]/e[eligible]
            require(not len(ratio) or ratio.min() >= 1-1e-10, "Graph distance violates Euclidean lower bound")
            if index == 4:
                require(np.allclose(e, features[:, 23], rtol=3e-7, atol=3e-8), "Input loaded-member Euclidean slot23 differs")
            records[index].update(unreachable_nodes=int((~finite).sum()), finite_nodes=int(finite.sum()),
                graph=distribution(g[finite]), euclidean=distribution(e), detour_non_source=distribution(ratio),
                graph_Euclidean_correlation=float(np.corrcoef(g[eligible], e[eligible])[0,1])
                    if eligible.sum() > 1 and g[eligible].std() > 0 and e[eligible].std() > 0 else None)
            if not finite.all():
                records[index]["unreachable_first20_node_indices"] = np.flatnonzero(~finite)[:20].tolist()
                blocked.append(f"channel{index}: {int((~finite).sum())} unreachable originalnodes")
            euclidean[:, index], distances[:, index] = e, g
        facts["input_parity"] = {"XYZ_bbox_length_exact": True, "flags0_to6_exact": True,
                                  "length_feature24": True, "loaded_member_Euclidean_feature23": True}
        timings["both_five_distance_metrics_seconds"] = time.monotonic()-tick
        require(not blocked, "; ".join(blocked))
        euclidean, distances = euclidean.astype(np.float32), distances.astype(np.float32)
        require(np.isfinite(euclidean).all() and np.isfinite(distances).all(), "Float32 cache nonfinite; no clipping/infill")
        for identity in source.values():
            same_stat(identity["path"], identity)
        tick = time.monotonic()
        record = {"status": "completed", "cad_id": cad, "canonical_case": f"{cad}/ver", "contract_sha256": contract_sha,
            "input_sources": source, **facts, "feature_names": FEATURE_NAMES, "array_keys": ["euclidean", "graph"],
            "units": "physical mm distance / characteristic_length_mm", "array_dtype": "float32", "finite": True,
            "target_independent": True, "labels_read": 0, "targets_fitted": False, "input_labels_read": 0,
            "euclidean_features_sha256": array_sha(euclidean, "<f4"), "graph_features_sha256": array_sha(distances, "<f4"),
            "timings": timings, "elapsed_seconds": time.monotonic()-began}
        target = output / f"{cad}.npz"
        temporary = target.with_suffix(".tmp.npz")
        offsets = np.concatenate(([0], np.cumsum([len(s) for s in mapped])))
        with temporary.open("wb") as handle:
            np.savez(handle, euclidean=euclidean, graph=distances,
                source_node_indices=np.concatenate(mapped), source_GRID_IDs=np.concatenate(member_ids), source_offsets=offsets,
                feature_names=np.asarray(FEATURE_NAMES), node_count=np.asarray(len(xyz)), cad_id=np.asarray(cad),
                xyz_sha256=np.asarray(facts["xyz_sha256"]), characteristic_length_mm=np.asarray(length),
                contract_sha256=np.asarray(contract_sha), target_independent=np.asarray(True),
                record_json=np.asarray(json.dumps(record, ensure_ascii=False, allow_nan=False)))
        temporary.replace(target)
        timings["cache_write_seconds"] = time.monotonic()-tick
        record["elapsed_seconds"] = time.monotonic()-began
        record.update(file=target.name, **{key: value for key, value in file_identity(target, sha=True).items() if key != "path"})
        return {"status": "completed", "CAD": cad, "record": record}
    except Exception as exc:
        return {"status": "blocked", "CAD": cad, "error": f"{type(exc).__name__}: {exc}", "input_sources": source,
                "facts": facts, "timings": timings, "elapsed_seconds": time.monotonic()-began,
                "labels_read": 0, "finite_features_fabricated": False}


def recover_record(path, expected_sha, source):
    """Only embedded small JSON metadata for an interrupted worker's atomic file."""
    import numpy as np
    with np.load(path, allow_pickle=False) as stored:
        record = json.loads(str(stored["record_json"].item()))
    require(record["contract_sha256"] == expected_sha and record["input_sources"] == source
            and record["finite"] and record["labels_read"] == 0 and record["status"] == "completed", "Existing CADcache contract differs")
    record.update(file=Path(path).name, **{key: value for key, value in file_identity(path, sha=True).items() if key != "path"})
    return record


def contract_sha(contract):
    import hashlib
    return hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run(args):
    contract = request_contract(args.plan)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "arrays_opened": False}
    require(1 <= args.workers <= 4, "At most4 CPU workers, with treeworkers1")
    output = host_path(args.output)
    fingerprint = contract_sha(contract)
    config = {"contract": contract, "contract_sha256": fingerprint, "scope": "45 input-only CADs",
              "constants": {"mapping_tolerance_mm": MAP_TOL_MM, "graph_edges_per_tet": 12, "tree_workers": 1,
                "normalization": "original canonical isotropic length", "dtype": "float32 arrays; float64 distance computation"},
              "comparison": "both same-member-set Euclidean5 and mesh-edge graph5; no target scalers fit"}
    records, failures = {}, {}
    previous_elapsed = 0.
    if output.exists() and any(output.iterdir()):
        require((output / "config.json").is_file() and read_json(output / "config.json") == config,
                "Nonempty data directory requires exact source/scope/recipe identity; preserve existing artifacts")
        old = read_json(output / "manifest.json") if (output / "manifest.json").is_file() else {}
        require(not old or old["contract_sha256"] == fingerprint, "Existing manifest differs")
        records = old.get("CADs", {})
        failures = old.get("failures", {})
        previous_elapsed = old.get("elapsed_seconds", 0.)
        for cad, record in records.items():
            require(cad in contract["CAD_order"] and record["contract_sha256"] == fingerprint
                    and record["input_sources"] == contract["inputs"][cad], "Recorded CAD cache source differs")
            same_stat(output / record["file"], record)
        if old.get("complete") and len(records) == 45:
            return {"status": "completed_reused", "manifest": portable(output / "manifest.json"), "CADs": 45}
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "config.json").exists():
        write_json(output / "config.json", config)
    began = time.monotonic()
    reused = len(records)
    for cad in contract["CAD_order"]:
        if cad not in records and (output / f"{cad}.npz").is_file():
            records[cad] = recover_record(output / f"{cad}.npz", fingerprint, contract["inputs"][cad])
            reused += 1
    remaining = [cad for cad in contract["CAD_order"] if cad not in records and cad not in failures]
    def manifest(status):
        result = {"schema_version": 1, "status": status, "complete": status == "completed", "contract": contract,
            "contract_sha256": fingerprint, "groups": contract["groups"], "split_sha256": contract["split"]["sha256"],
            "CAD_order": contract["CAD_order"], "CADs": records, "failures": failures, "canonical_load": "ver",
            "array_keys": ["euclidean", "graph"], "feature_names": FEATURE_NAMES, "channels_per_array": 5,
            "units": "physical mm / characteristic_length_mm", "constants": config["constants"],
            "source_group_order": "support member groups by ascending master_node_id then actual loadmembers",
            "labels_read": 0, "input_labels_read": 0, "target_scalers_fit": False, "GPU_used": False,
            "model_or_weights_loaded": False, "workers_this_invocation": args.workers, "tree_workers": 1,
            "reused_CADs_this_invocation": reused, "elapsed_seconds": previous_elapsed+time.monotonic()-began,
            "total_nodes": sum(record["node_count"] for record in records.values()),
            "total_cache_bytes": sum(record["bytes"] for record in records.values()),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": read_json(args.plan)["limits"]}
        write_json(output / "manifest.json", result)
        write_json(output / "status.json", {"status": status, "completed_CADs": len(records), "blocked_CADs": len(failures),
            "total_CADs": 45, "elapsed_seconds": result["elapsed_seconds"], "updated_at_utc": result["updated_at_utc"]})
        return result
    manifest("preparing")
    if remaining:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(prepare_one, (cad, contract["inputs"][cad], str(output), fingerprint)): cad for cad in remaining}
            for future in as_completed(pending):
                cad = pending[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "blocked", "CAD": cad, "error": f"worker {type(exc).__name__}: {exc}", "labels_read": 0}
                if result["status"] == "completed":
                    records[cad] = result["record"]
                else:
                    failures[cad] = result
                manifest("preparing")
                print(json.dumps({"CAD": cad, "status": result["status"], "completed": len(records), "blocked": len(failures)}, ensure_ascii=False), flush=True)
    require(request_contract(args.plan) == contract, "Scope/plan/source identities changed during preparation")
    result = manifest("completed" if len(records) == 45 and not failures else "blocked")
    return {"status": result["status"], "complete": result["complete"], "manifest": portable(output / "manifest.json"),
            "CADs": len(records), "blocked_CADs": len(failures), "elapsed_seconds": result["elapsed_seconds"],
            "total_nodes": result["total_nodes"], "total_cache_bytes": result["total_cache_bytes"]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", default="notes/geobc_graph_context_data_plan.json")
    result.add_argument("--output", default="data/geobc_graph_context")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--dry-run", action="store_true", help="smallJSON/source/stat only, no arrays or scientific imports")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, allow_nan=False))
