"""New frozen smallest-group topology input pilot; no target/model/GPU access.

Only connected ownership changes from the immutable bounded-BFS preparation.
The old module's input and numerical helpers are called without modifying its
globals. request_contract remains standard-library JSON/source/stat only.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import heapq
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_topology_context as base

PLAN_SHA = "63b0e2a1d680966f295e3358c2e3ed1923ddd19d7c762830edd739fbf3809b18"
BASE_SHA = "9ccffda61a05880401b8da346e77e873893e50c90e64d2ccd264b80bf6d96593"
ScientificBlock = base.ScientificBlock
require, path, read_json = base.require, base.path, base.read_json
identity, fingerprint, write_json = base.identity, base.fingerprint, base.write_json


def request_contract(plan_path=ROOT / "notes/geobc_topology_balanced_plan.json"):
    """Only small JSON/SHA/stat, including frozen parent rejection identity."""
    plan_id = identity(plan_path, sha=True)
    require(plan_id["sha256"] == PLAN_SHA, "The formally frozen balanced plan changed")
    plan = read_json(plan_path)
    adoption = plan["adoption"]
    require(plan["status"] == "predeclared" and plan["scope"]["CAD_order"] == base.CAD_ORDER
            and identity(adoption["base_plan"], sha=True)["sha256"] == adoption["base_plan_sha256"] == base.PLAN_SHA
            and identity(adoption["base_tool"], sha=True)["sha256"] == adoption["base_tool_sha256"] == BASE_SHA,
            "The frozen original plan/source/scope changed")
    parent_id = identity(adoption["parent_result"], sha=True)
    require(parent_id["sha256"] == adoption["parent_result_sha256"], "Parent rejection evidence changed")
    parent = read_json(adoption["parent_result"])
    old = base.request_contract(adoption["base_plan"])
    require(parent["status"] == "completed" and parent["feasibility_pass"] is False
            and parent["outcome"] == "infeasible" and parent["config"]["contract"] == old,
            "A scientifically completed original rejection on these exact inputs is required")
    original_plan = read_json(adoption["base_plan"])
    require(all(plan[key] == original_plan[key] for key in ("scope", "data", "graph", "prolongation"))
            and plan["hierarchy"]["soft_targets"] == original_plan["hierarchy"]["soft_targets"]
            and plan["hierarchy"]["hard_maxima"] == original_plan["hierarchy"]["hard_maxima"]
            and plan["output"]["max_cache_bytes"] == original_plan["output"]["max_cache_bytes"],
            "Only the predeclared ownership algorithm may change")
    require(plan["output"]["cache"] == "data/geobc_topology_balanced_train4"
            and plan["output"]["run"] == "runs/geobc_exploration/topology_context_balanced_train4", "New cache/run must be isolated")
    return {**old, "plan": plan_id, "base_plan": old["plan"], "base_source_sha256": BASE_SHA,
        "parent_evidence": parent_id, "source_sha256": {**old["source_sha256"],
            "tools/prepare_topology_context_balanced.py": identity(__file__, sha=True)["sha256"]},
        "cache": plan["output"]["cache"], "formal_result": plan["output"]["formal_result"].split(";")[0],
        "recipe": {**old["recipe"], "aggregation": "minimum mergeable current-level-size/minDesc group into minimum same-role actual neighbor",
            "initial_hard_cap_check": False, "desired": "max(soft_target, role_component_lower_bound)",
            "representative": "retain chosen neighbor; small-to-large adjacency migration; generation-checked heap and DSU",
            "connectivity": "final original current-level same-owner BFS parent/admission proof"}}


def balanced_owners(W, roles, descendant_min, soft_target, hard_maximum, level):
    """Deterministic actual-edge agglomeration, followed by one parent BFS."""
    import numpy as np
    began = time.monotonic()
    n = W.shape[0]
    require(n > 0 and roles.shape == (n,) and descendant_min.shape == (n,)
            and len(np.unique(descendant_min)) == n and W.has_sorted_indices,
            "Canonical role/descendant/sorted graph inputs required")
    lower = base.role_components(W, roles)
    if lower["lower_bound"] > hard_maximum:
        raise ScientificBlock("Exact-role component lower bound exceeds aggregate cap",
            {"level": level, "role_components": lower, "hard_maximum": hard_maximum})
    desired = max(soft_target, lower["lower_bound"])
    # Singleton count intentionally is not tested against the final hard cap.
    neighbors = [set(int(v) for v in W.indices[W.indptr[u]:W.indptr[u + 1]] if roles[v] == roles[u]) for u in range(n)]
    dsu = np.arange(n, dtype=np.int64)
    sizes = np.ones(n, np.int64)
    minimum = descendant_min.copy()
    active = np.ones(n, bool)
    generation = np.zeros(n, np.int64)
    heap = [(1, int(minimum[u]), u, 0) for u in range(n)]
    heapq.heapify(heap)
    count, unions, stale, terminal, migrations = n, 0, 0, 0, 0
    while count > desired:
        chosen = None
        while heap:
            size, minimum_id, a, version = heapq.heappop(heap)
            if (not active[a] or version != generation[a] or size != sizes[a] or minimum_id != minimum[a]):
                stale += 1
                continue
            if not neighbors[a]:
                terminal += 1  # Whole role component; cannot gain new neighbors later.
                continue
            chosen = a
            break
        require(chosen is not None, "Agglomeration exhausted mergeable edges above desired; adjacency/implementation invariant failed")
        a = chosen
        b = min(neighbors[a], key=lambda u: (int(sizes[u]), int(minimum[u]), u))
        require(active[b] and roles[a] == roles[b] and a in neighbors[b] and sizes[a] <= sizes[b],
                "Smallest-mergeable/symmetric-role/small-to-large invariant failed")
        popped_neighbors = tuple(neighbors[a])
        neighbors[b].remove(a)
        for other in popped_neighbors:
            if other == b:
                continue
            require(active[other] and a in neighbors[other] and roles[other] == roles[a], "Dynamic actual-edge symmetry failed")
            neighbors[other].remove(a)
            neighbors[other].add(b)
            neighbors[b].add(other)
        migrations += len(popped_neighbors)
        neighbors[a].clear()
        active[a] = False
        dsu[a] = b
        sizes[b] += sizes[a]
        minimum[b] = min(minimum[a], minimum[b])
        generation[a] += 1
        generation[b] += 1
        heapq.heappush(heap, (int(sizes[b]), int(minimum[b]), b, int(generation[b])))
        count -= 1
        unions += 1
    merge_elapsed = time.monotonic() - began

    def find(vertex):
        root = vertex
        while dsu[root] != root:
            root = int(dsu[root])
        while dsu[vertex] != vertex:
            following = int(dsu[vertex])
            dsu[vertex] = root
            vertex = following
        return root

    roots = np.fromiter((find(u) for u in range(n)), dtype=np.int64, count=n)
    representatives = np.flatnonzero(active)
    representatives = representatives[np.argsort(minimum[representatives], kind="stable")]
    root_to_owner = np.full(n, -1, np.int64)
    root_to_owner[representatives] = np.arange(len(representatives))
    owner = root_to_owner[roots]
    require(len(representatives) == count and (owner >= 0).all()
            and (roles == roles[representatives[owner]]).all(), "Final DSU coverage/homogeneity failed")
    if count > hard_maximum:
        raise ScientificBlock("Final aggregate count exceeds cap", {"level": level,
            "observed_aggregate_count": count, "desired": desired, "hard_maximum": hard_maximum})
    # Reconstruct the public parent witness from the original full graph, never
    # use the mutable same-role adjacency sets as quotient numerical weights.
    parent, admission = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    seen_owners = set()
    for seed in np.argsort(descendant_min, kind="stable"):
        if admission[seed] >= 0:
            continue
        group = int(owner[seed])
        require(group not in seen_owners, "Final owner contains disconnected original-edge components")
        seen_owners.add(group)
        parent[seed], admission[seed] = seed, 0
        queue, admitted = deque([int(seed)]), 1
        while queue:
            current = queue.popleft()
            for neighbor in W.indices[W.indptr[current]:W.indptr[current + 1]]:
                if owner[neighbor] == group and admission[neighbor] < 0:
                    parent[neighbor], admission[neighbor] = current, admitted
                    admitted += 1
                    queue.append(int(neighbor))
    require(len(seen_owners) == count and (admission >= 0).all() and (owner[parent] == owner).all()
            and (roles[parent] == roles).all(), "Final original-edge parent coverage/role proof failed")
    nonseed = np.flatnonzero(parent != np.arange(n))
    require((admission[parent[nonseed]] < admission[nonseed]).all(), "Parent is not earlier admitted")
    for u in nonseed:
        row = W.indices[W.indptr[u]:W.indptr[u + 1]]
        at = np.searchsorted(row, parent[u])
        require(at < len(row) and row[at] == parent[u], "Final parent is not an original actual edge")
    final_sizes = np.bincount(owner, minlength=count)
    require(np.array_equal(final_sizes, sizes[representatives]) and np.array_equal(
        np.minimum.reduceat(descendant_min[np.argsort(owner, kind="stable")], np.r_[0, np.cumsum(final_sizes)[:-1]]),
        minimum[representatives]), "Current-level sizes/minimum descendant IDs differ")
    facts = {"algorithm": "deterministic smallest-mergeable actual-edge agglomeration",
        "vertices": n, "initial_singleton_count": n, "aggregate_count": count, "desired": desired,
        "soft_target": soft_target, "hard_maximum": hard_maximum, "role_components": lower,
        "size_min": int(final_sizes.min()), "size_max": int(final_sizes.max()), "size_mean": float(final_sizes.mean()),
        "coverage_complete": True, "disjoint": True, "homogeneous": True,
        "parent_connectivity_witness": True, "renumbered_by_min_original_descendant": True,
        "merges": unions, "stale_heap_pops": stale, "terminal_component_pops": terminal,
        "migrated_adjacency_entries": migrations, "small_to_large_checked_every_union": True,
        "merge_seconds": merge_elapsed, "total_ownership_and_witness_seconds": time.monotonic() - began}
    return owner, parent, admission, roles[representatives], minimum[representatives], facts


def build_hierarchy(xyz, cells, volumes, role_bits, named, soft_targets=(2048, 128), hard_maxima=(4096, 512)):
    """Same immutable numerical helpers as parent; only owner call is new."""
    import numpy as np
    began, timings = time.monotonic(), {}
    W0, edges, vectors, lengths = base.graph_from_cells(xyz, cells)
    mass0 = np.zeros(len(xyz), np.float64)
    np.add.at(mass0, cells.ravel(), np.repeat(volumes / 10, 10))
    require(np.isfinite(mass0).all() and (mass0 > 0).all(), "Uncovered/zero-mass source node")
    require(math.isclose(float(mass0.sum()), float(volumes.sum()), rel_tol=1e-12, abs_tol=1e-12), "Source mass sum differs from cell volumes")
    timings["graph_and_mass_seconds"] = time.monotonic() - began
    roles0 = np.asarray(role_bits, np.uint8)
    raw0 = ((roles0[:, None] >> np.arange(7)) & 1).astype(np.int64)
    arrays = {"xyz0_normalized": np.asarray(xyz, np.float32), "role_bits0": roles0, "mass0_mm3": mass0,
        "named5_counts0": named.astype(np.int64), "role_member_counts0": raw0, "edge_endpoints": edges.astype(np.int64),
        "edge_vectors_normalized": vectors, "edge_lengths_normalized": lengths}
    source = {"W": W0, "mass": mass0, "xyz": xyz, "roles": roles0, "named": named, "raw": raw0,
        "descendant": np.arange(len(xyz), dtype=np.int64)}
    graph_records, owner_records, maps = [], [], {}
    for level in range(2):
        tick = time.monotonic()
        graph_record, degree = base.graph_facts(source["W"])
        graph_records.append(graph_record)
        arrays[f"degree{level}"] = degree
        base.pack_csr(arrays, f"W{level}", source["W"], np.float64)
        owner, parent, admission, roles, descendant, facts = balanced_owners(source["W"], source["roles"],
            source["descendant"], soft_targets[level], hard_maxima[level], level)
        owner_records.append(facts)
        arrays[f"aggregate{level}{level + 1}"], arrays[f"bfs_parent{level}"] = owner, parent
        arrays[f"bfs_admission_rank{level}"], arrays[f"descendant_min{level}"] = admission, source["descendant"]
        W, G, R, mass, centers, named_counts, raw_counts = base.aggregate(source["W"], owner,
            source["mass"], source["xyz"], source["named"], source["raw"])
        require(np.array_equal(named_counts.sum(0), named.sum(0)) and np.array_equal(raw_counts.sum(0), raw0.sum(0))
                and math.isclose(float(mass.sum()), float(mass0.sum()), rel_tol=1e-12, abs_tol=1e-12), "Hard source/mass counts not conserved")
        saved_R, r_facts = base.mapping_facts(R)
        P, p_facts = base.prolongation(source["W"], G, source["roles"], roles, level)
        maps[f"P{level}{level + 1}"], maps[f"R{level}{level + 1}"] = p_facts, r_facts
        base.pack_csr(arrays, f"P{level}{level + 1}", P, np.float32)
        base.pack_csr(arrays, f"R{level}{level + 1}", saved_R, np.float32)
        arrays[f"mass{level + 1}_mm3"], arrays[f"xyz{level + 1}_normalized"] = mass, centers.astype(np.float32)
        arrays[f"role_bits{level + 1}"], arrays[f"named5_counts{level + 1}"] = roles, named_counts
        arrays[f"role_member_counts{level + 1}"] = raw_counts
        timings[f"level{level}_hierarchy_and_maps_seconds"] = time.monotonic() - tick
        source = {"W": W, "mass": mass, "xyz": centers, "roles": roles, "named": named_counts,
                  "raw": raw_counts, "descendant": descendant}
    last_graph, last_degree = base.graph_facts(source["W"])
    graph_records.append(last_graph)
    base.pack_csr(arrays, "W2", source["W"], np.float64)
    arrays["degree2"], arrays["descendant_min2"] = last_degree, source["descendant"]
    return arrays, {"N0": len(xyz), "N1": len(arrays["mass1_mm3"]), "N2": len(arrays["mass2_mm3"]),
        "graphs": graph_records, "ownership": owner_records, "maps": maps, "timings": timings,
        "input_volume_proxy_sum_mm3": float(mass0.sum()), "named5_member_counts": named.sum(0).astype(int).tolist(),
        "mass_and_source_count_conservation": True, "labels_read": 0, "CUDA_initialized": False}


def prepare_one(cad, source, cache, contract_sha, current_bytes):
    import numpy as np
    began = time.monotonic()
    xyz, cells, volumes, roles, named = base.read_inputs(cad, source)
    input_elapsed = time.monotonic() - began
    arrays, facts = build_hierarchy(xyz, cells, volumes, roles, named)
    require(base.array_sha(arrays["edge_endpoints"], "<i8") == source["edges_sha256"], "Original edge structural hash differs")
    facts["timings"]["input_reads_seconds"] = input_elapsed
    record = {"status": "completed", "cad_id": cad, "canonical_case": f"{cad}/ver", "contract_sha256": contract_sha,
        "source": source, "xyz_sha256": source["xyz_sha256"], "normalization": source["normalization"],
        "source_groups": source["source_groups"], "facts": facts, "plain_array_bytes": sum(x.nbytes for x in arrays.values()),
        "units": {"xyz_and_edges": "mm / characteristic_length_mm", "mass": "corner P1 cell mm3 / 10 per original incident node", "W": "binary or cut multiplicity", "P_R": "dimensionless"},
        "accessed_members": {"prepared": ["xyz", "features", "sample_key", "label_alignment", "bbox_min_mm", "characteristic_length_mm"],
            "gradient": ["cells10", "volumes_mm3"], "interface": ["euclidean", "cad_id", "xyz_sha256"]},
        "aggregation": "balanced actual-edge smallest-group contraction", "labels_read": 0,
        "raw_sources_opened": False, "CUDA_initialized": False, "target_independent": True,
        "elapsed_seconds": time.monotonic() - began, "peak_RSS": base.peak_rss()}
    if current_bytes + record["plain_array_bytes"] > 1073741824:
        raise ScientificBlock("Total cache size lower bound exceeds 1GiB", {"completed_bytes": current_bytes,
            "candidate_array_bytes": record["plain_array_bytes"], "limit": 1073741824, "facts": facts})
    arrays.update(cad_id=np.asarray(cad), xyz_sha256=np.asarray(source["xyz_sha256"]),
        contract_sha256=np.asarray(contract_sha), record_json=np.asarray(json.dumps(record, ensure_ascii=False, allow_nan=False)))
    target, temporary = cache / f"{cad}.npz", cache / f"{cad}.tmp.npz"
    require(not target.exists(), "Will not overwrite a completed CAD cache")
    tick = time.monotonic()
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    candidate_bytes = temporary.stat().st_size
    if current_bytes + candidate_bytes > 1073741824:
        temporary.unlink()
        raise ScientificBlock("Actual total cache bytes exceeds 1GiB", {"completed_bytes": current_bytes,
            "candidate_bytes": candidate_bytes, "limit": 1073741824, "facts": facts})
    for key in ("prepared", "gradient_geometry", "interface_inputs"):
        base.match_stat(source[key]["path"], source[key])
    temporary.replace(target)
    record.update(file=target.name, file_identity=identity(target, sha=True))
    record["facts"]["timings"]["cache_write_and_hash_seconds"] = time.monotonic() - tick
    record["elapsed_seconds"] = time.monotonic() - began
    return record


def run(args):
    began = time.monotonic()
    contract = request_contract(args.plan)
    if args.dry_run:
        return {"status": "metadata_preflight_passed", "contract": contract, "arrays_opened": False}
    output, cache = path(args.output), path(contract["cache"])
    require(output != cache, "Run artifacts and immutable caches must use distinct directories")
    config = {"contract": contract, "contract_sha256": fingerprint(contract),
        "scope": "four fixed TRAIN canonical ver inputs only; new balanced ownership", "optimizer_updates": 0}
    summary_path, config_path, manifest_path = output / "summary.json", output / "config.json", cache / "manifest.json"
    if config_path.exists():
        require(read_json(config_path) == config, "Nonempty output config differs; no silent overwrite")
        if summary_path.exists() and read_json(summary_path).get("status") == "completed":
            old = read_json(summary_path)
            require(old["config"]["contract"] == contract, "Completed summary contract differs")
            for cad, record in old["CADs"].items():
                base.recover_cache(cad, cache, config["contract_sha256"], contract["inputs"][cad], record)
            return old
    else:
        require(not output.exists() or not any(output.iterdir()), "Nonempty run directory has no matching config")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        require(manifest["contract"] == contract, "Existing balanced cache manifest contract differs")
    else:
        require(not any(p.name not in [f"{cad}.npz" for cad in base.CAD_ORDER] + [f"{cad}.tmp.npz" for cad in base.CAD_ORDER]
                        for p in cache.iterdir()), "Cache contains unrelated files")
        manifest = {"schema_version": 1, "status": "running", "contract": contract, "CADs": {}}
    completed, blocked, failed, untouched = {}, {}, {}, list(base.CAD_ORDER)
    status = {"status": "running", "config": config, "CAD_order": base.CAD_ORDER, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        for cad in base.CAD_ORDER:
            source = contract["inputs"][cad]
            record = base.recover_cache(cad, cache, config["contract_sha256"], source, manifest["CADs"].get(cad))
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
            require(current_bytes + record["file_identity"]["bytes"] <= 1073741824, "Existing cache bundle exceeds frozen bytes cap")
            completed[cad], manifest["CADs"][cad] = record, record
            untouched.remove(cad)
            manifest.update(status="running", cache_bytes=sum(r["file_identity"]["bytes"] for r in completed.values()))
            write_json(manifest_path, manifest)
        passed = len(completed) == len(base.CAD_ORDER)
        summary = {**status, "status": "completed", "outcome": "feasible" if passed else "infeasible",
            "feasibility_pass": passed, "CADs": completed, "completed_CADs": list(completed), "blocked_CADs": blocked,
            "untouched_CADs": untouched, "failed_CADs": {}, "cache_bytes": sum(r["file_identity"]["bytes"] for r in completed.values()),
            "elapsed_seconds": time.monotonic() - began, "peak_RSS": base.peak_rss(), "labels_read": 0,
            "CUDA_initialized": False, "no_automatic_training": True}
        manifest.update(status="completed", feasibility_pass=passed, outcome=summary["outcome"], blocked_CADs=blocked, untouched_CADs=untouched)
        write_json(manifest_path, manifest)
        write_json(summary_path, summary)
        write_json(output / "status.json", summary)
        write_json(ROOT / "notes/geobc_topology_balanced_feasibility_result.json", summary)
        return summary
    except Exception as exc:
        current = status.get("current_CAD")
        if current:
            failed[current] = {"error": f"{type(exc).__name__}: {exc}"}
        failure = {**status, "status": "failed", "outcome": "invalid_input_or_execution_error",
            "error": f"{type(exc).__name__}: {exc}", "CADs": completed, "completed_CADs": list(completed),
            "blocked_CADs": blocked, "failed_CADs": failed, "untouched_CADs": untouched,
            "elapsed_seconds": time.monotonic() - began, "peak_RSS": base.peak_rss(), "labels_read": 0,
            "CUDA_initialized": False, "completed_files_preserved": True}
        write_json(summary_path, failure)
        write_json(output / "status.json", failure)
        raise


def ownership_fixture():
    """One new ownership fixture: path, star, cross-role edges and isolate."""
    import numpy as np
    from scipy.sparse import csr_matrix
    # Free paths/star are connected only through a different-role support pair;
    # role components remain distinct, with one isolated load-role vertex.
    edges = [(i,i+1) for i in range(5)] + [(6,i) for i in range(7,12)] + [(13,14),(5,13),(11,14)]
    rows, cols = np.asarray(edges,np.int64).T
    W = base.clean_csr(csr_matrix((np.ones(2*len(edges)),(np.r_[rows,cols],np.r_[cols,rows])),shape=(15,15)))
    roles=np.zeros(15,np.uint8);roles[12]=64;roles[13:]=63
    owner,parent,admission,coarse_roles,minimum,facts=balanced_owners(W,roles,np.arange(15,dtype=np.int64),3,4,0)
    require(facts["initial_singleton_count"]==15 and facts["aggregate_count"]==facts["desired"]==4
            and facts["role_components"]["lower_bound"]==4 and facts["merges"]==11,
            "Fixture incorrectly caps initial groups or merges across role components")
    require(owner[12] not in set(owner[np.r_[0:12,13:15]]) and parent[12]==12 and admission[12]==0,
            "Fixture isolated role is not a singleton")
    require(len(set(owner[:6]))==len(set(owner[6:12]))==len(set(owner[13:15]))==1
            and owner[0]!=owner[6] and np.array_equal(coarse_roles[owner],roles),
            "Fixture path/star connected same-role groups failed")
    require(np.array_equal(minimum,np.array([0,6,12,13])) and facts["small_to_large_checked_every_union"],
            "Fixture canonical renumber/small-to-large failed")
    # The new ownership requires no rerun of the frozen old P/R/native tests.
    return {"status":"passed","fixture":"one15vertex path/star/cross-role-components/isolated ownership graph",
        "initial_groups":15,"final_groups":4,"hard_cap":4,"merges":11,
        "checks":["initial singleton cap not applied","minimum mergeable generation-checked heap",
            "actual same-role small-to-large DSU","role-component desired lower bound",
            "canonical descendants","original-edge final BFS parent","isolated singleton"],
        "actual_arrays_read":False,"old_numerical_fixture_repeated":False,"labels_read":0,"CUDA_initialized":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default=str(ROOT / "notes/geobc_topology_balanced_plan.json"))
    parser.add_argument("--output",default=str(ROOT / "runs/geobc_exploration/topology_context_balanced_train4"))
    parser.add_argument("--dry-run",action="store_true",help="Only small metadata/SHA/stat contract")
    parser.add_argument("--self-test",action="store_true",help="Only the new synthetic ownership fixture")
    args=parser.parse_args()
    result=ownership_fixture() if args.self_test else run(args)
    if args.dry_run or args.self_test:
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    else:
        print(json.dumps({k:result[k] for k in ("status","outcome","feasibility_pass","completed_CADs","cache_bytes","elapsed_seconds")},ensure_ascii=False))


if __name__=="__main__":
    main()
