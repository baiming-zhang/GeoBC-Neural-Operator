"""Fixed-checkpoint native replay, query stability and bounded FNO field checks.

This diagnostic reads geometry inputs and committed predictions, never GT.
Default invocation has no IO.  An outer guard owns the absolute deadline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def run():
    from geobc import available_cad_pilot_data as d
    plan = d.read("notes/geobc_continuous_query_followup_plan.json")
    assert d.digest(plan["scientific"]) == plan["scientific_digest"]
    for name, sha in plan["source_pins"].items():
        assert d.sha(name) == sha
    outer = json.loads(__import__("os").environ["GEOBC_FIELD_QUERY_GUARD"])
    assert outer["scientific_digest"] == plan["scientific_digest"]
    assert d.process_snapshot(__import__("os").getppid()) == outer["guard_process"]
    parent = d.read(plan["parent_campaign"])
    assert parent["closed"] and not parent["active"] and parent["within_external_budget"]
    assert all(d.exact_process_exited(r) for r in [parent["guard_process"],
               *[s["child_process"] for s in parent["stages"].values()]])
    output = d.path(plan["output"])
    assert not (output / "execution.json").exists(), "Existing actual execution must be reviewed"
    output.mkdir(parents=True, exist_ok=True)
    execution = {"run_process": d.process_snapshot(__import__("os").getpid()),
                 "guard": outer, "scientific_digest": plan["scientific_digest"]}
    d.write(output / "execution.json", execution)
    receipt = {"raw_CSV_GT_reads": 0, "owned_GT_array_reads": 0,
               "H5_nodal_variable_reads": 0, "training_updates": 0,
               "scaler_fits": 0, "checkpoint_reads": [], "input_arrays": [],
               "committed_prediction_arrays": [], "H5_geometry_member_reads": []}
    result = {"status": "running", "active": True, "closed": False,
              "execution": execution, "receipt": receipt, "cases": {},
              "interpretation": "Bounded learned-field interface diagnostics; not independent off-node FEM accuracy or architecture ceiling"}
    d.write(output / "status.json", result)
    torch = None
    def check():
        if time.time() >= outer["deadline_epoch"] or time.monotonic() >= outer["deadline_monotonic"]:
            raise TimeoutError("Original whole diagnostic deadline reached")
        if torch is not None:
            a = int(torch.cuda.max_memory_allocated()); r = int(torch.cuda.max_memory_reserved())
            result["peak_allocated_bytes"] = a; result["peak_reserved_bytes"] = r
            if max(a, r) > 24 * 2**30:
                raise MemoryError("Fixed allocated/reserved24GiB cap exceeded")
    def array(identity, owner):
        check(); filename = d.path(identity["path"])
        assert d.sha(filename) == identity["sha256"]
        values = np.load(filename, allow_pickle=False)
        assert list(values.shape) == identity["shape"] and np.isfinite(values).all()
        receipt[owner].append(identity["path"])
        return values
    with d.gpu_lock("runs/geobc_exploration/.coordinator.lock"):
        try:
            import numpy as np
            import torch
            import h5py
            from geobc.solver_selection_models import SelectionModel, close_selection_process_group
            from geobc.fno_local_condition_pair import model_class
            from geobc.fno import query_grid
            from geobc.fno_arbitrary_query import tet10_queries, query_fixed_case, physical_query_gradients
            from geobc.solver_selection_data import _stencil
            torch.cuda.init()
            device_index = torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(min(1., 24*2**30 / torch.cuda.get_device_properties(device_index).total_memory), device_index)
            torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            pilot = d.read(plan["parent_summary"])
            inputs = d.read("data/geobc_available_cad_pilot/manifest.json")
            assert inputs["status"] == "completed"
            geometry = {}
            for cad in plan["scientific"]["CADs"]:
                source = inputs["contract"]["raw_sources"][cad]["FieldMesh"]
                assert d.sha(source["path"]) == source["sha256"]
                with h5py.File(d.path(source["path"]), "r") as h:
                    cells = h["cells"][:8].astype(np.int64)
                    ids, inverse = np.unique(cells.ravel(), return_inverse=True)
                    vertices = h["vertices"][ids].astype(np.float32)
                geometry[cad] = (cells, vertices[inverse].reshape(-1, 10, 3))
                receipt["H5_geometry_member_reads"].append({"CAD": cad, "members": ["cells_first8", "vertices_selected_nodes"], "source": source})
            for cid in plan["scientific"]["candidates"]:
                check(); endpoint = pilot["endpoints"][cid]; identity = endpoint["checkpoint"]
                assert endpoint["status"] == "valid" and d.sha(identity["path"]) == identity["sha256"]
                saved = torch.load(d.path(identity["path"]), map_location="cpu", weights_only=False)
                assert saved["step"] == 4800
                receipt["checkpoint_reads"].append(identity)
                torch.manual_seed(2026)
                is_head = cid.startswith("FNO_latent")
                is_pp = cid.startswith("transolver_pp")
                model = model_class(torch)("FNO_latent_query_grid25") if is_head else SelectionModel("transolver_pp" if is_pp else "fno", "cuda", 2026)
                model.load_state_dict(saved["model"], strict=True); del saved
                model = model.cuda().eval().requires_grad_(False)
                for key in plan["scientific"]["cases"]:
                    check(); record = inputs["records"][key]
                    def tensor(name):
                        return torch.as_tensor(array(record["file_identities"][name+"_path"], "input_arrays"), device="cuda")
                    case = {"xyz": tensor("xyz"), "features": tensor("features"),
                            "cloud": tensor("cloud").reshape(1,1024,3), "mask": tensor("mask").reshape(1,1024).bool()}
                    if not is_pp:
                        case.update(grid=tensor("grid").reshape(1,29,64,64,64),
                                    query_indices=tensor("query_indices"), query_weights=tensor("query_weights"))
                    entry = pilot["prediction_inventory"][cid+":"+key]
                    assert d.sha(entry["path"]) == entry["sha256"]
                    prediction = d.read(entry["path"])
                    reference = array(prediction["encoded"], "committed_prediction_arrays")
                    with torch.no_grad():
                        context = model.encoder(case["cloud"], case["mask"])
                        latent = None
                        if not is_pp:
                            if is_head:
                                grid = torch.cat((case["grid"],context[:,:,None,None,None].expand(1,32,64,64,64)),dim=1)
                                latent = model.fno(grid)
                            else:
                                latent = model.forward_grid(case["grid"], context)
                    def query(xyz, chunk=32768):
                        if is_pp:
                            raise ValueError("PP has no fixed-context direct query interface")
                        if is_head:
                            return query_fixed_case(model,case,xyz,context,latent,chunk_size=chunk)
                        inds, weights = _stencil(xyz,64)
                        return query_grid(latent[0],inds,weights,chunk)
                    def seeded_native(seed, subset=False):
                        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                        xyz = case["xyz"][:64] if subset else case["xyz"]
                        features = case["features"][:64] if subset else case["features"]
                        return model.forward_points(xyz,features,context) if is_pp else query(xyz)
                    torch.cuda.synchronize(); began = time.monotonic()
                    with torch.no_grad():
                        native = seeded_native(prediction["eval_rng_seed"])
                    torch.cuda.synchronize(); seconds = time.monotonic()-began
                    ref = torch.as_tensor(reference,device="cuda")
                    replay = float((native-ref).double().norm()/ref.double().norm().clamp_min(1e-12))
                    row = {"native_encoded_replay_relative_L2": replay,
                           "native_replay_pass": replay <= 2e-5, "native_query_seconds": seconds,
                           "native_nodes": len(reference)}
                    if is_pp:
                        with torch.no_grad():
                            same = seeded_native(prediction["eval_rng_seed"])
                            others = [seeded_native(seed) for seed in (2027,2028)]
                            subset = seeded_native(prediction["eval_rng_seed"],True)
                        row.update(same_seed_relative_L2=float((same-native).double().norm()/native.double().norm().clamp_min(1e-12)),
                            other_seed_relative_L2=[float((v-native).double().norm()/native.double().norm().clamp_min(1e-12)) for v in others],
                            native_vs_query_subset_relative_L2=float((subset-native[:64]).double().norm()/native[:64].double().norm().clamp_min(1e-12)),
                            direct_continuous_interface_implemented=False,
                            subset_is_sensitivity_diagnostic_not_an_equivalent_solver=True)
                    else:
                        cad = key.split('/')[0]; cells, physical = geometry[cad]
                        nodes = torch.as_tensor(cells,device="cuda")
                        r = torch.full((len(cells),3),.25,device="cuda",requires_grad=True)
                        geo = tet10_queries(torch.as_tensor(physical,device="cuda"),case["xyz"][nodes],case["features"][nodes],r)
                        J = geo["physical_jacobian"]
                        condition = torch.linalg.cond(J.double()); assert bool(torch.isfinite(condition).all()) and float(condition.max()) < 1e6
                        values = query(geo["xyz"])
                        scaler = inputs["scaler"]
                        def physical_decode(value):
                            v = value.double()*torch.tensor(scaler['std'],device='cuda',dtype=torch.float64)+torch.tensor(scaler['mean'],device='cuda',dtype=torch.float64)
                            if scaler['stress_scale'] is not None:
                                v = torch.cat((v[:,:3],scaler['stress_scale']*torch.sinh(v[:,3:4])),dim=1)
                            return v
                        physical_values = physical_decode(values)
                        grad = physical_query_gradients(physical_values,r,J)
                        assert bool(torch.isfinite(grad).all())
                        with torch.no_grad():
                            small = query(geo['xyz'],1)
                            perm = torch.arange(len(cells)-1,-1,-1,device='cuda')
                            shuffled = query(geo['xyz'][perm])[perm]
                            finite_difference = []
                            base_indices, _ = _stencil(geo['xyz'],64)
                            smooth = torch.ones(len(cells),device='cuda',dtype=torch.bool)
                            for axis in range(3):
                                rp = r.detach().clone(); rm = r.detach().clone()
                                rp[:,axis] += .001; rm[:,axis] -= .001
                                gp = tet10_queries(torch.as_tensor(physical,device='cuda'),case['xyz'][nodes],case['features'][nodes],rp)
                                gm = tet10_queries(torch.as_tensor(physical,device='cuda'),case['xyz'][nodes],case['features'][nodes],rm)
                                ip,_ = _stencil(gp['xyz'],64); im,_ = _stencil(gm['xyz'],64)
                                smooth &= (ip==base_indices).all(-1) & (im==base_indices).all(-1)
                                finite_difference.append((physical_decode(query(gp['xyz']))-physical_decode(query(gm['xyz'])))/.002)
                            fd_r = torch.stack(finite_difference,dim=-1)
                            fd_phys = torch.linalg.solve(J.double().transpose(-1,-2)[:,None],fd_r[...,None]).squeeze(-1)
                            fd_errors = [float((fd_phys[smooth,c]-grad[smooth,c]).norm()/grad[smooth,c].norm().clamp_min(1e-8))
                                         for c in range(4)] if bool(smooth.any()) else None
                        row.update(bounded_TET10_centroid_queries=len(cells),
                            maximum_physical_J_condition=float(condition.max()),
                            finite_physical_gradient=True,
                            physical_gradient_max_abs_by_channel=grad.abs().amax(dim=(0,2)).cpu().tolist(),
                            chunk_max_encoded_difference=float((small-values).abs().max()),
                            permutation_max_encoded_difference=float((shuffled-values).abs().max()),
                            physical_gradient_centered_FD_relative_errors_by_channel=fd_errors,
                            physical_gradient_FD_epsilon_reference=.001,
                            FD_cells_without_stencil_boundary_crossing=int(smooth.sum()),
                            FD_cells_crossing_stencil_boundaries=int((~smooth).sum()),
                            geometry_context_encodes_per_case=1,
                            off_node_independent_FEM_accuracy_measured=False,
                            whole_CAD_locator_certified=False)
                    result["cases"][cid+":"+key] = row
                    d.write(output / "status.json",result); check()
                    del native,ref,case,context,latent
                del model; close_selection_process_group(); torch.cuda.empty_cache()
            result["status"] = "completed"
        except BaseException as error:
            result.update(status="failed", error_class=type(error).__name__, error=str(error))
        finally:
            if torch is not None:
                try:
                    from geobc.solver_selection_models import close_selection_process_group
                    close_selection_process_group(); torch.cuda.synchronize()
                except Exception as error:
                    result.update(status="failed",cleanup_error_class=type(error).__name__)
            result.update(active=False,closed=True,ended_at_utc=d.utc(),
                elapsed_epoch_seconds=time.time()-outer['entry_epoch'],
                elapsed_monotonic_seconds=time.monotonic()-outer['entry_monotonic'])
            d.write(output / "summary.json",result); d.write(output / "status.json",result)
    print(json.dumps({"status":result['status'],"cases":len(result['cases']),
                      "GT_reads":0,"error":result.get('error')}))
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');args=parser.parse_args()
    if args.start:
        raise SystemExit(run())
    print('{"status":"source_only","reads":0,"writes":0}')
