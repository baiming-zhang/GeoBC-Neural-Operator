"""One bounded native-CUDA interface check; no targets or optimizer updates."""
import argparse
import os
from pathlib import Path
import time

ENTRY_EPOCH, ENTRY_MONO = time.time(), time.monotonic()
PLAN = 'notes/geobc_trained_operator_hash_queries_plan.json'


def metadata():
    from geobc import available_cad_pilot_data as d
    from geobc.operator_hash_readout_contract_v5 import check_identity
    p = d.read(PLAN); r = d.read(p['root_review'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['scientific_digest'] == p['scientific_digest'] == d.digest(p['scientific'])
    assert r['source_pins'] == p['source_pins']
    assert all(d.sha(k) == v for k, v in p['source_pins'].items())
    s = p['scientific']
    for name, binding in r['upstream_bindings'].items():
        for item in binding.values(): check_identity(item)
        summary = d.read(binding['summary']['path']); external = d.read(binding['external']['path'])
        assert summary['status'] == 'completed' and summary['closed'] and not summary['active']
        assert external['closed'] and not external['active'] and external['exit_code'] == 0
        assert external['within_external_budget'] and all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
        if name == 'reader': assert summary['protocol_qualified'] and summary['committed_finite_predictions'] == 240 and summary['owned_GT_array_reads'] == 60
    manifest = d.read(check_identity(s['dataset_identity']))
    assert s['key'] == manifest['train_keys']['100'][0]
    assert manifest['case_roles'][s['key']] == 'TRAIN'
    check_identity(s['checkpoint_identity'])
    assert s['memory_limit_bytes'] == 30500000000 and s['whole_budget_seconds'] == 180
    assert s['targets_allowed'] is False and s['training_updates'] == 0
    return p, r, manifest


def child():
    from geobc import available_cad_pilot_data as d
    p, r, manifest = metadata(); s = p['scientific']; output = d.path(p['output'])
    outer = d.read(output/'external_launch.json')
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    result = dict(status='running', active=True, closed=False, scientific_digest=p['scientific_digest'],
                  process=d.process_snapshot(os.getpid()), checkpoint_reads=0, numerical_GT_reads=0,
                  training_updates=0, FINAL_reads=0, raw_CSV_H5_reads=0, scaler_fits=0,
                  independent_offnode_FEM_accuracy_measured=False)
    torch = None; lock = None

    def guard():
        assert min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic()) > 0
        if torch is not None and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            result['peak_sampled_device_bytes'] = max(result.get('peak_sampled_device_bytes',0), total-free)
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            assert max(result[k] for k in ('peak_sampled_device_bytes','peak_allocated_bytes','peak_reserved_bytes')) <= s['memory_limit_bytes']

    try:
        import fcntl
        lock = d.path('runs/geobc_exploration/.coordinator.lock').open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch as torch_module
        torch = torch_module
        from geobc.prospective_dataset_v2 import load_case
        from geobc.operator_hash_metadata_v6 import build_solver, encode_case
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats(); guard()
        saved = torch.load(d.path(s['checkpoint_identity']['path']),map_location='cpu',weights_only=False)
        result['checkpoint_reads'] += 1
        assert saved['family'] == 'FNO_GLOBAL_HASH' and saved['step'] == saved['cursor'] == 4800
        assert saved['scientific_digest'] == s['training_scientific_digest']
        assert saved['dataset_identity'] == s['dataset_identity'] and saved['scaler'] == manifest['scaler']
        model = build_solver('FNO_GLOBAL_HASH',seed=2026); model.load_state_dict(saved['model'],strict=True); del saved
        model.eval(); guard()
        case = load_case(manifest,s['key'],with_targets=False,train_CAD_count=100,device='cuda',for_fno=True)
        assert not {'target','physical'} & set(case)
        result['input_read_receipt'] = case['read_receipt']
        with torch.no_grad():
            context, latent = encode_case(model,case)
            xyz, features = case['xyz'][:257], case['features'][:257]
            direct = model.query(case,0,257,context,latent)
            arbitrary = model.query_points(xyz,features,context,latent,chunk_size=257)
            chunks = model.query_points(xyz,features,context,latent,chunk_size=31)
            permutation = torch.arange(256,-1,-1,device='cuda')
            shuffled = model.query_points(xyz[permutation],features[permutation],context,latent,chunk_size=17)
            comparisons = dict(node_vs_recomputed_stencil=(direct,arbitrary),chunk=(arbitrary,chunks),permutation=(arbitrary[permutation],shuffled))
            result['query_equalities'] = {}
            for name,(a,b) in comparisons.items():
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                error = float((a-b).abs().max())
                passed = bool(torch.allclose(a,b,atol=s['FP16_equality_atol'],rtol=s['FP16_equality_rtol']))
                result['query_equalities'][name] = dict(passed=passed,max_abs_encoded4_error=error)
                assert passed, name+' native numerical mismatch'
            # Declared constant feature extension, not an FEM target or BC reconstruction.
            generator = torch.Generator(device='cuda').manual_seed(2026)
            points = .2 + .6*torch.rand((64,3),generator=generator,device='cuda')
            supplied = features[:64].clone()
        points.requires_grad_(True); supplied.requires_grad_(True)
        c = context.detach().requires_grad_(True); l = latent.detach().requires_grad_(True)
        values = model.query_points(points,supplied,c,l,chunk_size=64)
        gradients = torch.autograd.grad(values.sum(),(points,supplied,c,l))
        names = ('xyz','supplied_native25','geometry_context','operator_latent')
        result['native_input_gradient_L2'] = {n:float(g.norm()) for n,g in zip(names,gradients)}
        assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)
        result['perturbations'] = []
        with torch.no_grad():
            baseline = values.detach()
            for epsilon in s['coordinate_perturbations']:
                shifted = points.detach().clone(); shifted[:,0] += epsilon
                changed = model.query_points(shifted,supplied.detach(),context,latent,chunk_size=64)
                assert torch.isfinite(changed).all()
                result['perturbations'].append(dict(epsilon=epsilon,max_abs_encoded4_change=float((changed-baseline).abs().max())))
        guard(); torch.cuda.synchronize()
        result.update(status='completed',native_TCNN_interface_verified=True,
                      offnode_feature_extension='Held constant supplied native25; only a numerical interface check',
                      field_accuracy_claim=False,all_original_nodes_in_accuracy_check=False)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error_message=str(error))
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONO)
        d.write(output/'summary.json',result)
        if lock is not None: lock.close()


def start():
    from geobc import available_cad_pilot_data as d
    import subprocess,signal,sys
    p, r, m = metadata(); output=d.path(p['output']);output.mkdir(parents=True,exist_ok=True)
    launch=output/'external_launch.json'; assert not launch.exists()
    budget=p['scientific']['whole_budget_seconds']
    receipt=dict(guard_process=d.process_snapshot(os.getpid()),scientific_digest=p['scientific_digest'],
                 entry_epoch=ENTRY_EPOCH,entry_monotonic=ENTRY_MONO,deadline_epoch=ENTRY_EPOCH+budget,
                 deadline_monotonic=ENTRY_MONO+budget,active=True,closed=False,automatic_replay=False,
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),guard_error_type=None)
    d.write(launch,receipt); process=None;timeout=False
    try:
        with (output/'process.log').open('x',encoding='utf-8') as log:
            argv=[sys.executable,'-u','-m','tools.check_trained_operator_hash_queries','--child']
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=d.ROOT)
            for _ in range(50):
                identity=d.process_snapshot(process.pid)
                if identity and identity['argv']==argv: break
                time.sleep(.01)
            assert identity['argv']==argv and identity['ppid']==receipt['guard_process']['pid']
            receipt['child_process']=identity;d.write(launch,receipt)
            while process.poll() is None:
                remaining=min(receipt['deadline_epoch']-time.time(),receipt['deadline_monotonic']-time.monotonic())
                if remaining<=0: timeout=True; break
                try:process.wait(timeout=min(remaining,1))
                except subprocess.TimeoutExpired:pass
    except BaseException as error:
        receipt['guard_error_type']=type(error).__name__
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        receipt.update(active=False,closed=True,exit_code=None if process is None else process.returncode,external_timeout=timeout,
                       elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONO)
        receipt['within_external_budget']=receipt['guard_error_type'] is None and receipt['exit_code']==0 and not timeout and max(receipt['elapsed_epoch_seconds'],receipt['elapsed_monotonic_seconds'])<=budget
        d.write(output/'external_terminal.json',receipt);d.write(launch,receipt)
    print({'status':'terminal','exit_code':receipt['exit_code'],'within_budget':receipt['within_external_budget']})
    return 0 if receipt['within_external_budget'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--child',action='store_true')
    arguments=parser.parse_args()
    if arguments.start:raise SystemExit(start())
    elif arguments.child:child()
