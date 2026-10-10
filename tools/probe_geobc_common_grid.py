"""One real TRAIN case/full R64 gradient probe, zero optimizer updates."""
import argparse
import json
import os
import time
from pathlib import Path
from geobc import available_cad_pilot_data as d

PLAN = 'notes/geobc_common_grid_probe_plan.json'


def run():
    start_epoch = time.time(); start_mono = time.monotonic()
    plan = d.read(PLAN); science = plan['scientific']; root = d.read(plan['root_review'])
    assert d.digest(science) == plan['scientific_digest'] == root['scientific_digest']
    assert root['passed'] and root['actual_probe_released'] and root['source_pins'] == plan['source_pins']
    assert all(d.sha(p) == h for p,h in plan['source_pins'].items())
    outer = json.loads(os.environ['GEOBC_COMMON_PROBE_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    output = d.path(plan['output']); output.mkdir(parents=True,exist_ok=True)
    assert not (output/'execution.json').exists(), 'Prior probe exists; no replay'
    execution = {'run_process':d.process_snapshot(os.getpid()),'guard':outer,
                 'scientific_digest':plan['scientific_digest']}
    d.write(output/'execution.json',execution)
    result = {'status':'running','active':True,'closed':False,'execution':execution,
              'arms':{},'optimizer_updates':0,'raw_GT_reads':0,'DEV_reads':0,'scaler_fits':0}
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Original 900-second probe budget expired')
    lock = None
    try:
        # Terminal assertions and exact identities, then the same nonblocking
        # GPU lock used by all prior campaigns, before any CUDA or array load.
        for folder in science['predecessors']:
            ext = d.read(folder+'/external_terminal.json')
            summary = d.read(folder+'/summary.json')
            assert ext['closed'] and not ext['active'] and ext['exit_code'] == 0 and summary['closed']
            assert all(d.exact_process_exited(ext[k]) for k in ('child_process','guard_process'))
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        from geobc import coverage_comparison_data as data
        from geobc.common_grid_solver import CommonGridSolver, full_case_backward, parameter_gradients
        from geobc.transformers import close_owned_process_group
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        device = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        manifest = data.read_bootstrap_TRAIN40(science['parent_plan'])
        assert science['TRAIN_case'] in manifest['train_keys']['40']
        # Input-only loader, then only the known TRAIN encoded target. No
        # physical labels are necessary for this gradient/resource question.
        case = data.load_case(manifest,science['TRAIN_case'],device='cuda',for_fno=True)
        target = manifest['records'][science['TRAIN_case']]['file_identities']['target_path']
        data.stat_matches(target)
        case['target'] = torch.as_tensor(np.array(np.load(data.path(target['path']),allow_pickle=False),copy=True),device='cuda')
        result['input_read_receipt'] = case['_actual_read_receipt']
        result['cached_TRAIN_encoded_target_reads'] = 1
        shared = None
        for arm in science['arms']:
            guard(); torch.manual_seed(science['seed']); torch.cuda.manual_seed_all(science['seed'])
            row = {'status':'running'}; result['arms'][arm]=row; d.write(output/'status.json',result)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            model = CommonGridSolver(arm,science['seed']).to('cuda').train()
            row['counts'] = model.counts(); row['shared_init_sha256'] = model.shared_identity()
            if shared is None: shared = row['shared_init_sha256']
            assert shared == row['shared_init_sha256']
            torch.cuda.synchronize(); compute = time.monotonic()
            loss, leaves = full_case_backward(model,case,guard)
            torch.cuda.synchronize()
            row.update(loss=loss,full_case_forward_backward_seconds=time.monotonic()-compute,
                       gradients=parameter_gradients(model),leaf_gradients=leaves,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       total_arm_seconds=time.monotonic()-began,original_nodes=len(case['xyz']),grid_tokens=262144)
            row['gradient_pass'] = all(not g['missing'] and g['finite'] and g['L2']>0 for g in row['gradients'].values()) and all(v>0 for v in leaves.values())
            row['resource_pass'] = row['peak_reserved_bytes']<=24*2**30 and row['peak_allocated_bytes']<=24*2**30
            assert row['gradient_pass'] and row['resource_pass']
            row['status']='completed'; d.write(output/'status.json',result)
            del model; close_owned_process_group(); torch.cuda.empty_cache()
        totals = [row['counts']['total'] for row in result['arms'].values()]
        result.update(status='completed',same_shared_init=True,capacity_relative_gap=(max(totals)-min(totals))/min(totals))
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_replay=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-start_epoch,
                      elapsed_monotonic_seconds=time.monotonic()-start_mono)
        d.write(output/'summary.json',result); d.write(output/'status.json',result)
        if lock is not None: lock.close()
    print(json.dumps({'status':result['status'],'seconds':result['elapsed_epoch_seconds']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
