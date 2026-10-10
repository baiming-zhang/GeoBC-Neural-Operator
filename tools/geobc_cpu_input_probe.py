"""One known CAD, original input-only producer on CPU, compared with saved GPU inputs."""
import argparse
import ast
import copy
import csv
import os
import time
from geobc import available_cad_pilot_data as d

PLAN='notes/geobc_cpu_input_probe_plan.json'


def freeze():
    assert not d.path(PLAN).exists()
    basis=d.read('data/geobc_available_cad_pilot/manifest.json')
    assert basis['status']=='completed' and basis['feasibility_pass']
    cad=sorted(basis['contract']['selected_CADs'])[0]
    contract=copy.deepcopy(basis['contract'])
    contract.update(selected_CADs=[cad],case_keys=d.keys([cad]),groups={'train_new160':[],'validation20':[cad]})
    contract['raw_sources']={cad:contract['raw_sources'][cad]}
    recipe=copy.deepcopy(d.read('notes/geobc_available_cad_pilot_plan.json')['recipe_reference'])
    recipe['device']='cpu'
    guard='tools/run_geobc_cpu_input_probe.py'
    source=d.path('tools/run_geobc_common_readout_guard.py').read_text(encoding='utf-8')
    source=source.replace('1800','900').replace('geobc_common_grid_readout_plan','geobc_cpu_input_probe_plan')
    source=source.replace('actual_readout_released','actual_execution_released').replace('GEOBC_COMMON_READOUT_GUARD','GEOBC_CPU_INPUT_GUARD')
    source=source.replace('tools.evaluate_geobc_common_grid','tools.geobc_cpu_input_probe')
    source=source.replace("    child=None;code=None;timeout=False", "    env['CUDA_VISIBLE_DEVICES']=''\n    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]='4'\n    child=None;code=None;timeout=False")
    d.path(guard).write_text(source,encoding='utf-8')
    science=dict(question='Can original target-free input mathematics run on CPU with bounded differences from saved GPU inputs?',
        cad=cad,helper_contract=contract,recipe=recipe,whole_budget_seconds=900,CPU_affinity_cores=4,
        reference_input_manifest=d.identity('data/geobc_available_cad_pilot/manifest.json',hashed=True),
        checks='xyz/cloud/features/mask/query_indices exact; query_weights abs2e-6; grid occupancy exact, xyz abs2e-6, feature-raster per-channel abs1e-4*(1+reference peak)',
        scope='Known development input-only, no numeric target/scaler/strain/H5nodal read; no new80 preparation until qualified',
        CUDA_visible_devices='',GPU_lock_used=False)
    pins=dict(d.read('notes/geobc_prospective_target_free_input_contract.json')['scientific']['source_pins'])
    for p in ['tools/geobc_cpu_input_probe.py',guard,'geobc/solver_selection_data.py','geobc/conditioned_field.py']:
        pins[p]=d.sha(p)
    for p in ['tools/geobc_cpu_input_probe.py',guard]:ast.parse(d.path(p).read_text(encoding='utf-8'))
    plan=dict(scientific=science,scientific_digest=d.digest(science),source_pins=pins,
        output='data/geobc_CPU_input_probe_v1',root_review='notes/geobc_cpu_input_probe_root_review.json')
    review=dict(passed=False,whole_SOURCE_reviewed=False,actual_execution_released=False,
        scientific_digest=plan['scientific_digest'],source_pins=pins,actual_preparation_or_array_reads=0)
    d.write(PLAN,plan);d.write(plan['root_review'],review)
    print({'status':'source_prepared_not_released','cad':cad,'science':plan['scientific_digest']})


def run():
    plan=d.read(PLAN);science=plan['scientific'];review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert d.digest(science)==plan['scientific_digest']==review['scientific_digest']
    assert review['source_pins']==plan['source_pins'] and all(d.sha(p)==h for p,h in plan['source_pins'].items())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    outer=__import__('json').loads(os.environ['GEOBC_CPU_INPUT_GUARD'])
    assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']);output.mkdir(parents=True,exist_ok=True);assert not (output/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),scientific_digest=plan['scientific_digest'],guard=outer)
    d.write(output/'execution.json',execution)
    started=(time.time(),time.monotonic());torch=None
    result=dict(status='running',active=True,closed=False,execution=execution,comparisons={},GPU_work=0,
                numeric_GT_reads=0,scaler_fits=0)
    class Clock:
        def guard_budget(self):
            if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original900s CPU probe budget')
    clock=Clock();ledger=d.InputLedger(output,science['helper_contract'],clock)
    try:
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-4:]))
        result['actual_CPU_affinity']=sorted(os.sched_getaffinity(0))
        import numpy as np
        import torch
        from geobc import coverage_comparison_data as original
        torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized()
        pin=science['reference_input_manifest'];assert d.identity(pin['path'],hashed=True)==pin
        reference=d.read(pin['path']);contract=science['helper_contract'];assert not contract['groups']['train_new160']
        cad=science['cad'];file=contract['raw_sources'][cad]['Field']['path']
        ledger.attempt(file,'header_only')
        with d.path(file).open(encoding='utf-8-sig',newline='') as stream:columns=next(csv.reader(stream))
        ledger.complete(file,'CSV_header_only_reads','header_only')
        assert all(n in columns for n in d.COORDS+d.TARGET_COLUMNS)
        item=original._prepare_CAD(contract,{'scientific':{'recipe':science['recipe']}},cad,output,ledger,lambda _:clock.guard_budget())
        assert item['status']=='completed' and item['feasibility_pass']
        compared=set()
        for key,record in item['records'].items():
            d._record_is_target_free(record,contract)
            old=reference['records'][key]
            for name in d.MEMBERS:
                clock.guard_budget();a=record['file_identities'][name];b=old['file_identities'][name]
                pair=(a['path'],b['path'])
                if pair in compared:continue
                compared.add(pair);d.stat_matches(b);assert d.sha(b['path'])==b['sha256']
                left=np.load(d.path(a['path']),mmap_mode='r',allow_pickle=False)
                right=np.load(d.path(b['path']),mmap_mode='r',allow_pickle=False)
                assert left.shape==right.shape and left.dtype==right.dtype
                exact=np.array_equal(left,right);maximum=0.;passed=exact
                if name=='grid_path':
                    assert np.isfinite(left).all() and np.isfinite(right).all()
                    diffs=np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)),axis=(1,2,3))
                    peaks=np.max(np.abs(right.astype(np.float64)),axis=(1,2,3))
                    passed=np.array_equal(left[0],right[0]) and bool((diffs[1:4]<=2e-6).all()) and bool((diffs[4:]<=1e-4*(1+peaks[4:])).all())
                    maximum=float(diffs.max());details=dict(channel_max_absolute_difference=diffs.tolist(),reference_channel_peak=peaks.tolist())
                elif name=='query_weights_path':
                    maximum=float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))));passed=maximum<=2e-6;details={}
                else:details={}
                result['comparisons'][key+':'+name]=dict(passed=bool(passed),exact=bool(exact),max_absolute_difference=maximum,CPU=a,reference_GPU=b,**details)
                assert passed,'CPU/GPU input mathematics differ beyond the frozen tolerance: '+key+':'+name
                del left,right
        assert all(ledger.value[n]==0 for n in d.ZERO_COUNTERS) and not torch.cuda.is_initialized()
        result.update(status='completed',qualified=True,original_helper_cpu_device=True,prepared_cases=3,
            compare_unique_pairs=len(compared),read_receipt=ledger.value,CPU_CAD_manifest=d.identity(output/'geometry'/cad/'manifest.json',hashed=True),
            interpretation='One known CAD/all3 loads supports CPU backend reuse, not bit equality of all80 or a model accuracy result')
    except BaseException as exc:
        result.update(status='failed',qualified=False,error_type=type(exc).__name__,error=str(exc),automatic_replay=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-started[0],elapsed_monotonic_seconds=time.monotonic()-started[1],
                      torch_CUDA_initialized=torch.cuda.is_initialized() if torch else False,read_receipt=ledger.value)
        d.write(output/'summary.json',result);d.write(output/'status.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--start',action='store_true')
    args=parser.parse_args();assert not(args.freeze and args.start)
    if args.freeze:freeze()
    if args.start:run()
