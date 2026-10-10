"""Fixed80 prospective CAD inputs using the unchanged helper's CPU backend; no targets."""
import argparse
import ast
import copy
import csv
import json
import os
import shutil
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding

PLAN='notes/geobc_cpu_inputs80_plan.json'


def freeze():
    assert not d.path(PLAN).exists()
    probe=d.read('notes/geobc_cpu_input_probe_plan.json');folder=d.path(probe['output'])
    binding={k:d.identity(folder/f,hashed=True) for k,f in [('summary','summary.json'),('external','external_terminal.json')]}
    proof=terminal_binding(binding);assert proof['qualified'] and not proof['torch_CUDA_initialized']
    assert all(v['passed'] for v in proof['comparisons'].values())
    source=d.read('notes/geobc_prospective_target_free_input_contract.json')
    assert d.digest(source['scientific'])==source['scientific_digest']
    helper=copy.deepcopy(source['scientific']['helper_contract'])
    assert len(helper['selected_CADs'])==80 and len(helper['case_keys'])==240 and not helper['groups']['train_new160']
    output='data/geobc_prospective_CPU_inputs_v1';helper['owned_future_output']=output
    recipe=copy.deepcopy(source['scientific']['recipe_reference']);recipe['device']='cpu'
    guard='tools/run_geobc_cpu_inputs80.py'
    text=d.path('tools/run_geobc_cpu_input_probe.py').read_text(encoding='utf-8')
    text=text.replace('900','7200').replace('geobc_cpu_input_probe_plan','geobc_cpu_inputs80_plan')
    text=text.replace('GEOBC_CPU_INPUT_GUARD','GEOBC_CPU80_GUARD').replace('tools.geobc_cpu_input_probe','tools.prepare_geobc_cpu_inputs80')
    d.path(guard).write_text(text,encoding='utf-8')
    science=dict(question='Prepare exact fixed TRAIN60/FINAL20 geometry/BC inputs independently while the GPU trains',
        input_binding=d.identity('notes/geobc_prospective_target_free_input_contract.json',hashed=True),
        input_scientific_digest=source['scientific_digest'],helper_contract=helper,recipe=recipe,
        probe_binding=binding,CPU_affinity_cores=4,whole_budget_seconds=7200,worker_count=1,
        target_index=d.identity('results/geobc_prospective_train_preparation/source_cache_index.json',hashed=True),
        required_CADs=80,required_cases=240,CUDA_visible_devices='',GPU_lock_used=False,
        target_policy='All numerical GT forbidden; TRAIN-only input permutation SHA must equal previously prepared target alignment',
        raster_backend='Same float32 stencil/splat mathematics on CPU; knownCAD CPU/GPU differences qualified, not universal bit-identical claim',
        roles='Technical validation branch only; scientific roles TRAIN input180 / FINAL inference input60. No scaler/model/label fit.',
        failure='Whole fixed80 negative if any required CAD fails; preserve prefix and cost, no substitute IDs, automatic replay or budget reset')
    pins=dict(probe['source_pins'])
    for p in ['tools/prepare_geobc_cpu_inputs80.py',guard]:ast.parse(d.path(p).read_text(encoding='utf-8'));pins[p]=d.sha(p)
    plan=dict(scientific=science,scientific_digest=d.digest(science),source_pins=pins,output=output,
              root_review='notes/geobc_cpu_inputs80_root_review.json')
    review=dict(passed=False,whole_SOURCE_reviewed=False,actual_execution_released=False,
        scientific_digest=plan['scientific_digest'],source_pins=pins,actual_preparation_or_array_reads=0)
    d.write(PLAN,plan);d.write(plan['root_review'],review)
    print({'status':'source_prepared_not_released','science':plan['scientific_digest'],'CADs':80,'cases':240})


def run():
    plan=d.read(PLAN);science=plan['scientific'];review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert plan['scientific_digest']==review['scientific_digest']==d.digest(science)
    assert review['source_pins']==plan['source_pins'] and all(d.sha(p)==h for p,h in plan['source_pins'].items())
    proof=terminal_binding(science['probe_binding']);assert proof['qualified'] and not proof['torch_CUDA_initialized']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    outer=json.loads(os.environ['GEOBC_CPU80_GUARD']);assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']);output.mkdir(parents=True,exist_ok=True);assert not (output/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),scientific_digest=plan['scientific_digest'],guard=outer)
    d.write(output/'execution.json',execution)
    started=(time.time(),time.monotonic());torch=None
    contract=science['helper_contract'];roles=contract['scientific_CAD_roles']
    result=dict(status='running',active=True,closed=False,execution=execution,prepared_CADs=0,prepared_cases=0,
        fixed_CAD_denominator=80,fixed_case_denominator=240,GPU_work=0,numerical_GT_reads=0,scaler_fits=0)
    manifest=dict(schema='GeoBC_prospective80_CPU_target_free_inputs_v1',status='running',feasibility_pass=False,
        contract=contract,scientific_digest=plan['scientific_digest'],case_keys=contract['case_keys'],validation_keys=contract['case_keys'],
        train_keys={},scaler=contract['scaler'],records={},CADs={},scientific_CAD_roles=roles,
        helper_validation_is_technical_branch_only=True,fixed_CAD_denominator=80,fixed_case_denominator=240,
        TRAIN_input_case_keys=[k for k in contract['case_keys'] if roles[k.split('/')[0]]=='TRAIN_input_only'],
        FINAL_inference_input_case_keys=[k for k in contract['case_keys'] if roles[k.split('/')[0]]=='FINAL_inference_input_only'])
    class Clock:
        def guard_budget(self):
            if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original7200s CPU fixed80 input deadline')
    clock=Clock();ledger=d.InputLedger(output,contract,clock)
    def persist():
        result.update(elapsed_epoch_seconds=time.time()-started[0],elapsed_monotonic_seconds=time.monotonic()-started[1])
        manifest.update(read_receipt=ledger.value,elapsed_epoch_seconds=result['elapsed_epoch_seconds'])
        d.write(output/'status.json',result);d.write(output/'manifest.json',manifest)
    persist()
    try:
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-4:]));result['actual_CPU_affinity']=sorted(os.sched_getaffinity(0))
        import torch
        from geobc import coverage_comparison_data as original
        torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized()
        assert shutil.disk_usage(output).free>25*2**30,'Insufficient space for fixed240 input arrays'
        pin=science['target_index'];assert d.identity(pin['path'],hashed=True)==pin;targets=d.read(pin['path'])
        assert targets['scaler']==contract['scaler'] and len(targets['records'])==180
        assert not contract['groups']['train_new160'] and len(manifest['TRAIN_input_case_keys'])==180 and len(manifest['FINAL_inference_input_case_keys'])==60
        for cad in contract['selected_CADs']:
            clock.guard_budget();result['current_CAD']=cad;persist()
            filename=contract['raw_sources'][cad]['Field']['path'];ledger.attempt(filename,'header_only')
            with d.path(filename).open(encoding='utf-8-sig',newline='') as stream:columns=next(csv.reader(stream))
            ledger.complete(filename,'CSV_header_only_reads','header_only');assert all(n in columns for n in d.COORDS+d.TARGET_COLUMNS)
            item=original._prepare_CAD(contract,{'scientific':{'recipe':science['recipe']}},cad,output,ledger,lambda _:clock.guard_budget())
            manifest['CADs'][cad]=item;assert item['status']=='completed' and item['feasibility_pass'],'Fixed CAD input unsupported: '+cad
            assert set(item['records'])==set(d.keys([cad]))
            if roles[cad]=='TRAIN_input_only':
                alignment=targets['records'][cad+'/ver']['permutation_identity'];d.stat_matches(alignment)
                assert item['geometry']['h5_row_to_csv']['sha256']==alignment['sha256'],'CPU input/owned TRAIN target permutation differ'
                assert all(targets['records'][cad+'/'+load]['num_nodes']==item['num_nodes'] for load in d.LOADS)
                item['matches_owned_TRAIN_alignment']=True
            else:assert roles[cad]=='FINAL_inference_input_only'
            for key,record in item['records'].items():
                d._record_is_target_free(record,contract);record['scientific_role']=roles[cad]
                manifest['records'][key]=record
            result['prepared_CADs']+=1;result['prepared_cases']+=3;persist()
        clock.guard_budget();assert set(manifest['records'])==set(contract['case_keys'])
        assert all(ledger.value[n]==0 for n in d.ZERO_COUNTERS) and not torch.cuda.is_initialized()
        manifest.update(status='completed',feasibility_pass=True,all_fixed80=True)
        result.update(status='completed',all_fixed80=True,current_CAD=None)
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_replay=False)
        manifest.update(status='input_negative',feasibility_pass=False,error=str(exc),all_fixed80=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-started[0],elapsed_monotonic_seconds=time.monotonic()-started[1],
            torch_CUDA_initialized=torch.cuda.is_initialized() if torch else False,read_receipt=ledger.value)
        d.write(output/'summary.json',result);persist()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--start',action='store_true')
    args=parser.parse_args();assert not(args.freeze and args.start)
    if args.freeze:freeze()
    if args.start:run()
