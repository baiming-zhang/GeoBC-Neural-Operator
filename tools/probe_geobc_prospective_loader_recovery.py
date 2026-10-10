"""Two actual TRAIN cases and all protected access gates; CPU only, no model."""
import argparse
import ast
import json
import os
import sys
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding

PLAN='notes/geobc_prospective_loader_recovery_plan.json'


def freeze():
    assert not d.path(PLAN).exists()
    parent=d.read('notes/geobc_prospective_dataset_plan.json');folder=d.path(parent['output'])
    binding={k:d.identity(folder/f,hashed=True) for k,f in [('summary','summary.json'),('external','external_terminal.json')]}
    proof=terminal_binding(binding);assert proof['TRAIN_target_records']==300 and proof['protected_input_only_records']==120 and proof['array_GT_model_GPU_reads']==0
    manifest=d.read(folder/'manifest.json');old=sorted(manifest['train_keys']['40'])[0]
    additional=sorted(set(manifest['train_keys']['100'])-set(manifest['train_keys']['40']))[0]
    guard='tools/run_geobc_loader_probe_guard.py'
    text=d.path('tools/run_geobc_dataset_assembly_guard.py').read_text(encoding='utf-8')
    text=text.replace('180','300').replace('geobc_prospective_dataset_plan','geobc_prospective_loader_probe_plan')
    text=text.replace('GEOBC_DATASET_ASSEMBLY_GUARD','GEOBC_LOADER_PROBE_GUARD').replace('tools.prepare_geobc_prospective_dataset','tools.probe_geobc_prospective_loader')
    text=text.replace('prospective metadata/stat assembly','prospective CPU TRAIN loader probe')
    d.path(guard).write_text(text,encoding='utf-8')
    science=dict(question='Does the explicit100-CAD loader preserve existing inputs/labels and reject protected numeric targets?',
        parent_binding=binding,parent_digest=parent['scientific_digest'],manifest_identity=proof['manifest_identity'],
        cases=[old,additional],selection='Lexicographically first oldTRAIN and newTRAIN case, before inspecting numerical arrays',
        reference_legacy_index=parent['scientific']['source_bindings']['legacy_TRAIN'],
        protected_requests='All60DEV+60FINAL with_targetsTrue must reject before arrays/library imports; all180additionalTRAIN requests under40TRAINscope must reject too',
        numerical_scope='Onlytwo ownedTRAIN cases and one oldTRAIN legacy reference; zeroCSV/H5/scaler/DEV/FINAL/model/GPU work',
        whole_budget_seconds=300,CPU_cores=4,CUDA_VISIBLE_DEVICES='',GPU_lock_used=False,
        encoding_tolerance=dict(atol=2e-6,rtol=2e-6),coverage='Two-case runtime compatibility, not all300 numerical bodies or any model gradient/accuracy')
    pins=dict(parent['source_pins'])
    for p in ['tools/probe_geobc_prospective_loader.py',guard]:ast.parse(d.path(p).read_text(encoding='utf-8'));pins[p]=d.sha(p)
    plan=dict(scientific=science,scientific_digest=d.digest(science),source_pins=pins,
        output='runs/geobc_exploration/prospective100_loader_CPU_probe_v1',root_review='notes/geobc_prospective_loader_probe_root_review.json')
    d.write(PLAN,plan);d.write(plan['root_review'],dict(passed=False,whole_SOURCE_reviewed=False,actual_execution_released=False,
        scientific_digest=plan['scientific_digest'],source_pins=pins,actual_execution=0))
    print({'status':'source_frozen_actual0','science':plan['scientific_digest'],'cases':science['cases']})


def run():
    plan=d.read(PLAN);science=plan['scientific'];review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(science) and review['source_pins']==plan['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parent=terminal_binding(science['parent_binding']);assert parent['execution']['scientific_digest']==science['parent_digest']
    pin=science['manifest_identity'];assert d.identity(pin['path'],hashed=True)==pin;manifest=d.read(pin['path'])
    assert manifest['records_digest']==d.digest(manifest['records']) and not manifest['FINAL_numeric_targets_released']
    outer=json.loads(os.environ['GEOBC_LOADER_PROBE_GUARD']);assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']);output.mkdir(parents=True,exist_ok=True);assert not (output/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json',execution);began=(time.time(),time.monotonic());torch=None
    result=dict(status='running',closed=False,active=True,execution=execution,checks={},actual_TRAIN_array_reads=0,
        actual_input_array_reads=0,DEV_FINAL_target_array_reads=0,source_CSV_H5_reads=0,scaler_fits=0,model_GPU_work=0)
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original300s CPU loader deadline')
    try:
        from geobc.prospective_dataset_v2 import load_case,resolve_case
        modules_before=set(sys.modules);protected=manifest['development_keys']+manifest['FINAL_inference_keys']
        rejected=[]
        for key in protected:
            guard()
            try:load_case(manifest,key,with_targets=True,device='cpu',for_fno=True)
            except PermissionError:rejected.append(key)
            else:raise AssertionError('Protected numerical target request did not reject')
        outside=sorted(set(manifest['train_keys']['100'])-set(manifest['train_keys']['40']));rejected40=[]
        for key in outside:
            try:load_case(manifest,key,with_targets=True,train_CAD_count=40,device='cpu',for_fno=True)
            except PermissionError:rejected40.append(key)
            else:raise AssertionError('NewTRAIN escaped the TRAIN40 coverage scope')
        assert len(rejected)==120 and len(rejected40)==180
        imported=(set(sys.modules)-modules_before)&{'numpy','torch'};assert not imported
        assert set(manifest['case_keys'])==set(manifest['records']) and len(manifest['case_keys'])==420
        for key in manifest['case_keys']:resolve_case(manifest,key,with_targets=False,for_fno=True)
        result['checks'].update(all120_protected_requests_rejected_before_array_access=True,
            all180_newTRAIN_rejected_under40scope=True,all420_input_records_resolve=True,
            library_imports_added_by_protected_requests=sorted(imported))
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[-4:]));result['CPU_affinity']=sorted(os.sched_getaffinity(0))
        import numpy as np
        import torch
        from geobc.benchmark import FieldScaler
        from geobc.coverage_comparison_data import load_case as legacy_load
        torch.set_num_threads(4);torch.set_num_interop_threads(1);assert not torch.cuda.is_initialized()
        scaler=FieldScaler(**manifest['scaler']);result['case_checks']={}
        def opened(event,key,name,item):
            guard()
            if event=='opened':
                result['actual_TRAIN_array_reads' if name in ('target','physical') else 'actual_input_array_reads']+=1
        for i,key in enumerate(science['cases']):
            guard();case=load_case(manifest,key,with_targets=True,device='cpu',for_fno=True,on_read=opened)
            assert case['scientific_group']=='TRAIN' and case['target'].shape==case['physical'].shape
            encoded=scaler.encode(case['physical']);target=case['target'].numpy()
            assert np.allclose(encoded,target,**science['encoding_tolerance'])
            row=dict(nodes=len(case['xyz']),all_members_finite=True,target_encode_matches_physical=True,
                maximum_encoded_target_absolute_difference=float(np.abs(encoded-target).max()),read_receipt=case['read_receipt'])
            if i==0:
                pin=science['reference_legacy_index'];assert d.identity(pin['path'],hashed=True)==pin
                reference=legacy_load(d.read(pin['path']),key,with_targets=True,device='cpu',for_fno=True)
                result['actual_TRAIN_array_reads']+=reference['read_receipt']['cached_TRAIN_target_array_reads']
                result['actual_input_array_reads']+=reference['read_receipt']['input_array_reads']
                equality={name:torch.equal(case[name],reference[name]) for name in ('xyz','features','cloud','mask','grid','query_indices','query_weights','target')}
                equality['physical']=bool(np.array_equal(case['physical'],reference['physical']));assert all(equality.values())
                row['original_legacy_loader_exact_member_equality']=equality;del reference
            result['case_checks'][key]=row;del case,encoded,target
        guard();assert result['actual_TRAIN_array_reads']==6 and result['actual_input_array_reads']==21 and not torch.cuda.is_initialized()
        result.update(status='completed',qualified=True,all300_numeric_bodies_checked=False,
            model_gradient_or_accuracy_qualification=False,new_training_started=False,FINAL_targets_released=False)
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),qualified=False,automatic_replay=False);raise
    finally:
        result.update(closed=True,active=False,elapsed_epoch_seconds=time.time()-began[0],elapsed_monotonic_seconds=time.monotonic()-began[1],
            torch_CUDA_initialized=torch.cuda.is_initialized() if torch else False)
        d.write(output/'summary.json',result);d.write(output/'status.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--start',action='store_true')
    args=parser.parse_args();assert not(args.freeze and args.start)
    if args.freeze:freeze()
    if args.start:run()
