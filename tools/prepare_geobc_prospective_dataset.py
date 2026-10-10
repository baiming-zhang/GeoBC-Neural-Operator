"""Assemble existing100/20/20 records, metadata/stat only; no numeric-array reads."""
import argparse
import ast
import copy
import json
import os
import time
from geobc import available_cad_pilot_data as d
from geobc.prospective_dataset import SCHEMA

PLAN='notes/geobc_prospective_dataset_plan.json'
MEMBERS=('xyz_path','features_path','cloud_path','mask_path','grid_path','query_indices_path','query_weights_path')


def freeze():
    assert not d.path(PLAN).exists()
    old=d.read('notes/geobc_coverage_comparison_plan.json');inputs=d.read('notes/geobc_cpu_inputs80_plan.json')
    paths=dict(split='notes/geobc_prospective_cad_split_v1.json',
        legacy_TRAIN=d.path(old['data_cache'])/'train40_bootstrap.json',
        new_input_index='results/geobc_prospective_input_preparation/source_cache_index.json',
        input_acceptance='results/geobc_prospective_input_preparation/manifest.json',
        new_target_index='results/geobc_prospective_train_preparation/source_cache_index.json',
        target_acceptance='results/geobc_prospective_train_preparation/manifest.json',
        new_input_manifest=d.path(inputs['output'])/'manifest.json',DEV_inputs='data/geobc_available_cad_pilot/manifest.json')
    science=dict(question='Provide one explicit immutableTRAIN100/knownDEV20/reservedFINAL20 input/target interface for subsequent solvers',
        source_bindings={name:d.identity(path,hashed=True) for name,path in paths.items()},
        phase='CPU metadata/stat assembly only; no array bodies, labels, model, GPU or scaler fit',
        target_scope='OnlyTRAIN300 attach existing owned targets; DEV60/FINAL60 keep exactly seven input members',
        whole_budget_seconds=180,original_200_protocol_reused=False,
        future_training_and_FINAL_evaluation_released=False)
    guard='tools/run_geobc_dataset_assembly_guard.py'
    source=d.path('tools/run_geobc_cpu_inputs80.py').read_text(encoding='utf-8')
    source=source.replace('7200','180').replace('geobc_cpu_inputs80_plan','geobc_prospective_dataset_plan')
    source=source.replace('GEOBC_CPU80_GUARD','GEOBC_DATASET_ASSEMBLY_GUARD').replace('tools.prepare_geobc_cpu_inputs80','tools.prepare_geobc_prospective_dataset')
    source=source.replace('fixed-checkpoint field diagnostic','prospective metadata/stat assembly')
    d.path(guard).write_text(source,encoding='utf-8')
    pins={}
    for name in ['tools/prepare_geobc_prospective_dataset.py',guard,'geobc/prospective_dataset.py',
                 'geobc/available_cad_pilot_data.py','geobc/coverage_comparison_data.py']:
        ast.parse(d.path(name).read_text(encoding='utf-8'));pins[name]=d.sha(name)
    plan=dict(scientific=science,scientific_digest=d.digest(science),source_pins=pins,
        output='data/geobc_prospective_dataset_v1',root_review='notes/geobc_prospective_dataset_root_review.json')
    d.write(PLAN,plan);d.write(plan['root_review'],dict(passed=False,whole_SOURCE_reviewed=False,
        actual_execution_released=False,scientific_digest=plan['scientific_digest'],source_pins=pins,actual_execution=0))
    print({'status':'source_frozen_actual0','science':plan['scientific_digest']})


def run():
    plan=d.read(PLAN);science=plan['scientific'];review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(science) and review['source_pins']==plan['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    outer=json.loads(os.environ['GEOBC_DATASET_ASSEMBLY_GUARD']);assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']);assert not (output/'execution.json').exists();began=(time.time(),time.monotonic())
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json',execution)
    result=dict(status='running',closed=False,active=True,execution=execution,array_GT_model_GPU_reads=0,scaler_fits=0,
        owned_array_stat_checks=0,new_training_started=False,FINAL_numerical_labels_released=False)
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original180s assembly deadline')
    try:
        saved={}
        for name,pin in science['source_bindings'].items():
            guard();assert d.identity(pin['path'],hashed=True)==pin;saved[name]=d.read(pin['path'])
        split=saved['split'];s=split['scientific'];assert split['scientific_digest']==d.digest(s)
        assert saved['input_acceptance']['status']=='completed' and saved['input_acceptance']['actual_export']
        assert saved['target_acceptance']['status']=='completed_target_only' and saved['target_acceptance']['CADs']==60 and saved['target_acceptance']['cases']==180
        assert saved['target_acceptance']['FINAL_target_reads']==saved['target_acceptance']['scaler_fits']==0
        inputs=saved['new_input_index'];targets=saved['new_target_index'];dev=saved['DEV_inputs'];legacy=saved['legacy_TRAIN']
        assert legacy['status']==dev['status']=='completed' and legacy['feasibility_pass'] and dev['feasibility_pass']
        assert inputs['numerical_GT_members']==0 and inputs['inference_input_preparation_does_not_release_FINAL_targets']
        assert legacy['scaler']==inputs['scaler']==targets['scaler']==dev['scaler']==s['scaler']
        old=d.keys(s['old_TRAIN40_CADs']);new=d.keys(s['additional_TRAIN60_CADs']);train=d.keys(s['TRAIN_CADs'])
        development=d.keys(s['DEV_CADs']);final=d.keys(s['FINAL_CADs'])
        assert [len(old),len(new),len(train),len(development),len(final)]==[120,180,300,60,60]
        assert len(set(s['TRAIN_CADs']+s['DEV_CADs']+s['FINAL_CADs']))==140 and set(train)==set(old+new)
        assert set(legacy['records'])==set(old) and set(targets['records'])==set(new)
        assert set(inputs['records'])==set(new+final) and set(dev['records'])==set(development)
        records={};checked={};roles={};array_members_count=0
        def input_record(key,source,role):
            row=copy.deepcopy(source);row.update(scientific_group=role,source_input_group=source['group'])
            row['source_input_scientific_role']=source.get('scientific_role');row['scientific_role']=role
            assert row['key']==key and row['cad_id']==key.split('/')[0] and row['status']=='completed' and row['feasibility_pass']
            assert set(row['file_identities'])==set(MEMBERS) and 'target_path' not in row and 'physical_path' not in row
            return row
        for key in old:
            row=copy.deepcopy(legacy['records'][key]);row.update(scientific_group='TRAIN',source_input_group=row['group'],target_provenance='Previously completed oldTRAIN40 aligned target cache')
            assert row['key']==key and row['cad_id']==key.split('/')[0] and row['status']=='completed' and row['feasibility_pass']
            assert set(row['file_identities'])==set(MEMBERS)|{'target_path','physical_path'}
            records[key]=row
        for key in new:
            row=input_record(key,inputs['records'][key],'TRAIN');truth=targets['records'][key]
            assert truth['scientific_group']=='TRAIN' and truth['num_nodes']==row['num_nodes'] and truth['status']=='completed'
            assert row['permutation']['sha256']==truth['permutation_identity']['sha256']
            assert row['deferred_source']==truth['source']['Field']
            for member in ('target_path','physical_path'):
                row[member]=truth[member];row['file_identities'][member]=truth['file_identities'][member]
            row.update(target_provenance='Separate completed TRAIN-only target preparation',target_alignment=truth['permutation_identity'])
            records[key]=row
        for key in development:records[key]=input_record(key,dev['records'][key],'DEV')
        for key in final:records[key]=input_record(key,inputs['records'][key],'FINAL')
        assert len(records)==420
        for key,row in records.items():
            guard();role=row['scientific_group'];assert role in ('TRAIN','DEV','FINAL');roles[key]=role
            for name,pin in row['file_identities'].items():
                assert row[name]==pin['path']
                array_members_count+=1
                if pin['path'] not in checked:d.stat_matches(pin);checked[pin['path']]=pin
            if role!='TRAIN':assert 'target_path' not in row and 'physical_path' not in row
        manifest=dict(schema=SCHEMA,status='completed',feasibility_pass=True,scientific_digest=plan['scientific_digest'],
            split_scientific_digest=split['scientific_digest'],case_keys=train+development+final,case_roles=roles,
            train_keys={'40':old,'100':train},development_keys=development,FINAL_inference_keys=final,
            records=records,records_digest=d.digest(records),scaler=s['scaler'],scaler_provenance=legacy['contract']['legacy_manifest'],
            scaler_policy=s['scaler_policy'],source_bindings=science['source_bindings'],
            numeric_target_access='OnlyselectedTRAINscope; protected requests fail before library/array access',
            FINAL_numeric_targets_released=False,new_training_released=False,original200_protocol_reused=False,
            phase_scope='Metadata/stat assembly, not numeric-array validation or a solver result',array_body_reads=0,
            signed_fields='OriginalCSV GRID aligned signedU(mm)/S(MPa); no sign/scale replacement')
        d.write(output/'manifest.json',manifest)
        result.update(status='completed',TRAIN_CADs=100,knownDEV_CADs=20,reservedFINAL_CADs=20,
            total_case_records=420,TRAIN_target_records=300,protected_input_only_records=120,
            owned_array_stat_checks=len(checked),array_member_references=array_members_count,
            original200_protocol_reused=False,manifest_identity=d.identity(output/'manifest.json',hashed=True))
        guard()
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_replay=False)
        if (output/'manifest.json').exists():
            failed=d.read(output/'manifest.json');failed.update(status='failed',feasibility_pass=False,stage_error=str(exc))
            d.write(output/'manifest.json',failed)
        raise
    finally:
        result.update(closed=True,active=False,elapsed_epoch_seconds=time.time()-began[0],elapsed_monotonic_seconds=time.monotonic()-began[1])
        d.write(output/'summary.json',result);d.write(output/'status.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--start',action='store_true')
    args=parser.parse_args();assert not(args.freeze and args.start)
    if args.freeze:freeze()
    if args.start:run()
