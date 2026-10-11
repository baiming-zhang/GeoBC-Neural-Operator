"""Frozen FINAL20 evaluation metadata; never opens numerical targets."""
from . import available_cad_pilot_data as d
from .operator_hash_readout_contract_v5 import check_identity,complete_physical

KIND='GeoBC_frozen_FINAL20_two_solver_v9'


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']==KIND and r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    required=('geobc/final_cad_contract_v9.py','geobc/final_cad_truth_v9.py','tools/evaluate_geobc_FINAL_v9.py','tools/run_geobc_FINAL_guard_v9.py')
    assert all(k in p['source_pins'] for k in required) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    assert s['primary']=='FNO100_12000' and s['primary_change_after_FINAL_allowed'] is False
    assert s['whole_budget_seconds']==1800 and s['memory_limit_bytes']==30500000000
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['training_updates']==s['scaler_fits']==0 and s['fixed_prediction_denominator']==120 and s['fixed_GT_case_denominator']==60
    decision=d.read(check_identity(s['primary_decision_identity']))
    assert decision['primary']==s['primary'] and decision['status']=='actual_completed_matched12000_reviewed' and decision['PNG_root_actual_viewed']
    upstream=r['knownDEV_binding']
    for item in upstream.values():check_identity(item)
    old=d.read(upstream['summary']['path']);external=d.read(upstream['external']['path'])
    assert old['status']=='completed' and old['protocol_qualified'] and old['closed'] and not old['active']
    assert external['exit_code']==0 and external['within_external_budget'] and external['closed'] and not external['active']
    assert all(d.exact_process_exited(external[k]) for k in ('guard_process','child_process'))
    assert old['committed_finite_predictions']==old['owned_GT_array_reads']==60
    manifest=d.read(check_identity(s['dataset_identity']))
    assert manifest['status']=='completed' and manifest['feasibility_pass'] and not manifest['FINAL_numeric_targets_released']
    keys=s['case_keys'];assert keys==manifest['FINAL_inference_keys'] and len(keys)==len(set(keys))==60
    cads=list(dict.fromkeys(k.split('/')[0] for k in keys));assert len(cads)==20 and s['CADs']==cads
    assert set(keys)=={cad+'/'+load for cad in cads for load in d.LOADS}
    assert not set(keys)&set(manifest['development_keys']) and not set(keys)&set(manifest['train_keys']['100'])
    assert s['scaler']==manifest['scaler']
    for cad in cads:
        source=s['FINAL_sources'][cad]
        records=[manifest['records'][cad+'/'+load] for load in d.LOADS]
        assert all(row['scientific_group']=='FINAL' and not {'target_path','physical_path'}&set(row) for row in records)
        assert all(row['deferred_source']==source['CSV'] and row['permutation']==source['permutation'] and row['num_nodes']==source['num_nodes'] for row in records)
        assert source['CSV']['source_role']=='official_TRAIN_pool' and source['CSV']['prospective_role']=='FINAL_inference_input_only'
        assert source['CSV']['numeric_targets_permitted'] is False
        # Old source role stays immutable. This separate protocol authorizes a
        # single numerical parse only after the complete prediction barrier.
        d.stat_matches(source['CSV']);d.stat_matches(source['permutation'])
    candidates=s['candidates'];assert [c['name'] for c in candidates]==['FNO100_12000','GLOBAL_HASH100_12000']
    assert set(r['training_bindings'])=={c['name'] for c in candidates}
    qualified={}
    for candidate in candidates:
        name=candidate['name'];tp=check_identity(candidate['training_plan_identity'])
        if name=='FNO100_12000':
            from .fno_coverage_contract_v4 import approved_contract as training_contract
        else:
            from .operator_hash_extension_contract_v7 import approved_contract as training_contract
        train,_,dataset,order=training_contract(d.relative(tp));ts=train['scientific']
        binding=r['training_bindings'][name];folder=d.path(train['output'])
        for kind,file in [('summary','summary.json'),('external','external_terminal.json'),('status','status.json')]:
            assert d.relative(binding[kind]['path'])==d.relative(folder/file);check_identity(binding[kind])
        summary=d.read(binding['summary']['path']);outer=d.read(binding['external']['path'])
        assert summary==d.read(binding['status']['path']) and summary['status']=='completed' and summary['training_complete']
        assert summary['closed'] and not summary['active'] and outer['closed'] and not outer['active']
        assert outer['exit_code']==0 and outer['within_external_budget'] and not outer['external_timeout']
        assert all(d.exact_process_exited(outer[k]) for k in ('guard_process','child_process'))
        assert summary['execution']['run_process']==outer['child_process'] and summary['execution']['guard']['guard_process']==outer['guard_process']
        assert summary['execution']['scientific_digest']==outer['scientific_digest']==train['scientific_digest']
        assert ts['family']==candidate['family'] and ts['train_CAD_count']==100 and ts['seed']==2026
        assert dataset==manifest and ts['dataset_identity']==s['dataset_identity'] and ts['scaler']==s['scaler']
        assert len(order)==summary['step']==summary['optimizer_step_attempts']==summary['optimizer_step_completions']==12000
        assert summary['case_exposure_counts']=={key:40 for key in ts['TRAIN_keys']}
        assert summary['actual_largest_fullN_gradient_check']['passed'] and not summary['possible_partial_optimizer_update']
        assert summary['raw_GT_reads']==summary['DEV_FINAL_reads']==summary['scaler_fits']==0
        assert max(summary['training_epoch_seconds'],summary['training_monotonic_seconds'])<=ts['training_budget_seconds']
        assert max(summary['elapsed_epoch_seconds'],summary['elapsed_monotonic_seconds'],outer['elapsed_epoch_seconds'],outer['elapsed_monotonic_seconds'])<=ts['whole_budget_seconds']
        assert max(summary[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=s['memory_limit_bytes']
        assert all(p['source_pins'].get(k)==v for k,v in train['source_pins'].items())
        point=summary['endpoints']['12000'];complete_physical(point['full_TRAIN'],ts['TRAIN_keys'])
        assert candidate['step']==12000 and point['case_exposure_count']==40
        checkpoint=point['checkpoint_identity'];assert candidate['checkpoint_identity']==checkpoint
        d.stat_matches(checkpoint)
        qualified[name]=dict(plan=train,summary=summary,external=outer,checkpoint=checkpoint)
    return p,r,manifest,qualified
