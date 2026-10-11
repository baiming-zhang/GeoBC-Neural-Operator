"""Full knownDEV60 readout only after genuine fixed20 TRAIN qualification."""
from . import available_cad_pilot_data as d
from .fno_learning20_contract_v56 import terminal,metadata_inputs
from .physical10_readout_contract_v16 import check_identity,complete_physical

METHOD='FNO100_PHYSICAL_6000'
SOURCES=('geobc/fno20_readout_contract_v57.py','tools/evaluate_geobc_fno20_v57.py',
    'tools/run_geobc_fno20_readout_guard_v57.py','tools/prepare_geobc_fno20_readout_v57.py')


def training_binding():
    p,a,e,b=terminal('notes/geobc_fno_learning20_v56_plan.json');s=p['scientific'];up=metadata_inputs()
    assert a['training_complete'] and a['TRAIN_gate']['passed']
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==6000
    assert a['additional_optimizer_step_completions']==3000 and not a['possible_partial_optimizer_update']
    assert a['case_exposure_counts']=={k:20 for k in s['TRAIN_keys']}
    assert a['actual_largest_fullN_gradient_check']['passed'] and a['strict_loaded_checkpoint_tensor_state_verified']
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    complete_physical(a['endpoints']['6000']['full_TRAIN'],s['TRAIN_keys'])
    ck=a['endpoints']['6000']['checkpoint_identity'];d.stat_matches(ck)
    report=d.read('results/geobc_fno_learning20_v56/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    costs=d.read('results/geobc_fno_learning20_v56/metrics.json')['complete0to20_cost']
    return dict(plan=p,summary=a,external=e,binding=b,checkpoint=ck,parents=up,complete0to20_cost=costs)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=training_binding()
    assert p['kind']=='GeoBC_FNO_physical10_to20_knownDEV60_v57'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['plan']['source_pins'].items())
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['memory_limit_bytes']==30500000000
    assert s['whole_budget_seconds']==1800 and s['seed']==2026 and s['decoder_chunk_size']==32768
    assert s['GT_policy']=='All requested full-node predictions committed and revalidated before shared owned-knownDEV60 once'
    assert s['pool']=='known_development60_availability_conditional_not_FINAL'
    assert s['training_SCI']==up['plan']['scientific_digest']
    assert s['dataset_identity']==up['plan']['scientific']['dataset_identity'] and s['scaler']==up['plan']['scientific']['scaler']
    m=d.read(check_identity(s['dataset_identity']));keys=s['case_keys']
    assert keys==m['development_keys'] and len(keys)==len(set(keys))==60
    cads={k.split('/')[0]for k in keys}
    assert len(cads)==20 and set(keys)=={c+'/'+load for c in cads for load in ('ver','hor','tor')}
    assert all(m['case_roles'][k]=='DEV' for k in keys)
    known=up['parents']['DEV_plan']['scientific']
    assert s['GT_manifest_identity']==known['GT_manifest_identity'] and s['case_keys']==known['case_keys']
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for k in keys:
        t=truths['records'][k];assert t['all_original_nodes'] and t['signed_stress_unchanged']
        assert t['physical']['shape']==[m['records'][k]['num_nodes'],4]
        assert not {'target_path','physical_path'}&set(m['records'][k]['file_identities'])
    candidate=dict(name=METHOD,family='FNO9',step=6000,training_plan_identity=d.identity('notes/geobc_fno_learning20_v56_plan.json',hashed=True))
    assert s['candidates']==[candidate] and s['fixed_prediction_denominator']==60 and not s['FINAL_reuse_for_selection']
    assert r['training_bindings']=={METHOD:up['binding']}
    assert s['baseline_development_binding']==up['parents']['DEV_binding']
    assert s['baseline_development_SCI']==up['parents']['DEV_plan']['scientific_digest']
    assert s['training_report_metrics_identity']==d.identity('results/geobc_fno_learning20_v56/metrics.json',hashed=True)
    assert s['DEV_gate']==dict(primary_absolute_improvement_at_least=.03,
        maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60)
    assert p['output']=='runs/geobc_exploration/FNO100_knownDEV60_physical20_v57'
    return p,r,m,truths,{METHOD:up}
