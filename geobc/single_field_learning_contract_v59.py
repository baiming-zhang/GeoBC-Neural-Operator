"""One bounded three-field learning continuation from actual own100 endpoints."""
from . import available_cad_pilot_data as d
from .shared_fit_contract_v58 import terminal,check_identity,complete_physical

SOURCES=('geobc/single_field_learning_contract_v59.py','tools/train_geobc_single_field_v59.py',
    'tools/run_geobc_single_field_guard_v59.py','tools/prepare_geobc_single_field_v59.py')


def inputs():
    p,a,e,b=terminal('notes/geobc_shared_fit_v58_plan.json');s=p['scientific']
    assert a['training_complete'] and a['optimizer_step_attempts']==a['optimizer_step_completions']==1800
    assert a['parent_checkpoint_unchanged'] and a['baseline_parent_scalar_match']
    report=d.read('results/geobc_shared_fit_v58/manifest.json')
    assert report['status']=='completed' and report['SCI']==p['scientific_digest'] and report['terminal_binding']==b
    keys=[max([c+'/'+l for l in ('ver','hor','tor')],key=lambda k:(s['parent_TRAIN_rows'][k]['stress_relative_l2'],k))for c in s['selected_CADs']]
    for k in keys:
        row=a['arms']['independent9'][k]
        assert row['status']=='completed' and row['additional_updates']==100 and row['optimizer_step']==6100
        assert row['case_exposure_counts']=={key:(120 if key==k else 20)for key in s['parent_TRAIN_keys']}
        complete_physical(row['endpoints']['100'],[k]);d.stat_matches(row['checkpoint_identity'])
    return dict(plan=p,summary=a,external=e,binding=b,keys=keys)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();ps=up['plan']['scientific'];keys=up['keys']
    assert p['kind']=='GeoBC_FNO_three_single_fields_physical100_to1000_v59'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in up['plan']['source_pins'].items())
    assert r['upstream_terminal_binding']==up['binding']
    for k in ('family','seed','train_CAD_count','dataset_identity','scaler','optimizer','precision','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size','parent_TRAIN_keys'):
        assert s[k]==ps[k]
    assert s['selected_TRAIN_keys']==keys and len(keys)==3
    assert s['parent_checkpoints']=={k:up['summary']['arms']['independent9'][k]['checkpoint_identity']for k in keys}
    assert s['parent_scalar_rows']=={k:up['summary']['arms']['independent9'][k]['endpoints']['100']['per_case'][k]for k in keys}
    assert s['parent_training_SCI']==up['plan']['scientific_digest'] and s['parent_optimizer_step']==6100
    assert s['parent_case_exposure']==120 and s['additional_updates_per_field']==900 and s['total_additional_updates']==2700
    assert s['fixed_total_additional_exposure_endpoints_since_shared20']==[300,1000]
    assert s['final_optimizer_step_per_model']==7000 and s['final_case_exposure_per_field']==1020
    assert s['whole_budget_seconds']==900 and s['training_budget_seconds']==600 and s['cache_setup_budget_seconds']==180
    assert s['DEV_FINAL_reads_allowed']==0 and not s['automatic_continuation'] and not s['FINAL_reuse_for_selection']
    assert p['output']=='runs/geobc_exploration/FNO_three_single_fields_learning_v59'
    m=d.read(check_identity(s['dataset_identity']))
    assert all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in keys)
    return p,r,m,up
