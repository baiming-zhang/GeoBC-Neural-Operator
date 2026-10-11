"""Matched per-case exposure fitting diagnostic on three difficult TRAIN CADs."""
from . import available_cad_pilot_data as d
from .fno20_readout_contract_v57 import training_binding,terminal,check_identity,complete_physical

SOURCES=('geobc/shared_fit_contract_v58.py','tools/train_geobc_shared_fit_v58.py',
    'tools/run_geobc_shared_fit_guard_v58.py','tools/prepare_geobc_shared_fit_v58.py')


def inputs():
    vp,va,ve,vb=terminal('notes/geobc_fno20_readout_v57_plan.json');up=training_binding()
    assert va['protocol_qualified'] and va['DEV_gate']['passed']
    rows=up['summary']['endpoints']['6000']['full_TRAIN']['per_case']
    cads=sorted({k.split('/')[0] for k in rows})
    scores={c:sum(rows[c+'/'+l]['stress_relative_l2']for l in ('ver','hor','tor'))/3 for c in cads}
    selected=sorted(cads,key=lambda c:(-scores[c],c))[:3]
    keys=[c+'/'+l for c in selected for l in ('ver','hor','tor')]
    return up,vp,vb,selected,keys,scores


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up,vp,vb,cads,keys,scores=inputs()
    assert p['kind']=='GeoBC_FNO20_selected_TRAIN_shared_vs_independent_v58'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert all(p['source_pins'].get(k)==h for k,h in vp['source_pins'].items())
    assert r['upstream_terminal_bindings']==dict(TRAIN20=up['binding'],DEV20=vb)
    ps=up['plan']['scientific']
    for k in ('family','seed','train_CAD_count','dataset_identity','scaler','optimizer','precision','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size'):
        assert s[k]==ps[k]
    assert s['selected_CADs']==cads and s['selected_TRAIN_keys']==keys and len(keys)==9
    assert s['selection_rule']=='Three highest TRAIN20 mean signed stress relative L2 across ver/hor/tor; ties CAD lexicographic'
    assert s['selection_scores']=={c:scores[c]for c in cads}
    assert s['parent_checkpoint']==up['checkpoint'] and s['parent_training_SCI']==up['plan']['scientific_digest']
    assert s['parent_TRAIN_rows']=={k:up['summary']['endpoints']['6000']['full_TRAIN']['per_case'][k]for k in keys}
    assert s['arms']==['shared9','independent9'] and s['additional_exposures_per_case']==100
    assert s['endpoints_additional_exposures']==[0,20,100]
    assert s['arm_optimizer_updates']==900 and s['total_optimizer_updates']==1800
    assert s['cache_setup_budget_seconds']==300 and s['training_budget_seconds']==600 and s['whole_budget_seconds']==1200
    assert s['DEV_FINAL_reads_allowed']==0 and not s['automatic_continuation'] and not s['FINAL_reuse_for_selection']
    assert p['output']=='runs/geobc_exploration/FNO20_selected_shared_fit_v58'
    m=d.read(check_identity(s['dataset_identity']))
    assert all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in keys)
    return p,r,m,up
