"""Metadata-only release for one matched TRAIN9 geometry query screen."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity

PLAN='notes/geobc_local_query_geometry_v76_plan.json'
REVIEW='notes/geobc_local_query_geometry_v76_root_review.json'
DESIGN='notes/geobc_local_query_geometry_v76_design.json'
SOURCES=('geobc/local_query_geometry_v76.py','geobc/local_query_geometry_cache_v76.py',
    'geobc/local_query_geometry_contract_v76.py','tools/train_geobc_local_query_geometry_v76.py',
    'tools/run_geobc_local_query_geometry_guard_v76.py')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);design=d.read(DESIGN)
    assert p['kind']=='GeoBC_matched_local_global_geometry_TRAIN9_v76'
    assert s==design['scientific'] and p['scientific_digest']==design['scientific_digest']==d.digest(s)
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['scientific_digest']==p['scientific_digest'] and r['source_pins']==p['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    pp,pa,pe,pb=terminal('notes/geobc_field_amplitude_shape_v75_plan.json')
    assert r['upstream_terminal_binding']==pb and pa['protocol_qualified']
    decision=d.read(check_identity(design['basis_actual75_decision_identity']))
    assert decision['PNG_visually_verified'] and decision['terminal_binding']==pb
    assert d.read('results/geobc_field_amplitude_shape_v75/manifest.json')['status']=='completed'
    assert s['total_updates_per_arm']==900 and s['case_exposures']==100 and s['endpoints_case_exposures']==[0,10,100]
    assert s['decoder_chunk_size']==32768 and s['memory_limit_bytes']==30500000000 and not s['automatic_replay'] and not s['automatic100']
    m=d.read(check_identity(s['dataset_identity']));keys=s['selected_TRAIN_keys']
    assert len(keys)==len(set(keys))==9 and all(k in m['train_keys']['100'] and m['case_roles'][k]=='TRAIN' for k in keys)
    assert m['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,100,s['seed'])
    assert d.digest(order)==p['full_order_digest']
    assert p['output']=='runs/geobc_exploration/FNO_local_global_geometry_TRAIN9_v76'
    return p,r,m,order,pb
