"""Metadata-only one-shot release of fixed decoder package TRAIN9 screen."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity

PLAN='notes/geobc_decoder_packages_v77_plan.json'
REVIEW='notes/geobc_decoder_packages_v77_root_review.json'
DESIGN='notes/geobc_decoder_packages_v77_design.json'
SOURCES=('geobc/decoder_packages_v77.py','geobc/decoder_packages_cache_v77.py',
    'geobc/decoder_packages_contract_v77.py','tools/train_geobc_decoder_packages_v77.py',
    'tools/run_geobc_decoder_packages_guard_v77.py')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);design=d.read(DESIGN)
    assert p['kind']=='GeoBC_three_decoder_packages_TRAIN9_v77'
    assert s==design['scientific'] and d.digest(s)==p['scientific_digest']==design['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['scientific_digest']==p['scientific_digest'] and r['source_pins']==p['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    pp,pa,pe,pb=terminal('notes/geobc_local_query_geometry_v76_plan.json')
    assert r['upstream_terminal_binding']==pb and pa['protocol_qualified']
    decision=d.read(check_identity(design['basis_v76_actual_decision_identity']))
    assert decision['PNG_visually_verified'] and decision['terminal_binding']==pb
    assert s['selected_TRAIN_keys']==pp['scientific']['selected_TRAIN_keys']
    assert s['total_updates_per_arm']==900 and s['case_exposures']==100 and s['endpoints_case_exposures']==[0,10,100]
    assert s['decoder_chunk_size']==32768 and s['memory_limit_bytes']==30500000000
    assert not s['automatic100'] and not s['automatic_continuation'] and not s['automatic_replay']
    m=d.read(check_identity(s['dataset_identity']));keys=s['selected_TRAIN_keys']
    assert m['scaler']==s['scaler'] and len(keys)==len(set(keys))==9 and all(k in m['train_keys']['100'] and m['case_roles'][k]=='TRAIN' for k in keys)
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,100,s['seed']);assert d.digest(order)==p['full_order_digest']
    assert p['output']=='runs/geobc_exploration/FNO_decoder_packages_TRAIN9_v77'
    return p,r,m,order,pb
