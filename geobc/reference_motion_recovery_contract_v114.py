"""Independent schema recovery; missing derived witness stays negative."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import check_identity
PLAN='notes/geobc_reference_motion_recovery_v114_plan.json'
REVIEW='notes/geobc_reference_motion_recovery_v114_root_review.json'
SOURCE_REVIEW='notes/geobc_reference_motion_recovery_v114_source_review.json'
OUTPUT='runs/geobc_exploration/twoCAD_reference_motion_recovery_v114'
SOURCES=('geobc/reference_motion_recovery_contract_v114.py','tools/evaluate_geobc_reference_motion_recovery_v114.py','tools/guard_geobc_reference_motion_recovery_v114.py')

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);old=d.read(check_identity(p['original_plan_identity']))
    a=d.read(check_identity(p['original_terminal_binding']['summary']));e=d.read(check_identity(p['original_terminal_binding']['external']))
    assert p['kind']=='GeoBC_reference_motion_schema_recovery_v114' and p['output']==OUTPUT
    assert p['scientific']==old['scientific'] and p['scientific_digest']==old['scientific_digest']==d.digest(p['scientific'])==r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    assert a['status']=='failed' and a['closed']and not a['active'] and a['error_type']=='AssertionError'
    assert a['input_cache_reads']==1 and a['basis_array_reads']==a['independent_mixed_array_reads']==a['committed_pose_requests']==0
    assert e['exit_code']==1 and e['actual_child_exited'] and d.exact_process_exited(e['child_process'])and d.exact_process_exited(e['guard_process'])
    assert p['original_guard_deadline_epoch']==e['deadline_epoch'] and p['original_guard_deadline_monotonic']==e['deadline_monotonic']
    return p,r,a,e
