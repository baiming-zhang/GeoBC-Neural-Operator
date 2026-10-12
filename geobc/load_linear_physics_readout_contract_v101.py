"""SameSCI CPU readout recovery of six immutable independent P2 solutions."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import check_identity
PLAN='notes/geobc_load_linear_physics_readout_v101_plan.json'
REVIEW='notes/geobc_load_linear_physics_readout_v101_root_review.json'
OUTPUT='runs/geobc_exploration/sixload_P2_CPU_readout_v101'
SOURCES=('geobc/load_linear_physics_readout_contract_v101.py','tools/readout_geobc_load_linear_physics_v101.py','tools/guard_geobc_load_linear_physics_readout_v101.py')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);old=d.read(check_identity(p['original_plan_identity']))
    assert p['kind']=='GeoBC_P2_CPU_dtype_readout_recovery_v101'and p['output']==OUTPUT
    assert s==old['scientific']and d.digest(s)==p['scientific_digest']==old['scientific_digest']==r['scientific_digest']
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==v for k,v in p['source_pins'].items())
    a=d.read(check_identity(p['original_summary_identity']));e=d.read(check_identity(p['original_external_identity']))
    assert a['status']=='failed'and not a['active']and a['closed']and a['GT_reads']==0 and a['committed_finite_endpoints']==6 and a['error_type']=='AssertionError'
    assert e['closed']and not e['active']and e['exit_code']==1 and d.exact_process_exited(e['guard_process'])and d.exact_process_exited(e['child_process'])
    assert a['parallel_K_qualification']['passed']and a['algebra_guards']['K_applications']==8 and len(a['completed_groups'])==2
    return p,r,a,d.read(check_identity(s['owned_original_GT_manifest_identity']))
