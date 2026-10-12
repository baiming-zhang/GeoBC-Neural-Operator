"""Independent CPU readout recovery; sameSCI, no model/prediction replay."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import check_identity
PLAN='notes/geobc_load_linear_identity_readout_v98_plan.json'
REVIEW='notes/geobc_load_linear_identity_readout_v98_root_review.json'
OUTPUT='runs/geobc_exploration/frozen3_load_linear_identity_readout_v98'
SOURCES=('geobc/load_linear_identity_readout_contract_v98.py','tools/readout_geobc_load_linear_identity_v98.py','tools/run_geobc_load_linear_identity_guard_v98.py')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);original=d.read(check_identity(p['original_plan_identity']))
    assert p['kind']=='GeoBC_load_linear_CPU_identity_readout_recovery_v98'and p['output']==OUTPUT
    assert d.digest(s)==p['scientific_digest']==original['scientific_digest']==r['scientific_digest']and s==original['scientific']
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==h for k,h in p['source_pins'].items())
    a=d.read(check_identity(p['original_summary_identity']));e=d.read(check_identity(p['original_external_identity']))
    assert a['status']=='failed'and not a['active']and a['closed']and a['GT_reads']==0 and a['checkpoint_loads']==3 and a['committed_finite_predictions']==540 and a['prediction_barrier_complete']
    assert a['error_type']=='AssertionError'and not e['active']and e['closed']and e['exit_code']==1
    assert d.exact_process_exited(e['child_process'])and d.exact_process_exited(e['guard_process'])
    assert all(v['qualification']['passed']and v['predictions']==180 and v['strict_checkpoint_verified']and v['model_bytes_unchanged']for v in a['candidates'].values())
    assert len(a['physics_gate'])==20 and a['source_deck_parses']==20
    return p,r,a,d.read(check_identity(s['GT_manifest_identity']))
