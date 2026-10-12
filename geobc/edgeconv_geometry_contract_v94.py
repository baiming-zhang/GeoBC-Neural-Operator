"""Metadata-only plan gating for one dynamic geometry descriptor screen."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity

DESIGN='notes/geobc_edgeconv_geometry_v94_design.json'
PLAN='notes/geobc_edgeconv_geometry_v94_plan.json'
REVIEW='notes/geobc_edgeconv_geometry_v94_root_review.json'
SOURCE_REVIEW='notes/geobc_edgeconv_geometry_v94_source_review.json'
OUTPUT='runs/geobc_exploration/FNO_edgeconv_geometry_TRAIN9_v94'
SOURCES=('geobc/edgeconv_geometry_v94.py','geobc/edgeconv_geometry_contract_v94.py',
    'tools/train_geobc_edgeconv_geometry_v94.py','tools/run_geobc_edgeconv_geometry_guard_v94.py',
    'tools/prepare_geobc_edgeconv_geometry_v94.py')


def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    assert design['actual_source_implementation']==design['actual_GPU']==0
    bp,ba,be,bb=terminal(s['parent_reference_plan_identity']['path'])
    assert d.identity(s['parent_reference_plan_identity']['path'],hashed=True)==s['parent_reference_plan_identity']
    report=d.read(check_identity(s['reference_report_identity']));assert report['status']=='completed' and report['terminal_binding']==bb
    latest=d.read(check_identity(s['latest_real_holdout_terminal_identity']))
    rp,ra,re,rb=terminal('notes/geobc_reserve_holdout_v93_plan.json')
    assert latest['terminal_binding']==rb and ra['committed_finite_predictions']==18 and ra['source_GT_parses']==2
    decision=d.read('notes/geobc_reserve_holdout_v93_actual_decision.json')
    assert decision['PNG_visually_accepted'] and decision['reserves_now_revealed'] and not decision['reuse_for_tuning']
    old=ba['arms']['FNO9_plain_query_MLP'];assert old['qualification']['passed'] and old['step']==900
    assert s['selected_TRAIN_keys']==bp['scientific']['selected_TRAIN_keys'] and s['scaler']==bp['scientific']['scaler']
    assert not {k.split('/')[0]for k in s['selected_TRAIN_keys']}&set(rp['scientific']['CADs'])
    return design,bp,old,bb,rb


def approved_contract(plan_path):
    p=d.read(plan_path);r=d.read(p['root_review']);s=p['scientific'];design,bp,old,bb,rb=inputs()
    assert p['kind']=='GeoBC_edgeconv_geometry_TRAIN9_v94' and p['output']==OUTPUT
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert r['baseline_binding']==bb and r['latest_holdout_binding']==rb
    assert s['design_identity']==d.identity(DESIGN,hashed=True) and s['design_scientific_digest']==design['scientific_digest']
    for k in ('selected_TRAIN_keys','dataset_identity','scaler','seed'):
        assert s[k]==design['scientific'][k]
    assert s['candidate']==s['model_package']=='FNO9_EDGE_CONV'
    assert s['fixed_updates']==900 and s['case_exposures']==100 and s['endpoints_case_exposures']==[100]
    assert s['parameter_count_expected']==4562122 and s['expected_plain_model_initial_digest']==old['initialization']['model_digest']
    assert s['baseline_scalar_endpoints']=={'100':old['endpoints']['100']}
    assert s['expected_operator_initial_digest']==old['initialization']['shared_body_digests']['operator']
    assert s['optimizer']==bp['scientific']['optimizer']
    budget=design['scientific']['budget']
    assert s['memory_limit_bytes']==budget['memory_bytes']==30500000000 and s['GPU_lock']==budget['GPU_lock']
    assert s['whole_campaign_budget_seconds']==budget['whole_seconds']==1200
    assert s['cache_setup_budget_seconds']==budget['cache_seconds']==120 and s['arm_budget_seconds']==budget['train_seconds']==600
    assert s['readout_budget_seconds_per_endpoint']==budget['readout_seconds']==180
    assert not s['automatic100'] and not s['automatic_replay'] and s['parent_checkpoint_reads_allowed']==0
    m=d.read(check_identity(s['dataset_identity']));keys=s['selected_TRAIN_keys']
    assert len(keys)==len(set(keys))==9 and all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in keys)
    assert m['scaler']==s['scaler']
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,100,s['seed']);assert d.digest(order)==p['full_order_digest']==s['full_order_digest']
    return p,r,m,order,rb
