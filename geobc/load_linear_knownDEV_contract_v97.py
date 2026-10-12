"""Independent metadata/source gates for fixed540 prediction-first diagnostic."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_load_linear_knownDEV_v97_design.json'
PLAN='notes/geobc_load_linear_knownDEV_v97_plan.json'
REVIEW='notes/geobc_load_linear_knownDEV_v97_root_review.json'
SOURCE_REVIEW='notes/geobc_load_linear_knownDEV_v97_source_review.json'
OUTPUT='runs/geobc_exploration/frozen3_load_linear_knownDEV_v97'
SOURCES=('geobc/load_linear_knownDEV_io_v97.py','geobc/load_linear_knownDEV_contract_v97.py','tools/evaluate_geobc_load_linear_knownDEV_v97.py','tools/run_geobc_load_linear_knownDEV_guard_v97.py','tools/prepare_geobc_load_linear_knownDEV_v97.py')


def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_load_linear_adapter_v96_plan.json')
    assert s['upstream_API_terminal']==b and a['all_model_invariants_passed']and a['committed_finite_predictions']==108 and a['GT_reads']==0
    report=d.read(check_identity(s['upstream_API_report_identity']));assert report['PNG_visually_accepted']and not report['GT_accuracy_measured']
    for c in s['frozen_candidates']:
        tp,ta,te,tb=terminal(c['training_plan_identity']['path']);assert tb==c['terminal_binding'];d.stat_matches(c['checkpoint'])
    m=d.read(check_identity(s['dataset_identity']));assert len(s['case_keys'])==60 and all(m['case_roles'][k]=='DEV'for k in s['case_keys'])
    assert len({k.split('/')[0]for k in s['case_keys']})==20 and set(s['case_keys'])==set(m['development_keys'])
    return design,p,b,m


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);design,pp,pb,m=inputs()
    assert p['kind']=='GeoBC_load_linear_knownDEV_v97'and p['output']==OUTPUT
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert r['predecessor_binding']==pb and s['design_identity']==d.identity(DESIGN,hashed=True)
    for k,v in design['scientific'].items():assert s[k]==v
    assert s['prediction_protocol']['total']==540 and s['whole_budget_seconds']==1800 and s['memory_limit_bytes']==30500000000
    for item in s['raw_deck_identities'].values():d.stat_matches(item)
    truths=d.read(check_identity(s['GT_manifest_identity']));assert set(s['case_keys'])==set(truths['records'])
    return p,r,m,truths
