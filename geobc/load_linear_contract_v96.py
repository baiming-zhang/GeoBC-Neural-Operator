"""Freeze reusable API validation against real v95 terminal and owned refs."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_load_linear_adapter_v96_design.json'
PLAN='notes/geobc_load_linear_adapter_v96_plan.json'
REVIEW='notes/geobc_load_linear_adapter_v96_root_review.json'
SOURCE_REVIEW='notes/geobc_load_linear_adapter_v96_source_review.json'
OUTPUT='runs/geobc_exploration/frozen3_load_linear_API_v96'
SOURCES=('geobc/load_linear_adapter_v96.py','geobc/load_linear_io_v96.py','geobc/load_linear_contract_v96.py',
    'tools/evaluate_geobc_load_linear_v96.py','tools/run_geobc_load_linear_guard_v96.py','tools/prepare_geobc_load_linear_v96.py')


def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_load_amplitude_homogeneity_v95_plan.json')
    assert d.read(check_identity(s['predecessor_terminal_identity']))['terminal_binding']==b
    decision=d.read(check_identity(s['predecessor_decision_identity']));assert decision['PNG_visually_accepted']and decision['GT_reads']==0
    assert d.identity(s['physics_metadata_identity']['path'],hashed=True)==s['physics_metadata_identity']
    assert s['physics_metadata_identity']==b['summary']
    assert d.identity(s['source_prototype_identity']['path'],hashed=True)==s['source_prototype_identity']
    review=d.read(check_identity(s['prototype_review_identity']));assert review['passed']and not review['actual_execution_released']
    assert a['protocol_qualified']and a['prediction_barrier_complete']and a['committed_finite_predictions']==81 and a['property_rows']==54 and a['GT_reads']==0
    assert s['selected_TRAIN_input_keys']==p['scientific']['selected_TRAIN_input_keys']and s['frozen_candidates']==p['scientific']['frozen_candidates']
    assert all(x['same_constraints_across3']and x['zero_constraints']for x in a['physics_gate'].values())
    for c in s['frozen_candidates']:
        tp,ta,te,tb=terminal(c['training_plan_identity']['path']);assert tb==c['terminal_binding'];d.stat_matches(c['checkpoint'])
    return design,p,a,b


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);design,pp,pa,pb=inputs()
    assert p['kind']=='GeoBC_reusable_load_linear_API_v96'and p['output']==OUTPUT
    assert r['passed']and r['whole_source_reviewed']and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest']and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins'])and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert r['predecessor_binding']==pb and s['design_identity']==d.identity(DESIGN,hashed=True)
    for k,v in design['scientific'].items():assert s[k]==v
    assert s['candidate_prediction_denominator']==108 and s['amplitude_property_denominator']==54 and s['mixture_internal_property_denominator']==27
    assert s['budget']['whole_seconds']==900 and s['budget']['memory_bytes']==30500000000
    assert s['owned_v95_inventory']==pa['prediction_inventory']
    m=d.read(check_identity(s['dataset_identity']));assert all(m['case_roles'][k]=='TRAIN'for k in s['selected_TRAIN_input_keys'])
    return p,r,m,pa
