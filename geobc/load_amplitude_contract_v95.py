"""Metadata gate for target-free positive-load displacement consistency."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity
DESIGN='notes/geobc_load_amplitude_homogeneity_v95_design.json'
PLAN='notes/geobc_load_amplitude_homogeneity_v95_plan.json'
REVIEW='notes/geobc_load_amplitude_homogeneity_v95_root_review.json'
SOURCE_REVIEW='notes/geobc_load_amplitude_homogeneity_v95_source_review.json'
OUTPUT='runs/geobc_exploration/frozen3_load_amplitude_TRAIN9_v95'
SOURCES=('geobc/load_amplitude_contract_v95.py','tools/evaluate_geobc_load_amplitude_v95.py',
    'tools/run_geobc_load_amplitude_guard_v95.py','tools/prepare_geobc_load_amplitude_v95.py')


def inputs():
    design=d.read(DESIGN);s=design['scientific'];assert d.digest(s)==design['scientific_digest']
    p,a,e,b=terminal('notes/geobc_edgeconv_geometry_v94_plan.json')
    t=d.read(check_identity(s['prior_v94_terminal_identity']))
    decision=d.read(check_identity(s['prior_v94_decision_identity']))
    assert t['terminal_binding']==b and not a['TRAIN_gate']['passed'] and decision['PNG_visually_accepted'] and not decision['automatic100']
    assert s['selected_TRAIN_input_keys']==p['scientific']['selected_TRAIN_keys']
    for c in s['frozen_candidates']:
        tp,ta,te,tb=terminal(c['training_plan_identity']['path'])
        assert d.identity(c['training_plan_identity']['path'],hashed=True)==c['training_plan_identity']
        assert c['terminal_binding']==tb and c['training_SCI']==tp['scientific_digest']
        assert tp['scientific']['dataset_identity']==s['dataset_identity'] and tp['scientific']['scaler']==s['scaler']
        d.stat_matches(c['checkpoint'])
    for item in s['raw_deck_identities'].values():d.stat_matches(item)
    return design,b


def approved_contract(plan_path):
    p=d.read(plan_path);r=d.read(p['root_review']);s=p['scientific'];design,b=inputs()
    assert p['kind']=='GeoBC_frozen3_targetfree_load_amplitude_v95' and p['output']==OUTPUT
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert r['predecessor_binding']==b and s['design_identity']==d.identity(DESIGN,hashed=True)
    for k,v in design['scientific'].items():assert s[k]==v
    assert s['load_amplitudes']==[.5,1.,1.5] and s['prediction_denominator']==81 and s['property_denominator']==54
    assert s['budget']['whole_seconds']==900 and s['budget']['memory_bytes']==30500000000
    assert s['raster_absolute_tolerance']==.0001 and s['force_absolute_tolerance']==.000002
    m=d.read(check_identity(s['dataset_identity']))
    assert len(s['selected_TRAIN_input_keys'])==9 and all(m['case_roles'][k]=='TRAIN'for k in s['selected_TRAIN_input_keys'])
    return p,r,m
