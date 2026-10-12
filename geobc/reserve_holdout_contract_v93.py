"""Two already reserved CADs; one prospective prediction-first readout."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal

PLAN='notes/geobc_reserve_holdout_v93_plan.json'
REVIEW='notes/geobc_reserve_holdout_v93_root_review.json'
DESIGN='notes/geobc_reserve_holdout_v93_design.json'
SOURCES=('geobc/reserve_holdout_contract_v93.py','tools/evaluate_geobc_reserve_holdout_v93.py','tools/run_geobc_reserve_holdout_guard_v93.py','tools/prepare_geobc_reserve_holdout_v93.py')


def approved(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_reserved2_frozen3_same20_prediction_first_v93'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==d.digest(s)==r['scientific_digest']
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert s['CADs']==['50_380','624_483'] and s['loads']==['ver','hor','tor']
    assert s['case_keys']==d.keys(s['CADs']) and s['fixed_prediction_denominator']==18
    assert s['whole_campaign_budget_seconds']==900 and s['memory_limit_bytes']==30500000000
    assert s['training_updates_allowed']==s['scaler_fits_allowed']==0 and not s['automatic_continuation']
    assert p['output']=='runs/geobc_exploration/reserved2_frozen3_same20_v93'
    for item in [s['exposure_evidence_identity'],s['split_identity'],s['input_contract_identity']]:
        assert d.identity(item['path'],hashed=True)==item
    evidence=d.read(s['exposure_evidence_identity']['path'])
    assert evidence['unexposed_by_recorded_execution_metadata'] and evidence['array_filename_matches']==[]
    contract=d.read(s['input_contract_identity']['path'])
    assert contract['raw_sources']==s['raw_sources'] and contract['scaler']==s['scaler']
    assert contract['groups']['train_new160']==[] and contract['groups']['validation20']==s['CADs']
    for candidate in s['candidates']:
        tp,ta,te,tb=terminal(candidate['training_plan_identity']['path'])
        assert d.identity(candidate['training_plan_identity']['path'],hashed=True)==candidate['training_plan_identity']
        assert tb==candidate['terminal_binding'] and tp['scientific_digest']==candidate['training_SCI']
        assert s['scaler']==tp['scientific']['scaler'] and s['dataset_identity']==tp['scientific']['dataset_identity']
        d.stat_matches(candidate['checkpoint'])
    old=d.read(s['dataset_identity']['path'])
    assert all(k not in old['records'] for k in s['case_keys'])
    for cad,items in s['raw_sources'].items():
        assert cad in s['CADs']
        for item in items.values():d.stat_matches(item)
    return p
