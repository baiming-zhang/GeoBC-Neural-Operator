"""Metadata-only single endpoint readout plan; no numerical arrays or PT."""
import argparse
import datetime

PLAN='notes/geobc_fno20_readout_v57_plan.json'
REVIEW='notes/geobc_fno20_readout_v57_root_review.json'
SOURCE_REVIEW='notes/geobc_fno20_readout_v57_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.fno20_readout_contract_v57 import training_binding,SOURCES,METHOD
    up=training_binding();old=up['parents']['DEV_plan']['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert all(sr['source_pins'].get(k)==h for k,h in up['plan']['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:old[k] for k in ('GPU_lock','memory_limit_bytes','whole_budget_seconds','seed',
        'decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity')}
    s.update(training_SCI=up['plan']['scientific_digest'],
        candidates=[dict(name=METHOD,family='FNO9',step=6000,
            training_plan_identity=d.identity('notes/geobc_fno_learning20_v56_plan.json',hashed=True))],
        fixed_prediction_denominator=60,FINAL_reuse_for_selection=False,
        baseline_development_binding=up['parents']['DEV_binding'],
        baseline_development_SCI=up['parents']['DEV_plan']['scientific_digest'],
        training_report_metrics_identity=d.identity('results/geobc_fno_learning20_v56/metrics.json',hashed=True),
        DEV_gate=dict(primary_absolute_improvement_at_least=.03,
            maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60),
        release_scope='One knownDEV60 readout, zero training updates; no automatic continuation',
        limits=['KnownDEV used in iteration, single seed, availability-conditional CAD pool',
            'Exposure comparison within one trajectory, not equal-budget architecture comparison',
            'Original native signed U(mm)/S(MPa); no independent offnode FEM claim'])
    p=dict(kind='GeoBC_FNO_physical10_to20_knownDEV60_v57',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/FNO100_knownDEV60_physical20_v57',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        training_bindings={METHOD:up['binding']},source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),
        status='Prepared; genuine completed TRAIN20 passes gate; runtime awaits root release')
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_runtime_released',SCI=p['scientific_digest'],
        new_GT_array_reads=0,PT_body_reads=0,training_updates=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
