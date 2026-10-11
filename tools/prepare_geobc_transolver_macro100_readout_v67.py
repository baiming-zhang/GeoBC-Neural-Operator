"""Metadata-only preparation of completedTS endpoint on existing knownDEV60."""
import argparse

PLAN='notes/geobc_transolver_macro100_readout_v67_plan.json'
REVIEW='notes/geobc_transolver_macro100_readout_v67_root_review.json'
SOURCE_REVIEW='notes/geobc_transolver_macro100_readout_v67_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_macro100_readout_contract_v67 import training_binding,SOURCES,METHOD,TRAIN
    up=training_binding();sr=d.read(SOURCE_REVIEW);old=up['known']
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:old[k]for k in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity')}
    s.update(whole_budget_seconds=600,training_SCI=up['plan']['scientific_digest'],
        candidates=[dict(name=METHOD,family=up['family'],step=1500,training_plan_identity=d.identity(TRAIN,hashed=True))],
        fixed_prediction_denominator=60,FINAL_reuse_for_selection=False,automatic_continuation=False,
        reference_bindings=up['reference_bindings'],same5_reference_name=up['same5_reference_name'],
        DEV_gate=dict(primary_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60),
        release_scope='All finite completeTS1500 endpoint, even poorTRAIN;60 physical predictions committed/revalidated before oneGT pass. Seven immutable sameCAD references, no reference replay',
        limits=['KnownDEV already used iteratively, single seed and availability-conditional; not final blindtest',
            'Backbone capacity/precision/initialization/representation differ; some reference exposures10/20 differ fromfive; report physical values and actual costs',
            'No architecture ceiling, final solver, independent offnode FEM accuracy or automatic longtraining claim'])
    p=dict(kind='GeoBC_originalTS_BF16_continuous100_physical5_knownDEV60_v67',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/Transolver_BF16_continuous100_knownDEV60_physical5_v67',prepared_at_utc=d.utc())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        training_bindings={METHOD:up['binding']},reference_bindings=up['reference_bindings'],source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='prepared_not_released',SCI=p['scientific_digest'],GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
