"""Explicit metadata-only preparation of one fixed knownDEV60 readout."""
import argparse

PLAN='notes/geobc_global_hash10_readout_v63_plan.json'
REVIEW='notes/geobc_global_hash10_readout_v63_root_review.json'
SOURCE_REVIEW='notes/geobc_global_hash10_readout_v63_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash10_readout_contract_v63 import training_binding,SOURCES,METHOD,TRAIN
    up=training_binding();sr=d.read(SOURCE_REVIEW);old=up['known']
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:old[k]for k in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity')}
    s.update(whole_budget_seconds=600,training_SCI=up['plan']['scientific_digest'],
        candidates=[dict(name=METHOD,family='FNO_GLOBAL_HASH',step=3000,training_plan_identity=d.identity(TRAIN,hashed=True))],
        fixed_prediction_denominator=60,FINAL_reuse_for_selection=False,automatic_continuation=False,
        reference_bindings=up['reference_bindings'],Hash5_name=up['Hash5_name'],
        DEV_gate=dict(primary_absolute_improvement_at_least=.03,maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60),
        release_scope='One actualHash10 knownDEV60 fullnode readout after TRAIN gate; existing Hash5/FNO10/FNO20 scalar metrics only',
        limits=['KnownDEV already used iteratively; single seed and availability-conditional, not final blindtest',
            'Hash/FNO capacities and precision differ; FNO20 exposure differs; report absolute metrics and actual costs',
            'All signed physical U(mm)/S(MPa), original nodes; no independent offnode FEM accuracy claim'])
    p=dict(kind='GeoBC_globalHash_physical10_knownDEV60_v63',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/GlobalHash100_knownDEV60_physical10_v63',prepared_at_utc=d.utc())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        training_bindings={METHOD:up['binding']},reference_bindings=up['reference_bindings'],source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='prepared_not_released',SCI=p['scientific_digest'],GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
