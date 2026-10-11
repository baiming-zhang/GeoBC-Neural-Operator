"""Future terminal-only metadata preparation; no checkpoint/array bodies."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.matched20_readout_contract_v79 import inputs,SOURCES,TRAIN,PLAN,REVIEW,SOURCE_REVIEW


def prepare():
    up=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items()) and set(SOURCES)<=set(sr['source_pins'])
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    known=up['known'];s={k:known[k]for k in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size','GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity')}
    s.update(whole_budget_seconds=900,fixed_prediction_denominator=120,training_SCI=up['plan']['scientific_digest'],
        candidates=[dict(name=x['name'],family=x['family'],step=6000,training_plan_identity=d.identity(TRAIN,hashed=True))for x in up['plan']['scientific']['arms']],
        reference_binding=up['reference_binding'],reference_SCI=up['reference_plan']['scientific_digest'],
        DEV_gate=up['plan']['scientific']['comparative_gate'],TRAIN_gate_filters_candidates=False,
        automatic_continuation=False,FINAL_reuse_for_selection=False,
        limits=['KnownDEV iterated/single seed/availability-conditional, not finalblind','Same20 exposure/scaler/dataset but capacity and precision differ','Parent failures/earlier costs retained; new phase costs are not full project lifetime'])
    p=dict(kind='GeoBC_matched20_Hash_TS_knownDEV60_v79',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/Hash_TS_matched20_knownDEV60_v79',prepared_at_utc=d.utc())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        training_bindings={name:up['binding']for name in up['qualified']},reference_binding=up['reference_binding'],source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),status='True training terminal bound; separate root execution release pending')
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='terminal_metadata_prepared_actual0',SCI=p['scientific_digest'],PT_bodies=0,arrays=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
