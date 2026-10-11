"""Metadata-only paired endpoint plan; neither numerical arrays nor PT bodies."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.fno40_pair_readout_contract_v73 import inputs, SOURCES, METHODS, TRAIN, PLAN, REVIEW

SOURCE_REVIEW = 'notes/geobc_fno40_pair_readout_v73_source_review.json'


def prepare():
    up = inputs()
    old, sr = up['known'], d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES) <= set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s = {k:old[k] for k in ('GPU_lock','memory_limit_bytes','seed','decoder_chunk_size',
        'GT_policy','pool','dataset_identity','case_keys','scaler','GT_manifest_identity')}
    s.update(whole_budget_seconds=600,training_SCI=up['plan']['scientific_digest'],
        candidates=[dict(name=name,family='FNO9',step=step,training_plan_identity=d.identity(TRAIN,hashed=True))
                    for step,name in METHODS.items()],fixed_prediction_denominator=120,
        FINAL_reuse_for_selection=False,automatic_continuation=False,TRAIN_gate_filters_candidates=False,
        reference_binding=up['reference_binding'],reference_SCI=up['reference_plan']['scientific_digest'],
        training_report_identity=d.identity('results/geobc_fno_learning40_v72/metrics.json',hashed=True),
        DEV_gate=dict(primary_absolute_improvement_at_least=.03,
            maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=60),
        release_scope='Both predeclared30/40 endpoints, prediction-first sharedGT60 once; zero updates; no automatic80',
        limits=['KnownDEV has been used in iteration; single seed; availability-conditional pool, not FINAL',
            'Different exposure and cost; not an equal-budget architecture ranking',
            'Full native signed U(mm)/S(MPa); independent offnode FEM accuracy not measured'])
    p = dict(kind='GeoBC_FNO30_and40_knownDEV60_paired_v73',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/FNO100_knownDEV60_physical30_40_v73',prepared_at_utc=d.utc())
    r = dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        training_bindings={name:up['binding'] for name in METHODS.values()},reference_binding=up['reference_binding'],
        source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),
        status='SOURCE reviewed; both actual endpoints retained; root release pending')
    d.write(PLAN,p)
    d.write(REVIEW,r)
    print(dict(status='prepared_not_runtime_released',SCI=p['scientific_digest'],PT_body_reads=0,GT_reads=0,GPU=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        print('{"status":"source_only","writes":0,"GPU_execution":0}')
