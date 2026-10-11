"""Prepare immutable metadata; a separate true root release remains required."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.nativeTS100_contract_v80 import inputs, PLAN, REVIEW, SOURCE_REVIEW, DESIGN
from geobc.macro_training import balanced_schedule


def prepare():
    up = inputs(); sr = d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k) == h for k, h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s = dict(up['design']['scientific'])
    s.update({k: up['dense_plan']['scientific'][k] for k in
        ('optimizer', 'loss', 'expected_FP32_TS_initialization', 'official_TS_kwargs')})
    s.update(full_order_digest=d.digest(balanced_schedule(s['TRAIN_keys'], 10, 2026)),
        qualification_TRAIN_key=up['witness'], reference_TRAIN5=up['dense_summary']['models']['TS_ORIGINAL_BF16_GLOBAL_HASH']['endpoints']['5'],
        reference_TRAIN10=up['dense10'], gate=dict(primary_improvement_min=.03,
            stress_improvement_min=.03, field_deterioration_max=.02, hotspot_deterioration_max=.02))
    p = dict(kind='GeoBC_native_originalTS100_matched5_10_v80', scientific=s,
        scientific_digest=d.digest(s), source_pins=sr['source_pins'], root_review=REVIEW,
        design_identity=d.identity(DESIGN, hashed=True),
        output='runs/geobc_exploration/Native_originalTS100_matched10_v80', prepared_at_utc=d.utc())
    r = dict(passed=False, whole_source_reviewed=True, actual_execution_released=False,
        scientific_digest=p['scientific_digest'], source_pins=p['source_pins'],
        upstream_terminal_bindings=up['bindings'], dense5_binding=up['dense_binding'],
        source_review_identity=d.identity(SOURCE_REVIEW, hashed=True))
    d.write(PLAN, p); d.write(REVIEW, r)
    print(dict(status='metadata_prepared_actual0', SCI=p['scientific_digest'], PT_arrays_GPU=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        print('{"status":"source_only","scientific_bodies":0,"GPU":0}')
