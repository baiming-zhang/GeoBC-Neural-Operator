"""Metadata-only true predecessor binding; actual execution unreleased."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.official_uno_contract_v89 import inputs, PLAN, REVIEW, SOURCE_REVIEW, OUTPUT


def prepare():
    design, vb, bb = inputs(); sr = d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k) == h for k, h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    p = dict(kind='GeoBC_official_UNO_TRAIN9_v89', scientific=design['scientific'], scientific_digest=design['scientific_digest'],
        source_pins=sr['source_pins'], root_review=REVIEW, output=OUTPUT, prepared_at_utc=d.utc(),
        full_order_digest=design['scientific']['full_order_digest'])
    r = dict(passed=False, whole_source_reviewed=True, actual_execution_released=False, scientific_digest=p['scientific_digest'],
        source_pins=p['source_pins'], negative_binding=vb, baseline_binding=bb, source_review_identity=d.identity(SOURCE_REVIEW, hashed=True))
    d.write(PLAN, p); d.write(REVIEW, r)
    print(dict(status='metadata_prepared_actual0', SCI=p['scientific_digest'], GT_PT_arrays_GPU=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare: prepare()
    else: print('{"status":"source_only","GT_PT_arrays_GPU":0}')
