"""Terminal-only metadata binding; no checkpoint/array bodies."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.solver_seed_readout_contract_v83 import inputs, science_for, SOURCES, PLAN, REVIEW, SOURCE_REVIEW, OUTPUT


def prepare():
    up = inputs(); sr = d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES) <= set(sr['source_pins']) and all(d.sha(k) == h for k, h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s = science_for(up)
    p = dict(kind='GeoBC_paired_seeds_knownDEV60_v83', scientific=s, scientific_digest=d.digest(s),
        source_pins=sr['source_pins'], root_review=REVIEW, output=OUTPUT, prepared_at_utc=d.utc())
    r = dict(passed=False, whole_source_reviewed=True, actual_execution_released=False,
        scientific_digest=p['scientific_digest'], source_pins=p['source_pins'], training_binding=up['binding'],
        reference_binding=up['reference_binding'], source_review_identity=d.identity(SOURCE_REVIEW, hashed=True),
        status='True all4 training terminal bound; separate root execution release pending')
    d.write(PLAN, p); d.write(REVIEW, r)
    print(dict(status='terminal_metadata_prepared_actual0', SCI=p['scientific_digest'], PT_bodies=0, arrays=0, GPU=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        print('{"status":"source_only","scientific_IO":0,"GPU":0}')
