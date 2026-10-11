"""Actual terminal-bound metadata preparation, separate root release required."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.solver_seed_contract_v81 import inputs, PLAN, REVIEW, DESIGN, SOURCE_REVIEW
from geobc.macro_training import balanced_schedule


def prepare():
    up = inputs(); sr = d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k) == h for k, h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s = dict(up['design']['scientific']); s['qualification_TRAIN_key'] = up['witness']
    orders = {str(seed): d.digest(balanced_schedule(s['TRAIN_keys'], 20, seed)) for seed in s['fresh_seeds']}
    p = dict(kind='GeoBC_FNO_decoder_packages_paired_seeds_TRAIN100_v81', scientific=s,
        scientific_digest=d.digest(s), full_order_digests=orders, source_pins=sr['source_pins'],
        root_review=REVIEW, design_identity=d.identity(DESIGN, hashed=True),
        output='runs/geobc_exploration/FNO_Hash_paired_seeds_TRAIN100_v81', prepared_at_utc=d.utc())
    r = dict(passed=False, whole_source_reviewed=True, actual_execution_released=False,
        scientific_digest=p['scientific_digest'], source_pins=p['source_pins'],
        upstream_terminal_bindings=up['bindings'], source_review_identity=d.identity(SOURCE_REVIEW, hashed=True))
    d.write(PLAN, p); d.write(REVIEW, r)
    print(dict(status='metadata_prepared_actual0', SCI=p['scientific_digest'], PT_arrays_GPU=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        print('{"status":"source_only","scientific_bodies":0,"GPU":0}')
