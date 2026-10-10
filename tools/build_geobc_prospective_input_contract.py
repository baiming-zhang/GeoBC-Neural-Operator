"""Bind the reserved CAD split to a target-free input recipe, metadata only.

This command never calls the original CUDA producer or opens raw input bytes.
It does not release preparation, target parsing, model training or evaluation.
"""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from geobc import available_cad_pilot_data as d


SPLIT = 'notes/geobc_prospective_cad_split_v1.json'
OUTPUT = 'notes/geobc_prospective_target_free_input_contract.json'
OWN = 'tools/build_geobc_prospective_input_contract.py'


def build():
    began = time.monotonic()
    assert not d.path(OUTPUT).exists(), 'Existing input binding is immutable; do not rebuild silently'
    split = d.read(SPLIT)
    science = split['scientific']
    assert split['scientific_digest'] == d.digest(science)
    assert split['status'] == 'reserved_metadata_only'
    assert split['actual_FINAL_target_parses'] == split['actual_new_TRAIN_target_parses'] == 0
    old = science['old_TRAIN40_CADs']
    additional = science['additional_TRAIN60_CADs']
    dev = science['DEV_CADs']
    final = science['FINAL_CADs']
    assert [len(old), len(additional), len(dev), len(final)] == [40, 60, 20, 20]
    assert len(set(old + additional + dev + final)) == 140
    assert sorted(old + additional) == science['TRAIN_CADs']
    assert science['loads'] == ['ver', 'hor', 'tor']

    # Reuse the completed input-only DEV recipe, never its shared GT cache.
    previous_path = 'data/geobc_available_cad_pilot/manifest.json'
    previous = d.read(previous_path)
    assert previous['status'] == 'completed' and previous['feasibility_pass']
    basis = previous['contract']
    assert sorted(basis['selected_CADs']) == sorted(dev)
    assert basis['scaler'] == science['scaler']
    assert all(previous['read_receipt'][name] == 0 for name in d.ZERO_COUNTERS)

    new = sorted(additional + final)
    assert set(split['source_receipts']) == set(new)
    raw_sources = {}
    scientific_roles = {}
    for cad in new:
        scientific_roles[cad] = 'TRAIN_input_only' if cad in additional else 'FINAL_inference_input_only'
        raw_sources[cad] = {}
        assert set(split['source_receipts'][cad]) == {'FieldMesh', 'Field', 'SolverDeck'}
        for component, frozen in split['source_receipts'][cad].items():
            d.stat_matches(frozen)
            assert re.fullmatch('[0-9a-f]{64}', frozen['received_sha256'])
            # The helper expects sha256; this is an existing reception digest,
            # not an independently published upstream checksum or a new rehash.
            raw_sources[cad][component] = {
                **{name: frozen[name] for name in ('path', 'bytes', 'mtime_ns')},
                'sha256': frozen['received_sha256'],
                'checksum_origin': frozen['checksum_origin'],
                'source_role': 'official_TRAIN_pool',
                'prospective_role': scientific_roles[cad],
                'numeric_targets_permitted': False}

    recipe = dict(basis['input_recipe'])
    assert recipe['H5_members'] == ['vertices', 'cells']
    assert recipe['coordinate_columns'] == d.COORDS
    assert recipe['known_loads'] == science['loads']
    assert recipe['features'] == 25 and recipe['grid_resolution'] == 64
    assert recipe['grid_channels'] == 29 and recipe['cloud_points'] == 1024
    assert recipe['numeric_targets_before_barrier'] is False
    # All eighty enter the helper's input-only branch. This technical alias
    # cannot be used as a scientific train/validation membership declaration.
    helper = {
        'schema': 'GeoBC_prospective80_target_free_helper_contract_v1',
        'groups': {'train_new160': [], 'validation20': new},
        'selected_CADs': new, 'case_keys': d.keys(new),
        'scaler': science['scaler'], 'input_recipe': recipe,
        'raw_sources': raw_sources,
        'scientific_CAD_roles': scientific_roles,
        'helper_validation_is_technical_branch_only': True,
        'forbidden_helper_target_branch': 'groups.train_new160 must remain empty',
        'owned_future_output': 'data/geobc_prospective_target_free_inputs_v1',
        'fixed_CAD_denominator': 80, 'fixed_case_denominator': 240}
    source_names = [OWN, 'geobc/coverage_comparison_data.py',
                    'geobc/available_cad_pilot_data.py', 'geobc/solver_selection_data.py',
                    'geobc/blind_solver_inputs.py', 'geobc/elastic_pilot_data.py',
                    'tools/mesh_boundary.py', 'geobc/benchmark.py']
    contract_science = {
        'role': 'Future input-only preparation binding; not runtime preparation or an accuracy result',
        'split_scientific_digest': split['scientific_digest'],
        'split_identity': d.identity(SPLIT, hashed=True),
        'existing_DEV_input_manifest_identity': d.identity(previous_path, hashed=True),
        'TRAIN': science['TRAIN_CADs'], 'DEV': dev, 'FINAL': final,
        'helper_contract': helper,
        'old_TRAIN40_reuse': 'Existing original TRAIN120 input/scaler; no new scaler fit',
        'DEV20_reuse': 'Existing target-free input records only; no rereading GT for preparation',
        'source_pins': {name: d.sha(name) for name in source_names},
        'recipe_reference': d.read('notes/geobc_available_cad_pilot_plan.json')['recipe_reference'],
        'runtime_requirements': [
            'A separately frozen budget and one-shot guard must authorize any actual CUDA preparation',
            'The guard must wait for current training/readout/diagnostic GPU work to end and use the common lock',
            'Use target-free InputLedger, which rejects known3_U_S and all target counters',
            'Check every prepared record has exactly the input members and no physical_path/target_path',
            'Attach scientific_CAD_roles; do not treat the helper validation alias as DEV or TRAIN',
            'All fixed80 inputs must succeed or preserve a whole-input negative; no substitutions',
            'TRAIN numeric target preparation requires a separate TRAIN-only frozen stage',
            'FINAL numerical targets remain sealed until all selected full60 predictions or failures are submitted'],
        'limitations': [
            'Existing source stats and reception SHA bind files; raw bytes have not been reread here',
            'No runtime input shape, whole80 preparation, off-node accuracy or model result is proved',
            'Availability-conditional official TRAIN pool; not official TEST or demonstrated shape-family separation']}
    result = {
        'status': 'bound_metadata_only',
        'observed_UTC': datetime.now(timezone.utc).isoformat(),
        'scientific': contract_science, 'scientific_digest': d.digest(contract_science),
        'actual_source_stat_checks': 240,
        'actual_new_input_arrays_prepared': 0, 'actual_raw_byte_reads': 0,
        'actual_TRAIN_target_parses': 0, 'actual_DEV_target_parses': 0,
        'actual_FINAL_target_parses': 0, 'actual_scaler_fits': 0,
        'actual_training_or_GPU': 0, 'runtime_preparation_released': False,
        'metadata_seconds': time.monotonic()-began}
    d.write(OUTPUT, result)
    print(json.dumps({name: result[name] for name in ('status','metadata_seconds','actual_source_stat_checks',
                                                    'actual_FINAL_target_parses','runtime_preparation_released')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    if parser.parse_args().build:
        build()
