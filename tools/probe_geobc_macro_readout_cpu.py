"""One CPU synthetic negative-path check; no real dataset/checkpoint/GT bodies."""
import argparse
import ast
import copy
import json
import math
import subprocess
import sys
import time


def probe():
    started = (time.time(), time.monotonic())
    from geobc import available_cad_pilot_data as d
    sources = ('geobc/macro_readout_contract.py', 'geobc/macro_readout_protocol.py',
               'tools/evaluate_geobc_macro_endpoints.py', 'tools/run_geobc_macro_readout_guard.py',
               'tools/probe_geobc_macro_readout_cpu.py')
    for name in sources: ast.parse(d.path(name).read_text(encoding='utf-8'))
    defaults = []
    for module in ('tools.evaluate_geobc_macro_endpoints', 'tools.run_geobc_macro_readout_guard'):
        actual = subprocess.run([sys.executable, '-u', '-m', module], cwd=d.ROOT,
            capture_output=True, text=True, timeout=20, check=True)
        value = json.loads(actual.stdout.strip())
        assert value == dict(status='source_only', job_reads=0, writes=0, GPU_execution=0)
        defaults.append(value)
    from geobc import macro_readout_contract as contract
    # An explicitly synthetic unapproved record exercises admission *before*
    # NumPy/Torch import; it is never written as a scientific plan or release.
    before = ('torch' in sys.modules, 'numpy' in sys.modules)
    old_read = d.read
    fake = dict(kind=contract.KIND, scientific={}, root_review='fixture-only')
    d.read = lambda path: dict(passed=False) if path == 'fixture-only' else fake
    denied = False
    try:
        contract.approved_contract('not-an-actual-plan')
    except AssertionError:
        denied = True
    finally:
        d.read = old_read
    assert denied and before == (False, False) and not {'torch', 'numpy'} & set(sys.modules)
    fixture_dir = d.path('runs/geobc_exploration/macro_readout_CPU_identity_'+str(time.time_ns()))
    fixture_dir.mkdir()
    fixture_file = fixture_dir/'fixture.txt'
    try:
        fixture_file.write_text('synthetic identity check only', encoding='utf-8')
        item = d.identity(fixture_file, hashed=True)
        assert contract.check_identity(item) == fixture_file
        assert contract.check_identity(dict(item, path=str(fixture_file))) == fixture_file
        altered_identity_rejected = False
        try:
            contract.check_identity(dict(item, sha256='0'*64))
        except AssertionError:
            altered_identity_rejected = True
        assert altered_identity_rejected
    finally:
        fixture_file.unlink(); fixture_dir.rmdir()
    import numpy as np
    from geobc.macro_readout_protocol import shared_truth_pass
    from geobc.coverage_comparison_train import physical_metrics, aggregate
    keys = [str(cad)+'/load'+str(load) for cad in range(20) for load in range(3)]
    counts = {key: 2 for key in keys}
    candidates = [dict(name=name, family=family, step=12000, checkpoint={'fixture': name})
        for name, family in [('A', 'FNO9'), ('B', 'PP256_NATIVE')]]
    truth = np.asarray([[1, -2, 3, -4], [-2, 1, -3, 2]], dtype=np.float32)
    arrays = {}; records = {}; inventory = {}
    for c in candidates:
        for key in keys:
            token = c['name']+':'+key; inventory[token] = token
            records[token] = dict(candidate=c['name'], family=c['family'], key=key, step=c['step'],
                checkpoint=c['checkpoint'], all_original_nodes=2, status='completed',
                signed_stress_unchanged=True, encoded=token+':encoded', physical=token+':physical')
            value = truth*(1.1 if c['name'] == 'A' else 1.2)
            arrays[token+':encoded'] = value.copy(); arrays[token+':physical'] = value.copy()

    def evaluate(test_inventory, test_records, test_arrays, zero_stress=False):
        events = []; reads = []; seen = set()
        def read_array(item): seen.add(item); return test_arrays[item]
        def save(barrier):
            assert len(seen) == 240 and not reads and barrier['expected'] == 120
            events.append('barrier_durably_saved_fixture')
        def read_truth(key):
            assert events and key not in reads
            reads.append(key)
            value = truth.copy()
            if zero_stress: value[:, 3] = 0
            return value
        try:
            per = shared_truth_pass(np, candidates, keys, counts, test_inventory,
                lambda item: test_records[item], read_array, read_truth, save,
                lambda key: None, physical_metrics)
            return per, reads, events, None
        except ValueError as error:
            return None, reads, events, str(error)

    successful, reads, events, error = evaluate(inventory, records, arrays)
    assert error is None and reads == keys and len(events) == 1
    measured = {}
    for name, expected in [('A', .1), ('B', .2)]:
        metric = aggregate(successful[name], 60); contract.complete_physical(metric, keys)
        assert all(math.isclose(metric[field], expected, abs_tol=2e-7) for field in
            ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2'))
        measured[name] = {k: metric[k] for k in ('selection_score', 'fixed_denominator', 'all_cases_complete')}
    negatives = []
    missing = dict(inventory); missing.pop(next(iter(missing)))
    altered = copy.deepcopy(records); altered[next(iter(altered))]['step'] = 11999
    nonfinite = dict(arrays); token = next(iter(nonfinite)); nonfinite[token] = nonfinite[token].copy(); nonfinite[token][0, 0] = np.nan
    for name, inv, rec, arr in [('missing_prediction', missing, records, arrays),
                              ('endpoint_identity_mismatch', inventory, altered, arrays),
                              ('nonfinite_encoded_prediction', inventory, records, nonfinite)]:
        per, reads, events, error = evaluate(inv, rec, arr)
        assert per is None and error and reads == events == []
        negatives.append(dict(case=name, truth_callback_calls=0, error=error))
    zero, reads, events, error = evaluate(inventory, records, arrays, zero_stress=True)
    assert error is None and len(reads) == 60
    zero_metric = aggregate(zero['A'], 60)
    assert zero_metric['selection_score'] is None and not zero_metric['all_cases_complete']
    try:
        contract.complete_physical(zero_metric, keys)
    except AssertionError:
        zero_rejected = True
    else:
        zero_rejected = False
    assert zero_rejected
    report = dict(status='completed_CPU_synthetic_protocol_check',
        elapsed_epoch_seconds=time.time()-started[0], elapsed_monotonic_seconds=time.monotonic()-started[1],
        source_pins={p: d.sha(p) for p in sources}, default_entry_checks=defaults,
        unreleased_admission_precedes_Torch_NumPy=True, synthetic_candidates=2, synthetic_cases=60,
        synthetic_absolute_and_relative_file_identities_equivalent=True, synthetic_changed_hash_rejected=True,
        synthetic_predictions=120, all_reopened_arrays_before_truth_callback=240,
        successful_synthetic_truth_calls=60, physical_metrics_expected_case_errors=measured,
        negative_paths=negatives, zero_norm_keeps_full_denominator_and_blocks_ranking=zero_rejected,
        real_input_GT_PT_array_reads=0, real_model_CUDA_training_or_inference=0,
        real_checkpoint_memory_timeout_runtime_qualified=False, new_scientific_plan_or_release_created=False)
    d.write('notes/geobc_macro_readout_CPU_probe.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'elapsed_epoch_seconds', 'elapsed_monotonic_seconds',
        'real_input_GT_PT_array_reads', 'real_model_CUDA_training_or_inference')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true')
    if parser.parse_args().probe: probe()
    else: print('{"status":"source_only","reads":0,"writes":0,"GPU_execution":0}')
