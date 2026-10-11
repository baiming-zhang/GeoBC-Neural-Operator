"""One complete Cartesian prediction barrier and shared known-DEV truth pass.

Callbacks allow a CPU negative-path check without opening real scientific data.
Production callbacks verify immutable files and are frozen with the evaluator.
"""


def verify_barrier(np, candidates, keys, node_counts, inventory, record_reader, array_reader, guard=lambda: None):
    expected = {c['name']+':'+key for c in candidates for key in keys}
    if len(expected) != len(candidates)*len(keys) or set(inventory) != expected:
        raise ValueError('Incomplete or duplicate Cartesian prediction inventory; GT remains closed')
    for candidate in candidates:
        for key in keys:
            guard()
            record = record_reader(inventory[candidate['name']+':'+key])
            if any(record[k] != value for k, value in (
                ('candidate', candidate['name']), ('key', key), ('family', candidate['family']),
                ('step', candidate['step']), ('checkpoint', candidate['checkpoint']),
                ('all_original_nodes', node_counts[key]), ('status', 'completed'),
                ('signed_stress_unchanged', True))):
                raise ValueError('Prediction identity/context differs; GT remains closed')
            for member in ('encoded', 'physical'):
                array = array_reader(record[member])
                if array.shape != (node_counts[key], 4) or str(array.dtype) != 'float32' or not np.isfinite(array).all():
                    raise ValueError('Incomplete/nonfinite physical4 or encoded4; GT remains closed')
                del array
    return dict(status='completed', expected=len(expected), finite_predictions=len(expected),
                all_cartesian_entries=True, owned_GT_reads_before_barrier=0, inventory=inventory)


def shared_truth_pass(np, candidates, keys, node_counts, inventory, record_reader, array_reader,
                      truth_reader, save_barrier, on_truth, metrics, guard=lambda: None):
    # Nothing may read numerical truth until *all* full original-node predictions
    # have been reopened and validated, and the barrier is durably committed.
    barrier = verify_barrier(np, candidates, keys, node_counts, inventory, record_reader, array_reader, guard)
    save_barrier(barrier)
    per = {c['name']: {key: None for key in keys} for c in candidates}
    for key in keys:
        guard()
        truth = truth_reader(key)
        on_truth(key)
        if truth.shape != (node_counts[key], 4) or str(truth.dtype) != 'float32' or not np.isfinite(truth).all():
            raise ValueError('Incomplete/nonfinite signed physical truth')
        for candidate in candidates:
            guard()
            record = record_reader(inventory[candidate['name']+':'+key])
            prediction = array_reader(record['physical'])
            per[candidate['name']][key] = metrics(np, truth, prediction)
            del prediction
        del truth
    return per
