"""Bind genuine completed upstream scalar receipts; no models or arrays."""
import argparse
import json
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding, select_case


def run():
    started = time.monotonic()
    plan = d.read('notes/geobc_single_case_fit_plan.json'); review = d.read(plan['root_review'])
    assert review['passed'] and review['whole_SOURCE_reviewed']
    assert not review['actual_execution_released']
    assert review['source_pins'] == plan['source_pins']
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert not (d.path(plan['output'])/'execution.json').exists()
    bindings = {}; upstream = {}
    for name, path in [('training','notes/geobc_common_grid_train_plan.json'),
                       ('development','notes/geobc_common_grid_readout_plan.json')]:
        parent = d.read(path); assert parent['scientific_digest'] == plan['scientific']['parent_'+name+'_digest']
        assert all(d.sha(p) == h for p, h in parent['source_pins'].items())
        folder = d.path(parent['output'])
        binding = {k:d.identity(folder/file, hashed=True) for k,file in
                   [('summary','summary.json'),('external','external_terminal.json')]}
        upstream[name] = terminal_binding(binding)
        assert upstream[name]['execution']['scientific_digest'] == parent['scientific_digest']
        bindings[name] = binding
    selected = select_case(upstream['training'], upstream['development'])
    assert selected['case_key'] in plan['scientific']['TRAIN_keys']
    review.update(upstream_bindings=bindings, selected=selected, actual_execution_released=True,
                  release_elapsed_seconds=time.monotonic()-started, release_epoch=time.time(),
                  release_array_GT_PT_GPU_reads=0)
    d.write(plan['root_review'], review)
    print(json.dumps({'status':'future_single_case_released','arm':selected['arm'],
                      'case_key':selected['case_key'],'new_training_or_GT':0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--release', action='store_true')
    if parser.parse_args().release: run()
