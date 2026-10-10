"""Freeze a diagnostic selection rule without arrays, PT, models or GPU."""
import ast
from geobc import available_cad_pilot_data as d


def run():
    training = d.read('notes/geobc_common_grid_train_plan.json')
    development = d.read('notes/geobc_common_grid_readout_plan.json')
    science = {
        'question': 'Can the selected matched solver independently fit its worst known TRAIN case?',
        'parent_training_digest': training['scientific_digest'],
        'parent_development_digest': development['scientific_digest'],
        'original_TRAIN_plan': training['scientific']['parent_plan'],
        'TRAIN_keys': training['scientific']['TRAIN_keys'], 'scaler': training['scientific']['scaler'],
        'selection': 'After qualified common pair and development60, lowest complete development primary score; exact arm tie lexical. Require selected TRAIN120 mean S>15%. Highest selected-arm TRAIN-case signedS relativeL2, exact case tie lexical.',
        'initialization': 'Fresh seed2026 CommonGridSolver; same shared initial SHA as parent; no pretrained PT',
        'precision_optimizer_loss': 'Same highest FP32/TF32off, Adam constant0.001/noWD/fused, full-original-node joint encoded4 MSE',
        'fixed_endpoints': [0, 40, 4800], 'original_nodes': True, 'signed_units': ['mm','mm','mm','MPa'],
        'exposure_interpretation': '40 same-case exposures vs shared40,4800 same-total updates vs shared4800. Optimizer history and exposures differ; no one-factor causal attribution.',
        'resident_case': 'Load existing one TRAIN input/encoded target/physical cache once; no rawCSV or DEV labels. Faster IO is explicit and not a solver-speed comparison.',
        'readout_RNG': 'Same parent TRAIN index seed; restore all Python/NumPy/Torch/CUDA RNG after each evaluation',
        'checkpoint': 'Adam/RNG/cursor every120 plus fixed40/4800; single attempt no earlystop/replay',
        'training_cap_seconds': 18000, 'whole_cap_seconds': 20000, 'memory_bytes': 24*2**30,
        'training_clock': 'From after initialreadout0 through final4800 checkpoint writes; includes step40 readout and all checkpoint IO. Whole wall also includes final4800readout.',
        'GPU_lock': training['scientific']['GPU_lock'],
        'DEV_GT_raw_GT_scalerfit_pretrained_reads': 0, 'final_solver_selection_claim': False,
        'failure': 'Retain checkpoint and all clocks; no automatic recovery, budget reset, subset or endpoint replacement'}
    pins = dict(development['source_pins'])
    for file in ('tools/train_geobc_single_case_diagnostic.py', 'tools/run_geobc_single_case_guard.py',
                 'tools/release_geobc_single_case_fit.py', 'tools/prepare_geobc_single_case_source.py'):
        ast.parse(d.path(file).read_text(encoding='utf-8-sig')); pins[file] = d.sha(file)
    plan = {'scientific': science, 'scientific_digest': d.digest(science), 'source_pins': pins,
        'output': 'runs/geobc_exploration/matched_solver_single_case_fit_v1',
        'root_review': 'notes/geobc_single_case_fit_root_review.json', 'status': 'source_ready_actual0'}
    assert not d.path('notes/geobc_single_case_fit_plan.json').exists()
    d.write('notes/geobc_single_case_fit_plan.json', plan)
    d.write(plan['root_review'], {'passed': False, 'SOURCE_ready': True,
        'actual_execution_released': False, 'scientific_digest': plan['scientific_digest'],
        'source_pins': pins, 'upstream_bindings': None, 'selected': None,
        'actual_array_GT_PT_GPU_training': 0,
        'review_pending': 'Full source read and genuine future terminal bindings; no launch yet'})
    print('SOURCE ready; actual execution0; future completed pair selection required')


if __name__ == '__main__': run()
