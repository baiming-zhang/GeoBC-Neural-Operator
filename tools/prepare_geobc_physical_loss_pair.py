"""Freeze a new fixed-case objective ablation; existing experiments stay immutable."""
import argparse
import ast
import copy
import json
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding


def prepare():
    base = d.read('notes/geobc_single_case_fit_plan.json')
    root = d.read(base['root_review'])
    folder = d.path(base['output'])
    binding = {k: d.identity(folder/file, hashed=True) for k, file in
               [('summary', 'summary.json'), ('external', 'external_terminal.json')]}
    baseline = terminal_binding(binding)
    assert baseline['fixed4800'] and baseline['step'] == 4800
    assert baseline['first_actual_fullN_gradient_check']['passed']
    assert baseline['selected'] == root['selected']
    assert all(d.sha(p) == h for p, h in base['source_pins'].items())
    output = 'runs/geobc_exploration/single_case_physical_loss4800_v1'
    plan_path = 'notes/geobc_physical_loss_pair_plan.json'
    assert not d.path(plan_path).exists() and not (d.path(output)/'execution.json').exists()
    science = copy.deepcopy(base['scientific'])
    science.update(question='Does physical U/S relative-squared loss improve the same fixed TRAIN-case fit?',
        baseline_digest=base['scientific_digest'], fixed_selected=baseline['selected'],
        loss='0.5 * (whole-case physical displacement relative L2 squared + signed stress relative L2 squared)',
        objective_norms='Complete original physical TRAIN truth, FP64 sums; reject zero, no epsilon',
        unchanged_controls=['CommonGridSolver', 'fresh seed2026', 'old TRAIN40 scaler',
                            'same case', 'Adam .001/fused/no weight decay', '4800 updates',
                            'highest FP32 backbone and decoder', 'readouts0/40/4800', 'checkpoints every120'],
        objective_precision='FP64 pointwise decode and residual only',
        training_budget_seconds=900, whole_budget_seconds=1200,
        future_decision='One TRAIN case diagnoses objective; no final solver or generalization claim')
    # The baseline source is copied with explicit, small changes. It is not edited.
    source = d.path('tools/train_geobc_single_case_diagnostic.py').read_text(encoding='utf-8')
    source = source.replace('One fresh single-TRAIN-case fit, following the completed matched pair.',
                            'One fresh fixed-case physical-objective ablation against the completed encoded-MSE baseline.')
    source = source.replace("PLAN = 'notes/geobc_single_case_fit_plan.json'", "PLAN = 'notes/geobc_physical_loss_pair_plan.json'")
    source = source.replace("    assert selected == review['selected']", "    assert selected == review['selected'] == science['fixed_selected']\n    baseline = terminal_binding(review['baseline_binding'])\n    assert baseline['execution']['scientific_digest'] == science['baseline_digest']\n    assert baseline['fixed4800'] and baseline['step'] == 4800")
    source = source.replace('GEOBC_SINGLE_CASE_GUARD', 'GEOBC_PHYSICAL_LOSS_GUARD')
    source = source.replace('training_start[0]+18000', "training_start[0]+science['training_budget_seconds']")
    source = source.replace('training_start[1]+18000', "training_start[1]+science['training_budget_seconds']")
    source = source.replace('Original single-case diagnostic deadline', 'Original physical-objective deadline')
    source = source.replace('CommonGridSolver, full_case_backward, parameter_gradients', 'CommonGridSolver, parameter_gradients')
    source = source.replace('        from geobc.benchmark import FieldScaler',
        '        from geobc.benchmark import FieldScaler\n        from geobc.physical_field_loss import PhysicalScale, norms_from_complete_physical_target, full_common_case_backward')
    source = source.replace("        result['cached_TRAIN_physical_reads'] = 1", "        result['cached_TRAIN_physical_reads'] = 1\n        physical_target = torch.as_tensor(np.array(truth, copy=True), device='cuda')\n        physical_scale = PhysicalScale.from_mapping(science['scaler'])\n        norms = norms_from_complete_physical_target(physical_target)\n        result['objective'] = science['loss']\n        result['physical_norms'] = dict(node_count=norms.node_count, U_squared=norms.displacement_squared_norm, S_squared=norms.stress_squared_norm)")
    source = source.replace("            random.setstate(saved['python']); np.random.set_state(saved['numpy'])", "            if step == 0:\n                result['initial_physical_metrics_exactly_match_baseline'] = (result['endpoints']['0']['physical'] == baseline['endpoints']['0']['physical'])\n                assert result['initial_physical_metrics_exactly_match_baseline'], 'Initial scalar readout differs; do not call this a controlled loss pair'\n            random.setstate(saved['python']); np.random.set_state(saved['numpy'])")
    source = source.replace('loss, leaves = full_case_backward(model, case, guard)', 'loss, leaves = full_common_case_backward(model, case, physical_target, physical_scale, norms, guard)')
    source = source.replace("                    assert passed, 'FullN gradient check failed; exact gradient/leaf/query-coverage diagnostics retained in summary'", "                    result['first_actual_fullN_gradient_check']['peak_allocated_bytes_before_update'] = torch.cuda.max_memory_allocated()\n                    result['first_actual_fullN_gradient_check']['peak_reserved_bytes_before_update'] = torch.cuda.max_memory_reserved()\n                    assert max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) <= 24*2**30\n                    assert passed, 'FullN physical-loss gradient check failed; retain failure without parameter rescue'")
    source = source.replace("'encoded_MSE': loss", "'physical_relative_squared_objective': loss")
    source = source.replace("exposure_limit='40 same-case exposures vs shared 40;4800 same-total updates vs shared4800. These controls differ in optimizer history and case exposure.'", "exposure_limit='New physical objective vs completed encoded MSE: same fixed case, fresh initialization, optimizer and 4800 case exposures; one known TRAIN case only.'")
    driver = 'tools/train_geobc_physical_loss_case.py'
    d.path(driver).write_text(source, encoding='utf-8')
    guard = d.path('tools/run_geobc_single_case_guard.py').read_text(encoding='utf-8')
    guard = guard.replace('original20000s', 'original1200s').replace('20000', '1200')
    guard = guard.replace('geobc_single_case_fit_plan', 'geobc_physical_loss_pair_plan')
    guard = guard.replace('GEOBC_SINGLE_CASE_GUARD', 'GEOBC_PHYSICAL_LOSS_GUARD')
    guard = guard.replace('tools.train_geobc_single_case_diagnostic', 'tools.train_geobc_physical_loss_case')
    guard = guard.replace("    assert select_case(parent, development) == review['selected']", "    assert select_case(parent, development) == review['selected'] == plan['scientific']['fixed_selected']\n    baseline = terminal_binding(review['baseline_binding'])\n    assert baseline['fixed4800'] and baseline['step'] == 4800\n    assert baseline['execution']['scientific_digest'] == plan['scientific']['baseline_digest']")
    guard_path = 'tools/run_geobc_physical_loss_guard.py'
    d.path(guard_path).write_text(guard, encoding='utf-8')
    for p in (driver, guard_path): ast.parse(d.path(p).read_text(encoding='utf-8'))
    pins = dict(base['source_pins'])
    for p in (driver, guard_path, 'geobc/physical_field_loss.py', 'tools/prepare_geobc_physical_loss_pair.py'):
        pins[p] = d.sha(p)
    plan = dict(scientific=science, scientific_digest=d.digest(science), source_pins=pins,
                output=output, root_review='notes/geobc_physical_loss_pair_root_review.json')
    review = dict(passed=False, whole_SOURCE_reviewed=False, actual_execution_released=False,
        scientific_digest=plan['scientific_digest'], source_pins=pins,
        upstream_bindings=root['upstream_bindings'], selected=root['selected'], baseline_binding=binding,
        preparation_epoch=time.time(), preparation_GT_PT_model_GPU_reads=0)
    d.write(plan_path, plan); d.write(plan['root_review'], review)
    print(json.dumps({'status':'source_prepared_not_released', 'scientific_digest':plan['scientific_digest'],
                      'case':baseline['selected']['case_key'], 'model':baseline['selected']['arm']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare: prepare()
