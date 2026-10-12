"""Summarize a completed physical-input production run without reading arrays."""
import argparse
import csv
import json
from pathlib import Path

def export(summary_path, plan_path, terminal_path, output):
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
    run, plan, terminal = read(summary_path), read(plan_path), read(terminal_path)
    if run['status'] != 'completed' or run['active'] or not run['closed'] or terminal['exit_code'] != 0:
        raise ValueError('Production has not completed')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    cads = plan['scientific']['selection']['CADs']
    rows = []
    for cad in cads:
        case = run['case_records'].get(cad, {})
        feature = run['feature_inventory'].get(cad)
        group = run['completed_groups'].get(cad)
        derivative = feature['independent_derivative_qualification'] if feature else {}
        rows.append({'CAD': cad, 'nodes': case.get('actual_allN'), 'cells': case.get('actual_cells'),
                     'inputs_completed': feature is not None, 'new_loads_solved': len(group['loads']) if group else None,
                     'reused_vertical': cad in plan['scientific']['legacy'],
                     'cache_operator_seconds': case.get('cache_operator_seconds'),
                     'coarse_seconds': case.get('coarse_seconds'), 'solve_seconds': case.get('actual_solve_seconds'),
                     'gradient_seconds': case.get('feature_seconds'), 'derivative_relative_error': derivative.get('relative')})
    with (destination/'inputs.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {'status': 'completed', 'complete_inputs': run['complete_input_gate_passed'], 'CADs': len(cads),
               'load_denominator': 3*len(cads), 'completed_input_CADs': run['committed_feature_CADs'],
               'saved_new_endpoints': run['new_zero_endpoint_count'], 'reused_endpoints': run['reused_VER_endpoint_count'],
               'whole_utc_seconds': terminal['elapsed_epoch_seconds'],
               'whole_monotonic_seconds': terminal['elapsed_monotonic_seconds'],
               'solve_seconds': run['combined_solve_seconds'], 'gradient_seconds': run['feature_seconds'],
               'peak_allocated_bytes': run['peak_allocated_bytes'], 'peak_reserved_bytes': run['peak_reserved_bytes'],
               'training_started': False}
    (destination/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    last = next(reversed(run['completed_groups']), None)
    not_started = len(cads)-len(run['case_records'])
    reason = ('The fixed input set is complete.' if metrics['complete_inputs'] else
              f'The cumulative solve budget ended while CAD {last} was being solved. Its finite endpoints did not meet the residual target; {not_started} later CADs were not started. The original {len(cads)}-CAD denominator is retained, and training remains pending.')
    (destination/'README.md').write_text('# Training physical inputs\n\n'
        f"Completed inputs: {metrics['completed_input_CADs']}/{len(cads)} CADs. "
        f"Saved new endpoints: {metrics['saved_new_endpoints']}; reused vertical endpoints: {metrics['reused_endpoints']}.\n\n"
        +reason+'\n\n'
        f"Whole UTC time {metrics['whole_utc_seconds']:.2f}s; monotonic time {metrics['whole_monotonic_seconds']:.2f}s. "
        'The two clocks are reported separately. Inputs are U(mm) and nine full displacement-gradient components scaled by 1000. This stage reads no target labels.\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = range(len(rows))
    axes[0].plot(x, [r['solve_seconds'] if r['solve_seconds'] is not None else float('nan') for r in rows])
    axes[0].set(xlabel='Fixed training CAD index', ylabel='Solve seconds', title='Independent physical solves')
    axes[1].scatter([i for i, r in enumerate(rows) if r['derivative_relative_error'] is not None],
                    [r['derivative_relative_error'] for r in rows if r['derivative_relative_error'] is not None], s=12)
    axes[1].set(xlabel='Fixed training CAD index', ylabel='Relative derivative difference', title='Physical-gradient checks')
    figure.suptitle(f"Physical inputs: {metrics['completed_input_CADs']}/{len(cads)} CADs; training pending")
    figure.savefig(destination/'comparison.png', dpi=160)
    plt.close(figure)
    print(json.dumps(metrics))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summary')
    parser.add_argument('plan')
    parser.add_argument('terminal')
    parser.add_argument('--output', default='results/training_physical_inputs')
    args = parser.parse_args()
    export(args.summary, args.plan, args.terminal, args.output)
