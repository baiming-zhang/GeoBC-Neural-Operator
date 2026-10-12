"""Export scalar paired costs and endpoint residuals from one completed run."""
import argparse
import csv
import json
from pathlib import Path


def export(folder, destination):
    folder, destination = Path(folder), Path(destination)
    read = lambda name: json.loads((folder/name).read_text(encoding='utf-8'))
    state, terminal = read('summary.json'), read('terminal.json')
    if state['status'] != 'completed' or state['active'] or not state['closed'] or terminal['exit_code'] != 0 or not terminal['within_budget']:
        raise ValueError('The comparison has not completed')
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for method in ['jacobi', 'chebyshev']:
        group = state['groups'][method]
        for endpoint in group['endpoints']:
            load = endpoint['load']; index = ['ver', 'hor', 'tor'].index(load)
            rows.append({'CAD': state['CAD'], 'method': method, 'load': load,
                         'nodes': endpoint['field']['shape'][0], 'iterations': group['solver']['iterations'][index],
                         'true_relative_residual': endpoint['true_relative_residual'],
                         'paired_group_solve_seconds': group['solve_seconds'],
                         'method_construction_checks_seconds': group['construction_checks_seconds'],
                         'same_solution_relative': state['same_solution_relative'][index]})
    if len(rows) != 6:
        raise ValueError('Incomplete six-field report')
    with (destination/'endpoints.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    baseline = state['groups']['jacobi']['solve_seconds']
    candidate = state['groups']['chebyshev']['solve_seconds']+state['groups']['chebyshev']['construction_checks_seconds']
    metrics = {'CAD': state['CAD'], 'fields': 6, 'precision_passed': state['precision_passed'],
               'candidate_passed': state['candidate_passed'], 'paired_cost_reduction': state['paired_cost_reduction'],
               'jacobi_seconds': baseline, 'chebyshev_construction_and_solve_seconds': candidate,
               'source_operator_seconds': state['source_operator_seconds'], 'shared_coarse_seconds': state['shared_coarse']['seconds'],
               'recovery_UTC_seconds': terminal['elapsed_seconds']['UTC'],
               'recovery_monotonic_seconds': terminal['elapsed_seconds']['monotonic'],
               'original_campaign_UTC_seconds': terminal['campaign_UTC_seconds'],
               'first_attempt_seconds': state.get('recovery', {}).get('prior_attempt_seconds'),
               'same_solution_relative': state['same_solution_relative'],
               'peak_allocated_bytes': state['peak_allocated_bytes'], 'peak_reserved_bytes': state['peak_reserved_bytes'],
               'GT_reads': 0}
    (destination/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    outcome = 'met' if metrics['candidate_passed'] else 'did not meet'
    (destination/'README.md').write_text('# Coarse preconditioner comparison\n\n'
        f"On fixed training CAD {state['CAD']}, the degree-32 Chebyshev candidate {outcome} the precision and 25% cost-reduction criteria. "
        f"Paired Jacobi solve: {baseline:.3f}s; Chebyshev construction/checks plus solve: {candidate:.3f}s "
        f"({100*metrics['paired_cost_reduction']:.2f}% reduction).\n\n"
        'Both methods use the same FP64 curved-P2 equation and three independent zero-start loads. '
        'Shared geometry and coarse assembly costs are separate from the paired comparison. Six fields were saved before the same-equation displacement comparison; no target labels were read.\n\n'
        'The first attempt stopped after a statistics-key error, before any PCG field. Its cost and the unsaved coarse reconstruction remain included in campaign accounting; saved stiffness was reused. '
        'This single-CAD result does not establish full-training-set cost or original stress accuracy.\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].bar(['Jacobi', 'Chebyshev'], [baseline, candidate], color=['#4477aa', '#228833'])
    axes[0].set(ylabel='Paired group seconds', title='Three independent loads')
    for method, color in [('jacobi', '#4477aa'), ('chebyshev', '#228833')]:
        values = [row['true_relative_residual'] for row in rows if row['method'] == method]
        axes[1].plot(['ver', 'hor', 'tor'], values, 'o-', label=method, color=color)
    axes[1].axhline(1e-8, color='gray', ls='--')
    axes[1].set(yscale='log', ylabel='True relative residual', title='Same equation, FP64')
    axes[1].legend()
    fig.suptitle('Coarse polynomial comparison: '+state['CAD'])
    fig.savefig(destination/'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', default='results/coarse_preconditioners')
    args = parser.parse_args()
    export(args.run, args.output)
