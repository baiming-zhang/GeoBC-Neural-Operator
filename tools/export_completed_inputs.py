"""Scalar report for the fixed 100-CAD input set assembled across numerical stages."""
import argparse
import csv
import json
from pathlib import Path


def export(folder, catalog_path, destination):
    folder, destination = Path(folder), Path(destination)
    state = json.loads((folder/'summary.json').read_text())
    terminal = json.loads((folder/'terminal.json').read_text())
    catalog = json.loads(Path(catalog_path).read_text())
    if state['status'] != 'completed' or state['active'] or not state['closed'] or terminal['exit_code'] != 0 or not terminal['within_budget']:
        raise ValueError('Production is not completed')
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for cad in catalog['CADs']:
        field = state['feature_inventory'].get(cad)
        new = state['case_records'].get(cad, {})
        rows.append({'CAD': cad, 'input_ready': field is not None, 'nodes': field['shape'][1] if field else None,
                     'reused_completed_input': cad in catalog['existing_features'],
                     'new_solve_seconds': new.get('solve_seconds'), 'new_gradient_seconds': new.get('gradient_seconds'),
                     'independent_derivative_relative': new.get('derivative', {}).get('relative'),
                     'method': 'existing accepted inputs' if cad in catalog['existing_features'] else 'validated Chebyshev candidate'})
    with (destination/'inputs.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    max_residual = max([v['true_relative_residual'] for row in state['case_records'].values() for v in row.get('endpoints', [])], default=None)
    metrics = {'status': 'completed', 'complete_inputs': state['complete_inputs'], 'CAD_denominator': 100,
               'canonical_load_denominator': 300, 'complete_CADs': len(state['feature_inventory']),
               'reused_input_CADs': 95, 'new_feature_CADs': state['completed_new_feature_CADs'],
               'new_zero_fields': state['new_zero_fields'], 'reused_comparison_fields': 3,
               'new_true_residual_max': max_residual,
               'new_stage_UTC_seconds': terminal['elapsed_seconds']['UTC'],
               'new_stage_monotonic_seconds': terminal['elapsed_seconds']['monotonic'],
               'combined_new_solve_seconds': state['combined_solve_seconds'], 'new_gradient_seconds': state['gradient_seconds'],
               'prior_production_seconds': catalog['original_production_seconds'],
               'paired_comparison_campaign_UTC_seconds': catalog['comparison_campaign_UTC_seconds'],
               'GT_reads': 0, 'training_started': False}
    (destination/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n', encoding='utf-8')
    (destination/'README.md').write_text('# Complete training physical inputs\n\n'
        f"Fixed-set input completion: {len(state['feature_inventory'])}/100 CADs, three canonical loads each. "
        f"The new stage produced {state['new_zero_fields']} independent zero-start fields and five full-node gradient inputs. "
        'It reused 95 previously qualified input arrays and three fields from the passed Chebyshev comparison.\n\n'
        f"New-stage cost: UTC {metrics['new_stage_UTC_seconds']:.3f}s / monotonic {metrics['new_stage_monotonic_seconds']:.3f}s. "
        'The earlier budget-limited run and the paired comparison remain separate costs. The earlier 5000-second run was not reopened; the changed preconditioner was tested before producing the remaining fields.\n\n'
        'Features are physical U(mm) and nine incident-volume-averaged P2 displacement-gradient components multiplied by 1000. Original-node Jacobians, incident coverage, finite values and independent physical-coordinate derivatives are checked. '
        'This establishes predictor availability, not original signed-S accuracy. Development-input qualification and a paired U/S learning experiment still follow.\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(['Reused CAD inputs', 'New CAD inputs'], [95, state['completed_new_feature_CADs']], color=['#4477aa', '#228833'])
    axes[0].set(ylabel='CADs', title='Fixed 100-CAD input set')
    changed = [row for row in rows if row['independent_derivative_relative'] is not None]
    axes[1].bar([row['CAD'] for row in changed], [row['independent_derivative_relative'] for row in changed], color='#228833')
    axes[1].set(ylabel='Relative derivative difference', title='Five new full-node input checks')
    axes[1].tick_params(axis='x', rotation=25)
    fig.suptitle('Physical U / full-gradient predictor inputs; neural training pending')
    fig.savefig(destination/'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--output', default='results/complete_training_inputs')
    args = parser.parse_args()
    export(args.run, args.catalog, args.output)
