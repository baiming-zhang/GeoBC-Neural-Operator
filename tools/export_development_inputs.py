"""Report scalar development-input checks without loading scientific fields."""
import argparse
import csv
import json
from pathlib import Path


def export(run, output):
    run, output = Path(run), Path(output)
    state = json.loads((run/'summary.json').read_text())
    terminal = json.loads((run/'terminal.json').read_text())
    if not state['complete_inputs'] or state['active'] or not state['closed'] or terminal['exit_code'] != 0 or not terminal['within_budget']:
        raise ValueError('Development production is incomplete')
    output.mkdir(parents=True, exist_ok=False)
    rows = [dict(CAD=cad, nodes=row['nodes'], cells=row['cells'],
                 load_seconds=row['load_seconds'], gradient_seconds=row['gradient_seconds'],
                 derivative_relative=row['derivative']['relative'],
                 maximum_node_J_condition=row['qualification']['maximum_node_J_condition'])
            for cad, row in state['case_records'].items()]
    with (output/'inputs.csv').open('x', encoding='utf-8', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics=dict(status='completed', CADs=len(rows), canonical_loads=state['displacement_reads'],
                 maximum_nodes=max(row['nodes'] for row in rows),
                 maximum_derivative_relative=max(row['derivative_relative'] for row in rows),
                 stage_seconds=terminal['elapsed_seconds'], gradient_seconds=state['gradient_seconds'],
                 geometry_reads=state['source_reads'], weight_reads=state['weight_reads'],
                 peak_allocated_bytes=state['peak_allocated_bytes'], peak_reserved_bytes=state['peak_reserved_bytes'],
                 K_PCG_runs=0, GT_reads=0, training_started=False)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# Development physical inputs\n\n'
        f"All {len(rows)} known development CADs and 60 canonical fields now have physical U and full-gradient covariates. "
        f"The largest geometry has {metrics['maximum_nodes']:,} original nodes. "
        f"Maximum independent physical-coordinate derivative difference: {metrics['maximum_derivative_relative']:.3g}.\n\n"
        f"Actual cost: UTC {metrics['stage_seconds']['UTC']:.3f}s / monotonic {metrics['stage_seconds']['monotonic']:.3f}s. "
        'Existing converged U and positive curved-cell volumes were loaded once; no elasticity solve or ground-truth read was needed. '
        'Features use three U(mm) components and nine volume-averaged P2 gradients multiplied by 1000. '
        'Source and solve costs belong to the earlier displacement results. This is input qualification; signed-stress learning remains to be tested.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar([row['CAD'] for row in rows],[row['derivative_relative'] for row in rows],color='#228833')
    ax.set(title='All 20 development CADs: full-node physical input recovery',ylabel='Independent derivative relative difference')
    ax.tick_params(axis='x',rotation=60)
    fig.savefig(output/'comparison.png',dpi=160); plt.close(fig)
    print(json.dumps(metrics))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='runs/development_physical_inputs')
    parser.add_argument('--output',default='results/development_physical_inputs')
    args=parser.parse_args(); export(args.run,args.output)
