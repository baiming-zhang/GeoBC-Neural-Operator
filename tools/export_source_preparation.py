"""Summarize final-set source preparation using CPU scalar records only."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget']:
        raise ValueError('Completed source stage required')
    rows=[]
    for cad,item in state['case_records'].items():
        facts=item.get('source_facts',{});geometry=item.get('expected_geometry',{})
        errors=[max(r['force_relative_error'],r['moment_relative_error'],r['virtual_work_relative_error']) for r in item.get('canonical_LOADs',{}).values()]
        rows.append(dict(CAD=cad,status=item['status'],nodes=geometry.get('nodes'),cells=geometry.get('cells'),
                         free_nodes=facts.get('free_nodes'),fixed_nodes=facts.get('fixed_nodes'),
                         GRID_XYZ_maximum_distance_mm=item.get('GRID_XYZ_maximum_distance_mm'),
                         maximum_load_transfer_relative=max(errors) if errors else None,
                         UTC_seconds=item['seconds']['UTC'],monotonic_seconds=item['seconds']['monotonic'],reason=item.get('reason')))
    if len(rows)!=20:raise ValueError('Fixed 20-CAD denominator differs')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'cases.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=dict(complete_sources=state['complete_sources'],CAD_denominator=20,load_denominator=60,
                 completed_source_CADs=len(state['sources']),counts=state['counts'],GT_reads=0,GPU_calls=0,K_PCG_runs=0,stage_seconds=terminal['elapsed_seconds'])
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# Final-set target-free sources\n\n'
        f"{len(state['sources'])}/20 fixed final CADs have source-qualified geometry and three canonical load vectors. "
        'The reader uses H5 vertices/cells and only the original CSV GRID/XYZ columns, preserving the existing row permutation. '
        'Actual deck material, TET10 connectivity, zero constraints and RBE load transfer are checked. No U/S labels, physics solves, GPU work or model predictions are read in this stage.\n\n'
        f"Actual source/import/IO/save cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. "
        'Source qualification precedes physical-input production; it is not an accuracy measurement.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,3.5),constrained_layout=True)
    ax.bar([r['CAD'] for r in rows],[r['UTC_seconds'] for r in rows],color=['#228833' if r['status']=='completed' else '#cc6677' for r in rows])
    ax.set(title='Final-set source qualification',ylabel='Source and save time (s)');ax.tick_params(axis='x',rotation=70)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/final_targetfree_sources');p.add_argument('--output',default='results/final_sources')
    args=p.parse_args();export(args.run,args.output)
