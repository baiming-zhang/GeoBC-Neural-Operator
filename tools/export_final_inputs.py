"""Scalar report for final-set physics input production."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget']:
        raise ValueError('Completed numerical-input stage required')
    split=json.loads(Path('configs/split.json').read_text());rows=[]
    for cad in split['final']:
        case=state['case_records'].get(cad,{})
        endpoints=case.get('endpoints',[])
        rows.append(dict(CAD=cad,features_ready=cad in state['features'],nodes=case.get('nodes'),cells=case.get('cells'),
                         saved_endpoints=len(endpoints),maximum_true_residual=max((v['true_relative_residual'] for v in endpoints),default=None),
                         physical_derivative_relative=case.get('derivative',{}).get('relative'),
                         setup_seconds=case.get('setup_seconds'),solve_seconds=case.get('solve_seconds'),gradient_seconds=case.get('gradient_seconds')))
    output.mkdir(parents=True,exist_ok=False)
    with (output/'cases.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=dict(complete_inputs=state['complete_inputs'],CAD_denominator=20,load_denominator=60,complete_feature_CADs=len(state['features']),
                 saved_endpoints=state['saved_endpoints'],source_reads=state['source_reads'],GT_reads=state['GT_reads'],raw_source_reads=state['raw_source_reads'],
                 model_reads=state['model_reads'],stage_seconds=terminal['elapsed_seconds'],combined_solve_seconds=state['combined_solve_seconds'],gradient_seconds=state['gradient_seconds'],
                 reason=state.get('reason'),maximum_true_residual=max((r['maximum_true_residual'] for r in rows if r['maximum_true_residual'] is not None),default=None),
                 maximum_derivative_relative=max((r['physical_derivative_relative'] for r in rows if r['physical_derivative_relative'] is not None),default=None))
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    text='# Final-set physical inputs\n\n'
    text+=f"{len(state['features'])}/20 fixed final CADs have full-node physical displacement and gradient inputs; {state['saved_endpoints']}/60 endpoints were saved. "
    text+='New curved-P2 operators and degree-32 Chebyshev coarse corrections solve the three registered loads independently from zero. Recovered inputs are FP64 [Umm3, 1000*full-gradient9]. No final labels, model weights or raw source files are read in this stage.\n\n'
    text+=f"Actual import/cache/operator/solve/gradient/save cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. Source qualification is a separately reported cost. "
    text+='These are numerical input qualifications; model accuracy is measured only after all final predictions are saved.\n'
    if state.get('reason'):text+='\nFixed-protocol negative: '+state['reason']+'\n'
    (output/'README.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,3.5),constrained_layout=True)
    ax.bar([r['CAD'] for r in rows],[(r['solve_seconds'] or 0) for r in rows],color=['#228833' if r['features_ready'] else '#999999' for r in rows])
    ax.set(title='Final-set three-load physics inputs',ylabel='Independent three-load solve time (s)');ax.tick_params(axis='x',rotation=70)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/final_physical_inputs');p.add_argument('--output',default='results/final_physical_inputs')
    args=p.parse_args();export(args.run,args.output)
