"""Compare fresh displacement/gradient input controls with frozen joint results."""
import argparse
import csv
import json
from pathlib import Path


def export(run,reference,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
    frozen=json.loads(Path(reference).read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget'] or state['committed_predictions']!=360 or state['development_GT_reads']!=60:
        raise ValueError('Complete component experiment required')
    output.mkdir(parents=True,exist_ok=False)
    means={}
    for mode in ['ZERO','ACTUAL','U_ONLY','GRAD_ONLY']:
        values=frozen['aggregate'] if mode in ['ZERO','ACTUAL'] else state['aggregate']
        means[mode]={field:sum(values[mode.lower()+'_seed'+str(seed)][field] for seed in [2026,2027,2028])/3 for field in ['U','S','primary']}
    rows=[dict(arm=name,**row) for name,cases in state['metrics'].items() for row in cases]
    with (output/'cases.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparable={mode:{field:means[mode][field]<=means['ACTUAL'][field]+1 for field in ['U','S']} for mode in ['U_ONLY','GRAD_ONLY']}
    metrics=dict(mean_errors=means,within_one_pp_of_joint=comparable,stage_seconds=terminal['elapsed_seconds'],arms=state['arms'],
                 new_prediction_rows=360,shared_development_GT_reads=60,reference_scope='Frozen ZERO/ACTUAL scalar results; no endpoint replay')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    text='# Displacement and gradient input contributions\n\nThree fresh matched seeds isolate U3 and gradient9, with the same FNO, initialization, original training40 scaler and 20 training passes. ZERO and ACTUAL are frozen references from the previous experiment.\n\n| Input | Mean U error | Mean S error | Mean primary |\n|---|---:|---:|---:|\n'
    for mode,row in means.items():text+=f"| {mode} | {row['U']:.3f}% | {row['S']:.3f}% | {row['primary']:.3f}% |\n"
    text+=f"\nNew trial cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. All 360 new predictions were saved before shared original development GT60 reads. The 20 development CADs are iterated; final data remains unused.\n"
    (output/'README.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for ax,field in zip(axes,['U','S']):
        ax.bar(list(means),[row[field] for row in means.values()],color=['#999999','#228833','#4477aa','#ee7733'])
        ax.set(title=field+' information control',ylabel='Mean relative error (%)')
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(mean_errors=means,within_one_pp_of_joint=comparable)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/input_component_trial');parser.add_argument('--reference',default='results/physical_information_learning/metrics.json');parser.add_argument('--output',default='results/input_components')
    args=parser.parse_args();export(args.run,args.reference,args.output)
