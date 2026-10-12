"""Report the compact shared-trunk field experiment from saved scalar results."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget'] or state['committed_predictions']!=180 or state['development_GT_reads']!=60:
        raise ValueError('Complete three-seed predictions and comparison required')
    joint=json.loads(Path('results/physical_information_learning/metrics.json').read_text())['aggregate']
    components=json.loads(Path('results/input_components/metrics.json').read_text())['mean_errors']
    mean=state['compact_field_means']
    joint_mean={f:sum(joint['actual_seed'+str(s)][f] for s in [2026,2027,2028])/3 for f in ['U','S','primary']}
    values={'Joint input model':joint_mean,'Two-model field route':dict(U=components['U_ONLY']['U'],S=components['GRAD_ONLY']['S']), 'Shared-trunk field heads':mean}
    values['Two-model field route']['primary']=.5*(values['Two-model field route']['U']+values['Two-model field route']['S'])
    output.mkdir(parents=True,exist_ok=False)
    rows=[dict(arm=arm,**row) for arm,cases in state['metrics'].items() for row in cases]
    with (output/'cases.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=dict(mean_errors=values,per_seed=state['aggregate'],compact_field_gate_passed=state['compact_field_gate_passed'],
                 parameters=4554762,stage_seconds=terminal['elapsed_seconds'],arms=state['arms'],predictions=180,development_GT_reads=60)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    report='# Shared-trunk field-specific learning\n\nThree fresh seeds use one geometry/FNO trunk and separate U3-input and gradient9-input heads. Each seed sees the same 100 CADs, 300 load cases and 20 training passes. The original training40 output scaler remains fixed.\n\n| Model | Mean U error | Mean S error | Mean primary |\n|---|---:|---:|---:|\n'
    for name,row in values.items():report+=f"| {name} | {row['U']:.3f}% | {row['S']:.3f}% | {row['primary']:.3f}% |\n"
    report+=f"\nThe compact model has 4,554,762 parameters, versus 9,056,788 for the two-network route. The predeclared within-1pp field gate {'passed' if state['compact_field_gate_passed'] else 'did not pass'}. All 180 predictions preceded shared GT60 reads. These are the same 20 iterated development CADs; field routing was chosen using development results. The two-model reference is inferred from component field metrics.\n\nActual full-stage cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. Physics input construction is a separate cost.\n"
    (output/'README.md').write_text(report,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    for ax,field in zip(axes,['U','S']):
        ax.bar(list(values),[v[field] for v in values.values()],color=['#4477aa','#228833','#ee7733'])
        ax.set(title=field+' field error',ylabel='Mean relative error (%)');ax.tick_params(axis='x',rotation=15)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(mean_errors=values,gate_passed=state['compact_field_gate_passed'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/shared_trunk_field_learning');p.add_argument('--output',default='results/shared_trunk_field_learning')
    args=p.parse_args();export(args.run,args.output)
