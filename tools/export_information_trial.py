"""Export a completed paired trial using scalar results only."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text()); terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget']:
        raise ValueError('Trial is not completed')
    if state['committed_predictions']!=360 or state['development_GT_reads']!=60:
        raise ValueError('Full paired denominator required')
    output.mkdir(parents=True,exist_ok=False)
    rows=[dict(arm=name,**row) for name,cases in state['metrics'].items() for row in cases]
    with (output/'cases.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics={key:state[key] for key in ['aggregate','paired_improvement_pp','paired_primary_seed_wins','practical_benefit_passed','training_target_reads','development_GT_reads','peak_allocated_bytes','peak_reserved_bytes']}
    metrics.update(stage_seconds=terminal['elapsed_seconds'],campaign_UTC_seconds=terminal['campaign_UTC_seconds'],
                   prior_cache_failure=state.get('prior_cache_failure'),arms=state['arms'])
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    improve=state['paired_improvement_pp']; decision='met' if state['practical_benefit_passed'] else 'did not meet'
    (output/'README.md').write_text('# Physical information in joint U/S learning\n\n'
        f"Three matched seeds on 100 training CADs and 20 known development CADs. ACTUAL {decision} the fixed practical-benefit criterion. "
        f"Mean improvement over ZERO: U {improve['U']:.3f} pp, S {improve['S']:.3f} pp, primary {improve['primary']:.3f} pp.\n\n"
        'Models share capacity, initialization, case order, 20 passes, loss and the original training40 output scaler. '
        'All 360 full-node development predictions were saved before the 60 shared original U/S targets were read. '
        'The development CADs are iterated rather than blind. Physics input production costs are reported separately.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    seeds=[2026,2027,2028]; x=np.arange(3)
    for ax,field in zip(axes,['U','S']):
        ax.bar(x-.18,[state['aggregate']['zero_seed'+str(seed)][field] for seed in seeds],.36,label='ZERO',color='#4477aa')
        ax.bar(x+.18,[state['aggregate']['actual_seed'+str(seed)][field] for seed in seeds],.36,label='ACTUAL',color='#228833')
        ax.set(xticks=x,xticklabels=seeds,ylabel='Mean relative error (%)',title=field+' on 20 known development CADs')
        ax.legend()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(practical_benefit_passed=state['practical_benefit_passed'],improvement_pp=improve)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/kinematic_information_trial');parser.add_argument('--output',default='results/physical_information_learning')
    args=parser.parse_args();export(args.run,args.output)
