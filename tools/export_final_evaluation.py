"""One scalar report for the fixed final model comparison."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget'] or state['committed_predictions']!=360 or state['final_GT_reads']!=60 or not state['prediction_barrier_completed']:
        raise ValueError('Full final prediction and readout stage required')
    means={mode:{field:sum(state['aggregate'][mode.lower()+'_seed'+str(seed)][field] for seed in [2026,2027,2028])/3 for field in ['U','S','primary']} for mode in ['ZERO','FIELDWISE']}
    rows=[dict(arm=arm,**row) for arm,cases in state['metrics'].items() for row in cases]
    output.mkdir(parents=True,exist_ok=False)
    with (output/'cases.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=dict(mean_errors=means,per_seed=state['aggregate'],paired_improvement_pp=state['paired_improvement_pp'],primary_seed_wins=state['paired_primary_seed_wins'],practical_benefit_passed=state['practical_benefit_passed'],
                 stage_seconds=terminal['elapsed_seconds'],cache_seconds=state['cache_seconds'],models=state['arms'],predictions=360,final_GT_case_reads=60,final_CSV_parses=state['final_CSV_parses'],training_updates=0,scaler_fits=0)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    report='# Fixed final-set model evaluation\n\nThe compact shared-trunk field model and its matched zero-physical-input control were selected using development results. Both use three registered seeds, the same original training40 output scaler, and all 20 fixed final CADs with three loads. No model is retrained or chosen using final errors.\n\n| Model | Mean U error | Mean signed S error | Mean primary |\n|---|---:|---:|---:|\n'
    for name,row in means.items():report+=f"| {name} | {row['U']:.3f}% | {row['S']:.3f}% | {row['primary']:.3f}% |\n"
    report+=f"\nPrimary improves by {state['paired_improvement_pp']['primary']:.3f} pp, with {state['paired_primary_seed_wins']}/3 paired seed wins. The predeclared practical-benefit criterion {'passed' if state['practical_benefit_passed'] else 'did not pass'}. All 360 full-node predictions preceded one shared read of original final U/S fields (20 CSV parses, 60 load cases), aligned by existing GRID row maps.\n\n"
    report+='The seeds share the same 20 CADs. This is an availability-conditional, identity-disjoint evaluation within the official training pool; separation by shape family is not established. Original signed S labels are used directly, without assuming an export sign rule.\n\n'
    report+=f"Actual checkpoint/cache/inference/save/readout cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. Physical-input generation and model training are reported separately.\n"
    (output/'README.md').write_text(report,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    for ax,field in zip(axes,['U','S']):
        ax.bar(list(means),[v[field] for v in means.values()],color=['#999999','#228833']);ax.set(title='Final '+field+' error',ylabel='Mean relative error (%)')
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(mean_errors=means,improvement_pp=state['paired_improvement_pp'],gate_passed=state['practical_benefit_passed'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/final_model_evaluation');p.add_argument('--output',default='results/final_evaluation')
    args=p.parse_args();export(args.run,args.output)
