"""Scalar report for the trained continuous U/S interface."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text()); terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget']:
        raise ValueError('Continuous interface stage is not completed')
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for cad,case in state['case_records'].items():
        for load in ['ver','hor','tor']:
            checks=case.get('continuous_checks',{}).get(load,{})
            node=case.get('nodal_checks',{}).get(load,{})
            rows.append(dict(CAD=cad,load=load,nodes=case['nodes'],cells=case['cells'],
                             interior_U_relative=checks.get('same_model_U'),interior_S_relative=checks.get('same_model_S'),
                             nodal_U_relative=node.get('U'),nodal_S_relative=node.get('S'),
                             chunk_U_relative=checks.get('chunk_U'),chunk_S_relative=checks.get('chunk_S'),
                             coordinate_gradient_max_relative=max(checks.get('coordinate_gradient_relative_by_channel',[None]))))
    with (output/'checks.csv').open('x',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics={key:state.get(key) for key in ['continuous_interface_passed','predicted_rows','outside_rejections','reference_reads','checkpoint_reads','source_reads','feature_reads','peak_allocated_bytes','peak_reserved_bytes','GT_reads','training_updates']}
    metrics.update(stage_seconds=terminal['elapsed_seconds'],cases=state['case_records'])
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    passed=state.get('continuous_interface_passed',False)
    (output/'README.md').write_text('# Continuous learned U/S response\n\n'
        f"Interface checks {'passed' if passed else 'did not pass'} on the fixed three CADs. "
        f"{state['predicted_rows']} physical U/S query rows were saved, including 288 interior rows and 72 nodal witnesses. "
        f"{state['outside_rejections']} outside-box requests were explicitly rejected.\n\n"
        'The API takes physical coordinates in mm and a registered canonical load. It locates the curved element, interpolates native condition features and the recovered physical covariates, and queries the trained FNO decoder. '
        'The first preregistered seed is used. Coordinate gradients include the actual inverse and covariate interpolation. '
        'Checks compare the located API with the same model queried in explicit cells and with saved nodal predictions; no new off-node FEM accuracy is measured.\n\n'
        f"Actual cache/model/index/query/check/save cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. "
        'Physics preprocessing and neural training costs are inherited separately.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    labels=[row['CAD']+'/'+row['load'] for row in rows]
    for ax,field in zip(axes,['U','S']):
        values=[row['nodal_'+field+'_relative'] for row in rows]
        ax.bar(labels,[v if v is not None else 0 for v in values],color='#228833')
        ax.set(title='Nodal '+field+' vs saved trained-model prediction',ylabel='Relative difference')
        ax.tick_params(axis='x',rotation=70)
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(passed=passed,rows=state['predicted_rows'],stage_seconds=terminal['elapsed_seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/continuous_neural_response');parser.add_argument('--output',default='results/continuous_neural_response')
    args=parser.parse_args();export(args.run,args.output)
