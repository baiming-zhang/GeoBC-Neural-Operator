"""Export only scalar prepared-session deployment results."""
import argparse
import csv
import json
from pathlib import Path


def export(run,output):
    run,output=Path(run),Path(output)
    state=json.loads((run/'summary.json').read_text());terminal=json.loads((run/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or not terminal['within_budget'] or terminal['exit_code']!=0 or state['completed_CADs']!=20 or state['predicted_rows']!=1200 or state['outside_rejections']!=60 or not state['prepared_response_delivery_passed']:
        raise ValueError('Complete closed prepared-session delivery required')
    counts=state['counts']
    if counts['GT_reads'] or counts['training_updates'] or counts['scaler_fits'] or counts['checkpoint_reads']!=1 or counts['geometry_reads']!=20 or counts['feature_reads']!=20:raise ValueError('Registered delivery read counts differ')
    output.mkdir(parents=True,exist_ok=False)
    rows=[dict(CAD=cad,nodes=row['nodes'],cells=row['cells'],prepare_UTC_s=row['prepare_seconds']['UTC'],prepare_monotonic_s=row['prepare_seconds']['monotonic'],query_UTC_s=row['query_seconds']['UTC'],query_monotonic_s=row['query_seconds']['monotonic'],interior_rows=row['interior_rows'],outside_rejections=row['outside_rejections']) for cad,row in state['case_records'].items()]
    with (output/'cases.csv').open('x',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=dict(prepared_response_delivery_passed=True,CADs=20,physical_response_rows=1200,outside_rejections=60,seed=2026,parameters=4554762,stage_seconds=terminal['elapsed_seconds'],model_load_seconds=state['model_load_seconds'],counts=counts,peak_allocated_bytes=state['peak_allocated_bytes'],peak_reserved_bytes=state['peak_reserved_bytes'],case_records=state['case_records'])
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    report='# Prepared coordinate-response delivery\n\nOne frozen first-registered-seed compact model served the same public JSON request handler across all 20 final CADs. Each CAD produced 60 new interior U/S rows at physical XYZ coordinates, with three explicit outside-domain refusals. All 1,200 interior responses were finite, and all 60 outside requests returned null fields. Registered loads remain ver/hor/tor.\n\n'
    report+=f"Actual model load, prepared-input loading, location-index construction, requests and output cost: UTC {terminal['elapsed_seconds']['UTC']:.3f}s / monotonic {terminal['elapsed_seconds']['monotonic']:.3f}s. One checkpoint and each CAD's geometry/physical-input body were loaded once; GT reads, training and scaler fits were zero.\n\n"
    report+='This stage checks delivery from already prepared caches. Source construction, physical solves and model training are separate costs. It provides no new off-node finite-element truth, engineering accuracy, C1 continuity, arbitrary-load or changed-boundary-condition qualification. Final error conclusions are unchanged. See [usage](../../docs/response_usage.md) and [frozen final evaluation](../final_evaluation/README.md).\n'
    (output/'README.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    x=list(range(20));setup=[r['prepare_UTC_s'] for r in rows];query=[r['query_UTC_s'] for r in rows]
    ax.bar(x,setup,label='Prepared inputs + location index',color='#4477aa');ax.bar(x,query,bottom=setup,label='JSON coordinate queries',color='#228833')
    ax.set_xticks(x,[r['CAD'] for r in rows],rotation=70);ax.set_ylabel('Actual UTC seconds');ax.set_title('Prepared-cache response delivery across final20');ax.legend()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:metrics[k] for k in ['CADs','physical_response_rows','outside_rejections','stage_seconds','counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/prepared_response_delivery');p.add_argument('--output',default='results/prepared_response_delivery')
    args=p.parse_args();export(args.run,args.output)
