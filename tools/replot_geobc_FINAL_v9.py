"""Repair FINAL field layout from immutable owned arrays; no inference or CSV."""
import argparse
import time


def run():
    started=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.final_cad_contract_v9 import approved_contract,check_identity
    plan,_,dataset,_=approved_contract('notes/geobc_FINAL_v9_plan.json')
    folder=d.path(plan['output']);summary=d.read(folder/'summary.json')
    terminal=d.read(folder/'external_terminal.json')
    assert summary['status']=='completed' and summary['protocol_qualified']
    assert summary['closed'] and not summary['active']
    assert terminal['exit_code']==0 and terminal['within_external_budget']
    assert terminal['closed'] and not terminal['active']
    assert all(d.exact_process_exited(terminal[k]) for k in ('child_process','guard_process'))
    assert summary['committed_finite_predictions']==120 and summary['FINAL_numeric_CSV_parses']==20
    out=d.path('results/geobc_FINAL_v9/plots_v2')
    assert not out.exists(),'Do not silently replay the layout repair'
    sci=plan['scientific'];key=sci['representative_field_keys'][0]
    assert sci['representative_field_keys']==[sci['case_keys'][0]]
    import numpy as np
    record=d.read(folder/'truth'/key/'manifest.json')
    truth=np.load(check_identity(record['physical']),allow_pickle=False)
    xyz=np.load(check_identity(dataset['records'][key]['file_identities']['xyz_path']),allow_pickle=False)
    fields=[truth]
    for candidate in sci['candidates']:
        item=d.read(check_identity(summary['prediction_inventory'][candidate['name']+':'+key]))
        fields.append(np.load(check_identity(item['physical']),allow_pickle=False))
    n=dataset['records'][key]['num_nodes']
    assert xyz.shape==(n,3) and all(a.shape==(n,4) and np.isfinite(a).all() for a in fields)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True)
    for channel,title,unit,cmap in [('U','Displacement magnitude','mm','viridis'),('S','Original signed stress','MPa','coolwarm')]:
        values=[np.linalg.norm(a[:,:3],axis=1) if channel=='U' else a[:,3] for a in fields]
        minimum=min(float(v.min()) for v in values);maximum=max(float(v.max()) for v in values)
        assert maximum>minimum
        fig=plt.figure(figsize=(11.3,4.5))
        grid=fig.add_gridspec(1,4,width_ratios=[1,1,1,.06],left=.06,right=.93,bottom=.14,top=.81,wspace=.33)
        axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
        for axis,v,label in zip(axes,values,['CSV reference','FNO primary','Global Hash secondary']):
            artist=axis.scatter(xyz[:,0],xyz[:,1],c=v,s=.12,cmap=cmap,vmin=minimum,vmax=maximum,rasterized=True)
            axis.set_aspect('equal');axis.set_title(label,fontsize=10)
            axis.set_xlabel('Normalized X');axis.set_ylabel('Normalized Y')
        fig.colorbar(artist,cax=fig.add_subplot(grid[0,3]),label=unit)
        fig.suptitle(title+' / '+key+' / all original nodes',fontsize=12)
        fig.savefig(out/('field_'+channel+'.png'),dpi=150);plt.close(fig)
    d.write(out/'manifest.json',dict(status='completed',purpose='colorbar layout repair only',
        source_sha256=d.sha('tools/replot_geobc_FINAL_v9.py'),scientific_digest=plan['scientific_digest'],
        original_export_manifest=d.identity(d.path('results/geobc_FINAL_v9/manifest.json'),hashed=True),
        summary_identity=d.identity(folder/'summary.json',hashed=True),field_key=key,plot_nodes=n,
        plot_only_reads=dict(owned_FINAL_physical_arrays=1,input_xyz_arrays=1,committed_prediction_arrays=2,rawCSV_parses=0,model_inference=0),
        original_artifacts_preserved=True,metrics_changed=False,CPU_seconds=time.monotonic()-started,
        files=['field_U.png','field_S.png']))
    print(dict(status='completed',CPU_seconds=time.monotonic()-started,nodes=n,rawCSV_parses=0,model_inference=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true')
    if parser.parse_args().plot:run()
