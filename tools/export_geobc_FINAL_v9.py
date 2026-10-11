"""Terminal FINAL scalar report and one predeclared original-node field pair."""
import argparse
import time

OWN='tools/export_geobc_FINAL_v9.py'
PLAN='notes/geobc_FINAL_v9_plan.json'
REVIEW='notes/geobc_FINAL_v9_exporter_review.json'


def run():
    start=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.final_cad_contract_v9 import approved_contract,check_identity,complete_physical
    marker=d.read(REVIEW);assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,_,dataset,qualified=approved_contract(PLAN);sci=p['scientific'];assert marker['scientific_digest']==p['scientific_digest']
    folder=d.path(p['output']);s=d.read(folder/'summary.json');e=d.read(folder/'external_terminal.json')
    assert s['status']=='completed' and s['protocol_qualified'] and s['closed'] and not s['active']
    assert e['exit_code']==0 and e['within_external_budget'] and e['closed'] and not e['active']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert s['committed_finite_predictions']==120 and s['FINAL_numeric_CSV_parse_attempts']==s['FINAL_numeric_CSV_parses']==20
    assert s['FINAL_scored_case_count']==s['FINAL_numeric_reads']==60 and s['raw_GT_reads']==20
    assert s['training_updates']==s['scaler_fits']==s['owned_GT_array_reads']==0
    assert s['primary_frozen_before_FINAL']=='FNO100_12000' and not s['primary_changed_from_FINAL']
    points=[]
    for candidate in sci['candidates']:
        name=candidate['name'];row=s['candidates'][name];m=row['full60'];complete_physical(m,sci['case_keys'])
        points.append(dict(name=name,primary=name==sci['primary'],U_percent=100*m['case_mean_u_relative_l2'],S_percent=100*m['case_mean_stress_relative_l2'],hot_percent=100*m['case_mean_stress_hotspot_relative_l2'],parameters=row['model_profile']['parameter_count'],inference_UTC_seconds=row['inference_epoch_seconds']))
    out=d.path('results/geobc_FINAL_v9');out.mkdir(parents=True,exist_ok=True);assert not (out/'manifest.json').exists(),'No silent export replay'
    lines=['# FINAL20：冻结后一次CAD隔离测试','',
           '**主模型FNO在读取FINAL标签前由knownDEV联合误差选定，保持不变；全局Hash作热点对照。**','',
           '| 方法 | 位移误差 | 应力误差 | 热点误差 |','|---|---:|---:|---:|---:|']
    for x in points:lines.append(f"| {x['name']} | {x['U_percent']:.2f}% | {x['S_percent']:.2f}% | {x['hot_percent']:.2f}% |")
    lines+=['','同100训练CAD×三载荷／12000步／seed2026／oldTRAIN40-only scaler；固定20 FINAL CAD×三载荷共60例，全原节点物理U(mm)／原signed S(MPa)。',
            '',f"120份完整有限预测先提交／重验，再20个官方CSV各数值解析一次，60份场共享两个候选；完整阶段{e['elapsed_epoch_seconds']:.2f}秒。没有训练更新、scalerfit、ID替换或根据FINAL改primary。",
            '', '该FINAL来自可用官方TRAIN池，CAD与训练和knownDEV隔离；不是官方独立DeepJEB测试或无偏可用性抽样。单seed、容量4.53M vs12.25M、精度FP32 MLP vsFP16 TCNN不同，不能称纯架构因果或显著性结论。',
            '', 'Stress保留CSV原列和符号，不重建Cauchy应力或伪造符号。真实连续查询一致性／梯度已验证，但独立离节点FEM准确性未测。场图固定第一FINAL CAD／ver，所有原节点参与物理评分；图为normalized XY投影，前后节点可能重叠，两个预测使用相同色阶。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,candidates=s['candidates'],paired_CAD20=s['paired_CAD20'],primary=sci['primary'],primary_changed_from_FINAL=False,stage_UTC_seconds=e['elapsed_epoch_seconds']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.3,3.9))
    for j,x in enumerate(points):ax.bar([i+(j-.5)*.3 for i in range(3)],[x['U_percent'],x['S_percent'],x['hot_percent']],width=.3,label=x['name'])
    ax.set_xticks([0,1,2],['Displacement','Signed stress','Stress hotspot']);ax.set_ylabel('Case mean relative L2 (%)');ax.set_title('FINAL20 x 3: frozen100 TRAIN CAD / 12,000 updates');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    # These are ownedFINAL cached truths after terminal; no rawCSV reparse or
    # model inference. Track these plot-only numerical reads separately.
    import numpy as np
    assert sci['representative_field_keys']==[sci['case_keys'][0]]
    key=sci['representative_field_keys'][0];truth_record=d.read(folder/'truth'/key/'manifest.json')
    truth=np.load(check_identity(truth_record['physical']),allow_pickle=False)
    xyz=np.load(check_identity(dataset['records'][key]['file_identities']['xyz_path']),allow_pickle=False)
    fields=[truth];plot_reads=dict(owned_FINAL_physical_arrays=1,input_xyz_arrays=1,committed_prediction_arrays=0,rawCSV_parses=0,model_inference=0)
    for point in points:
        inventory=s['prediction_inventory'][point['name']+':'+key];record=d.read(check_identity(inventory))
        fields.append(np.load(check_identity(record['physical']),allow_pickle=False));plot_reads['committed_prediction_arrays']+=1
    n=dataset['records'][key]['num_nodes'];assert xyz.shape==(n,3) and all(a.shape==(n,4) and np.isfinite(a).all() for a in fields)
    for channel,title,unit,cmap,file in [('U','Displacement magnitude','mm','viridis','field_U.png'),('S','Original signed stress','MPa','coolwarm','field_S.png')]:
        values=[np.linalg.norm(a[:,:3],axis=1) if channel=='U' else a[:,3] for a in fields]
        minimum=min(float(v.min()) for v in values);maximum=max(float(v.max()) for v in values)
        assert maximum>minimum
        fig,axes=plt.subplots(1,3,figsize=(11.8,4.1))
        for axis,v,label in zip(axes,values,['CSV reference','FNO primary','Global Hash secondary']):
            artist=axis.scatter(xyz[:,0],xyz[:,1],c=v,s=.12,cmap=cmap,vmin=minimum,vmax=maximum,rasterized=True)
            axis.set_aspect('equal');axis.set_title(label,fontsize=9);axis.set_xlabel('Normalized X');axis.set_ylabel('Normalized Y')
        fig.colorbar(artist,ax=list(axes),label=unit,fraction=.025,pad=.02);fig.suptitle(title+' / '+key+' / all original nodes',fontsize=11)
        fig.subplots_adjust(left=.05,right=.91,bottom=.15,top=.83,wspace=.3);fig.savefig(out/file,dpi=150);plt.close(fig)
    files=['BRIEF.md','metrics.json','comparison.png','field_U.png','field_S.png'];assert all((out/file).is_file() for file in files)
    d.write(out/'manifest.json',dict(status='completed',scientific_digest=p['scientific_digest'],exporter_source_sha256=d.sha(OWN),points=points,primary=sci['primary'],primary_changed_from_FINAL=False,plot_only_reads=plot_reads,plot_nodes=n,actual_CPU_export_seconds=time.monotonic()-start,stage_UTC_seconds=e['elapsed_epoch_seconds'],summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True),files=files,independent_offnode_FEM_accuracy_measured=False))
    print(dict(status='completed',points=points,CPU_seconds=time.monotonic()-start,plot_only_reads=plot_reads))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
