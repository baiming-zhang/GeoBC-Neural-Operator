"""One immutable scalar-only fullTRAIN100 report, before separate knownDEV readout."""
import argparse
import csv
import shutil
import time

OWN='tools/export_geobc_transolver_macro100_v66.py'
PLAN='notes/geobc_transolver_macro100_v66_plan.json'
REVIEW='notes/geobc_transolver_macro100_v66_exporter_root_review.json'
FAMILY='TS_ORIGINAL_BF16_GLOBAL_HASH'
FIELDS=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_macro100_contract_v66 import terminal,approved_contract,complete_physical,check_identity
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,_,_,up=approved_contract(PLAN);_,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['TRAIN_keys']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==1500
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==300 and a['resident_physical_target_reuses']==900
    assert a['input_array_attempts']==a['input_array_reads']
    assert all(a[k]==0 for k in ('encoded_TRAIN_reads','raw_GT_reads','DEV_FINAL_reads','scaler_fits','parent_checkpoint_hash_reads','parent_checkpoint_loads'))
    c=a['cache_receipt'];assert c['physical_TRAIN_loaded_cases']==300 and c['unused_encoded_target_files_removed_before_load']==300
    assert c['encoded_TRAIN_target_read_attempts']==c['encoded_TRAIN_target_reads']==0
    assert max(a['cache_setup_epoch_seconds'],a['cache_setup_monotonic_seconds'])<=600
    assert max(a['learning_interval_epoch_seconds'],a['learning_interval_monotonic_seconds'])<=900
    assert max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    row=a['models'][FAMILY];init=a['model_initialization'][FAMILY];expected=s['expected_FP32_TS_initialization']
    assert row['status']=='completed' and row['optimizer_step']==1500 and row['case_exposure_counts']=={k:5 for k in keys}
    assert row['fresh_Adam0'] and row['fresh_model_digest']==init['model_digest']==expected['model_digest']
    assert row['fresh_RNG_digest']==init['RNG_digest']==expected['RNG_digest'] and init['FP32_TS_exact_initial_bytes_verified']
    assert set(row['gradient_checks'])==set(s['qualification_TRAIN_keys'])
    for q in row['gradient_checks'].values():
        assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored'] and q['optimizer_updates']==0
        assert all(v>0 for v in q['groups_squared_L2'].values()) and q['leaves']['context']>0 and q['leaves']['latent']>0
    points=row['endpoints'];assert set(points)=={'0','1','5'} and points==a['aggregate_endpoints'][FAMILY]
    for metric in list(points.values())+list(s['reference_TRAIN5'].values()):complete_physical(metric,keys)
    d.stat_matches(row['checkpoint_identity']);plot=a['representative_field_plot'];image=check_identity(plot['identity'])
    assert image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n' and plot['metrics_all_original_nodes'] and plot['original_stress_sign_unchanged'] and plot['no_additional_GT_or_inference']
    rows=[('TS BF16 + continuous decoder / 5',points['5']),('FNO + continuous decoder / 5',s['reference_TRAIN5']['Hash']),('FNO direct / 5',s['reference_TRAIN5']['FNO'])]
    out=d.path('results/geobc_transolver_macro100_v66');assert not out.exists(),'One actual immutable report';out.mkdir(parents=True)
    text=['# 原版TS：100CAD短五遍能力比较','',
        '**相同100个TRAIN CAD×三载荷、全原FEM节点、scaler和物理U/S联合损失；fresh模型固定1500次更新，每个场5次曝光。三场拟合权重不转移。**','',
        '| 同5遍曝光方案 | TRAIN位移误差 | TRAIN应力误差 | TRAIN热点误差 |','|---|---:|---:|---:|']
    for label,z in rows:text.append('| '+label+' | '+' | '.join(f'{100*z[k]:.2f}%'for k in FIELDS)+' |')
    text+=['',f'TS完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，缓存准备{a["cache_setup_epoch_seconds"]:.2f}秒，学习区间{a["learning_interval_epoch_seconds"]:.2f}秒（含初始化、资格、0/1/5读出及保存）。已有参考指标只复用标量，不重训或重读真值；该区间不能直接当纯训练秒。','',
        '原版TS默认5层／宽256／8头／32 slices，完整262144个R64 token；BF16骨干算子、FP32参数/Adam/context/返回latent，官方TCNN FP16解码器。TS9,601,208参数，FNO条件Hash12,245,894，直接FNO4,526,858，容量、精度和初值跨方案不同，同曝光不等于严格架构因果消融。','',
        '实际完整fresh TS参数与四类RNG和此前三场精度对照相同；每种载荷最大TRAIN案例全节点参数/context/latent梯度通过，资格保持model/Adam/RNG不变。固定完整300案例顺序、每场5次曝光，没有挑best或训练子集。','',
        'TRAIN物理300份各读一次，常驻复用900次对应0/1/5端点；不用的编码标签在加载前移除，原CSV、DEV、FINAL、scaler拟合及父PT读取均0。旧FP32预算negative、三场拟合与旧基线失败成本全部保留，未计入本阶段学习区间。','',
        '当前TRAIN拟合不是新CAD准确率；所有有限完整终点将进入同20knownDEV×三载荷预测先提交的独立阶段，不根据TRAIN相对误差丢弃差候选。短五遍不证明收敛或架构上限，不自动延长训练或宣布最终solver。','',
        '原版DeepJEB GRID对齐、全原节点signed U(mm)/S(MPa)。云图共用truth展示样本|S|P99范围，超过范围仅显示饱和，指标用全部节点且不裁剪；连续查询接口不等于独立离节点FEM精度。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    with (out/'per_case.csv').open('w',encoding='utf8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['method','case','nodes','U_relative_L2','S_relative_L2','hotspot_relative_L2'])
        for label,metric in rows:
            for key in keys:
                z=metric['per_case'][key];writer.writerow([label,key,z['nodes'],z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    d.write(out/'metrics.json',dict(rows=dict(rows),actual_native=a,actual_external=e,terminal_binding=b,reference_bindings={n:up[n+'_binding']for n in ('Hash','FNO')},
        previous_shared_TS_binding=up['TS_binding'],knownDEV_readout_completed=False,final_solver_selected=False))
    shutil.copyfile(image,out/'representative_stress.png');assert d.sha(out/'representative_stress.png')==plot['identity']['sha256']
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));labels=['TS BF16 + decoder','FNO + decoder','FNO direct']
    for i,field in enumerate(FIELDS):axes[0].bar([j+(i-1)*.25 for j in range(3)],[100*z[field]for _,z in rows],width=.25,label=['U','S','Hotspot'][i])
    axes[0].set_xticks(range(3),labels,rotation=10);axes[0].set_ylabel('FullTRAIN300 mean relative L2 (%)');axes[0].legend()
    for field,label in zip(FIELDS,['U','Signed S','Hotspot']):axes[1].plot([0,1,5],[100*points[str(t)][field]for t in [0,1,5]],'-o',label=label)
    axes[1].set_xticks([0,1,5]);axes[1].set_xlabel('TS complete full-node case exposures');axes[1].set_ylabel('FullTRAIN300 mean relative L2 (%)');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','comparison.png','representative_stress.png'];assert all((out/f).is_file()for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        files=files,new_GT_array_reads=0,PT_body_reads=0,GPU_calls=0,knownDEV_readout_completed=False))
    print(dict(status='completed',TS_TRAIN_S_percent=100*points['5'][FIELDS[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
