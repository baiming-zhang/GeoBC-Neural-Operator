"""Actual shared3 backbone comparison, scalars and existing PNG only."""
import argparse
import csv
import shutil
import time

OWN='tools/export_geobc_transolver_decoder_v64.py'
PLAN='notes/geobc_transolver_decoder_v64_plan.json'
REVIEW='notes/geobc_transolver_decoder_v64_exporter_root_review.json'
FAMILY='TS_ORIGINAL_GLOBAL_HASH'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_decoder_contract_v64 import terminal,inputs,complete_physical,check_identity
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];up=inputs();keys=s['selected_TRAIN_keys']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==3000
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==3 and a['resident_physical_target_reuses']==9
    assert a['input_array_attempts']==a['input_array_reads']
    assert all(a[k]==0 for k in ('encoded_TRAIN_reads','raw_GT_reads','DEV_FINAL_reads','scaler_fits','parent_checkpoint_hash_reads','parent_checkpoint_loads'))
    assert max(a['cache_setup_epoch_seconds'],a['cache_setup_monotonic_seconds'])<=180
    assert max(a['learning_interval_epoch_seconds'],a['learning_interval_monotonic_seconds'])<=900
    assert max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    init=a['model_initialization'][FAMILY];ref=s['expected_reference_initialization']
    assert init['original_reference_initial_digest_verified'] and init['reference_RNG_digest_verified']
    assert init['replacement']['reference_FNO_Hash_model_digest']==ref['model_digest'] and init['RNG_digest']==ref['RNG_digest']
    assert init['replacement']['fresh_encoder_head_digests']==init['shared_encoder_head_digests'] and init['replacement']['only_operator_replaced']
    assert init['replacement']['shared_modules_unchanged'] and not init['replacement']['pretrained_weights_used']
    row=a['models'][FAMILY]
    assert row['status']=='completed' and row['optimizer_step']==3000 and row['case_exposure_counts']=={k:1000 for k in keys}
    assert row['fresh_Adam0'] and row['fresh_model_digest']==init['model_digest'] and row['fresh_RNG_digest']==init['RNG_digest']
    assert set(row['gradient_checks'])==set(keys)
    for q in row['gradient_checks'].values():
        assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored'] and q['optimizer_updates']==0
        assert all(v>0 for v in q['groups_squared_L2'].values()) and q['leaves']['context']>0 and q['leaves']['latent']>0
    reference=s['FNO_Hash_shared_reference'];endpoints=row['endpoints']
    assert endpoints==a['aggregate_endpoints'][FAMILY] and set(endpoints)=={'0','100','1000'}
    for metric in list(endpoints.values())+list(reference.values()):complete_physical(metric,keys)
    d.stat_matches(row['checkpoint_identity'])
    plot=a['representative_field_plot'];image=check_identity(plot['identity'])
    assert image.suffix=='.png' and image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    assert plot['metrics_all_original_nodes'] and plot['original_stress_sign_unchanged'] and plot['no_additional_GT_or_inference']
    old=reference['1000'];new=endpoints['1000']
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    differences={k:100*(new[k]-old[k])for k in fields}
    old_cost=up['shared_summary']['models']['FNO_GLOBAL_HASH']['segment_epoch_seconds']
    new_cost=row['segment_epoch_seconds']
    out=d.path('results/geobc_transolver_decoder_v64');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    text=['# 相同连续解码器：原版Transolver与FNO骨干','',
        '**同固定3个困难TRAIN场、每场1000次全原节点曝光、共享一个fresh模型。整体几何／边界条件输入、初始PointNet与TCNN解码器、scaler与物理U/S联合损失相同；仅换骨干，骨干容量和计算成本不同。**','',
        '| 骨干＋相同Hash解码器 | 位移误差 | 应力误差 | 热点误差 | 学习区间 |',
        '|---|---:|---:|---:|---:|']
    for label,m,cost in [('原版Transolver',new,new_cost),('FNO',old,old_cost)]:
        text.append(f'| {label} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {cost:.1f}秒 |')
    text+=['',f'TS减FNO：位移{differences[fields[0]]:+.2f}、应力{differences[fields[1]]:+.2f}、热点{differences[fields[2]]:+.2f}个百分点；正值表示更差。学习区间成本比{new_cost/old_cost:.2f}倍，包含构建、资格、读出和保存，不是纯训练时间。','',
        f'本轮完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，模型{a["family_profiles"][FAMILY]["parameter_count"]:,}参数；FNO参考12,245,894参数。TS采用官方原版默认5层／宽256／8头／32 slices，旧紧凑R16 TS的91.50%新CAD应力误差不能直接与本轮TRAIN拟合比较。','',
        '参考完整fresh FNOHash初始字节与既有v61一致，替换骨干前后encoder和head字节不变，CPU/CUDA初始化随机状态恢复；没有读取训练检查点。全部262144个R64 token参与整体TS，不将空间块当独立attention。三个原案例全节点参数、context、latent梯度通过，资格不更新model或Adam且恢复四类RNG。','',
        'TRAIN物理标签3份各读一次、常驻端点复用9次，encoded／原CSV／DEV／FINAL／scaler拟合和父PT读取均0；FNO只用已有不可变标量指标，不重训或再读真值。自己的恢复与最终检查点只保存在本地。','',
        '这是选择下一宏观实验的拟合与成本诊断，不能证明新CAD泛化或宣布最终solver。原版DeepJEB GRID对齐、全原节点signed U(mm)/S(MPa)。查询接口C0不等于独立离节点FEM准确率。云图truth展示样本|S|P99共用色标，范围外仅显示饱和，全部原节点数值未裁剪。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,reference_binding=up['shared_binding'],
        reference_endpoints=reference,differences_pp=differences,learning_interval_cost_ratio=new_cost/old_cost,
        fitting_only=True,capacity_matched=False,final_solver_selected=False,new_DEV_accuracy=None))
    with (out/'per_field.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['backbone','field','passes','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for name,points in [('TS',endpoints),('FNO',reference)]:
            for step in ['0','100','1000']:
                for key in keys:
                    z=points[step]['per_case'][key];writer.writerow([name,key,step,z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    shutil.copyfile(image,out/'representative_stress.png');assert d.sha(out/'representative_stress.png')==plot['identity']['sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for label,points,color in [('TS + Hash',endpoints,'#3274a1'),('FNO + Hash',reference,'#e1812c')]:
        axes[0].plot([0,100,1000],[100*points[str(step)][fields[1]]for step in [0,100,1000]],'-o',label=label,color=color)
        axes[1].plot(range(3),[100*points['1000']['per_case'][k]['stress_relative_l2']for k in keys],'-o',label=label,color=color)
    axes[0].set_xticks([0,100,1000]);axes[0].set_xlabel('Full-node exposures per field');axes[0].set_ylabel('Physical stress relative L2 (%)')
    axes[1].set_xticks(range(3),keys,rotation=15);axes[1].set_title('Shared3 endpoint1000 / same continuous decoder')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_field.csv','comparison.png','representative_stress.png'];assert all((out/f).is_file()for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        actual_CPU_seconds=time.perf_counter()-began,files=files,PT_body_reads=0,new_GT_array_reads=0,GPU_calls=0))
    print(dict(status='completed',TS_minus_FNO_S_pp=differences[fields[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
