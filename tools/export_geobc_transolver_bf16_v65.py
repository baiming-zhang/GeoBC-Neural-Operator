"""Scalar-only report of actual shared3 precision and backbone comparisons."""
import argparse
import csv
import shutil
import time

OWN='tools/export_geobc_transolver_bf16_v65.py'
PLAN='notes/geobc_transolver_bf16_v65_plan.json'
REVIEW='notes/geobc_transolver_bf16_v65_exporter_root_review.json'
FAMILY='TS_ORIGINAL_BF16_GLOBAL_HASH'
FIELDS=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_bf16_contract_v65 import terminal,inputs,complete_physical,check_identity,approved_contract
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    approved_contract(PLAN)
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
    init=a['model_initialization'][FAMILY];ref=s['expected_reference_initialization'];fpinit=s['expected_FP32_TS_initialization']
    assert init['FP32_TS_exact_initial_bytes_verified'] and init['original_reference_initial_digest_verified'] and init['reference_RNG_digest_verified']
    assert init['model_digest']==fpinit['model_digest'] and init['RNG_digest']==fpinit['RNG_digest']==ref['RNG_digest']
    assert init['shared_encoder_head_digests']==fpinit['shared_encoder_head_digests']==init['replacement']['fresh_encoder_head_digests']
    assert init['replacement']['reference_FNO_Hash_model_digest']==ref['model_digest']
    assert init['replacement']['only_operator_replaced'] and init['replacement']['shared_modules_unchanged'] and not init['replacement']['pretrained_weights_used']
    row=a['models'][FAMILY]
    assert row['status']=='completed' and row['optimizer_step']==3000 and row['case_exposure_counts']=={k:1000 for k in keys}
    assert row['fresh_Adam0'] and row['fresh_model_digest']==init['model_digest'] and row['fresh_RNG_digest']==init['RNG_digest']
    assert set(row['gradient_checks'])==set(keys)
    for q in row['gradient_checks'].values():
        assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored'] and q['optimizer_updates']==0
        assert all(v>0 for v in q['groups_squared_L2'].values()) and q['leaves']['context']>0 and q['leaves']['latent']>0
    points=row['endpoints'];reference=s['FNO_Hash_shared_reference'];fp32=s['FP32_TS_partial_endpoints']
    assert points==a['aggregate_endpoints'][FAMILY] and set(points)==set(reference)=={'0','100','1000'} and set(fp32)=={'0','100'}
    assert fp32==up['fp32_summary']['models']['TS_ORIGINAL_GLOBAL_HASH']['endpoints']
    for metric in list(points.values())+list(reference.values())+list(fp32.values()):complete_physical(metric,keys)
    d.stat_matches(row['checkpoint_identity'])
    failure=a['previous_FP32_budget_failure']
    assert failure['terminal_binding']==up['fp32_binding'] and failure['not_fixed1000_completed'] and failure['not_discarded']
    assert failure['completed_updates']==up['fp32_summary']['optimizer_step_completions']==2285
    assert failure['stage_epoch_seconds']==up['fp32_external']['elapsed_epoch_seconds']
    plot=a['representative_field_plot'];image=check_identity(plot['identity'])
    assert image.suffix=='.png' and image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    assert plot['metrics_all_original_nodes'] and plot['original_stress_sign_unchanged'] and plot['no_additional_GT_or_inference']
    old=reference['1000'];new=points['1000'];diff={k:100*(new[k]-old[k])for k in FIELDS}
    precision_diff100={k:100*(points['100'][k]-fp32['100'][k])for k in FIELDS}
    oldcost=up['shared_summary']['models']['FNO_GLOBAL_HASH']['segment_epoch_seconds'];newcost=row['segment_epoch_seconds']
    out=d.path('results/geobc_transolver_bf16_v65');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    text=['# 原版Transolver BF16：相同连续解码器对照','',
        '**同固定三个困难TRAIN场、每场1000次全原节点曝光、共享一个fresh模型；保持PointNet和TCNN连续解码器初值、输入、scaler、物理U/S联合损失与曝光顺序。只将原版TS骨干计算改为BF16，参数和Adam仍为FP32。**','',
        '| 骨干＋相同连续解码器 | 位移误差 | 应力误差 | 热点误差 | 学习区间 |',
        '|---|---:|---:|---:|---:|']
    for label,z,cost in [('原版TS BF16',new,newcost),('FNO FP32',old,oldcost)]:
        text.append(f'| {label} | {100*z[FIELDS[0]]:.2f}% | {100*z[FIELDS[1]]:.2f}% | {100*z[FIELDS[2]]:.2f}% | {cost:.1f}秒 |')
    text+=['',f'TS BF16减FNO：位移{diff[FIELDS[0]]:+.2f}、应力{diff[FIELDS[1]]:+.2f}、热点{diff[FIELDS[2]]:+.2f}个百分点；正值表示更差。学习区间成本比{newcost/oldcost:.2f}倍，包含初始化、资格、读出和保存，不是纯训练秒。','',
        f'原FP32 TS因学习900秒上限停止在2285次更新，完整阶段{failure["stage_epoch_seconds"]:.2f}秒；其检查点、原日志、成本和0/100端点保留，没有1000端点。BF16是独立fresh精度轨迹，不是原失败预算重置或检查点恢复。','',
        f'共同100次曝光端点：FP32 TS应力{100*fp32["100"][FIELDS[1]]:.2f}%，BF16 TS{100*points["100"][FIELDS[1]]:.2f}%，差{precision_diff100[FIELDS[1]]:+.2f}个百分点。精度改变数值轨迹，不能把BF16与FP32当相同数学执行或伪造FP32最终误差。','',
        f'本轮完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，加上已发生的FP32失败为{e["elapsed_epoch_seconds"]+failure["stage_epoch_seconds"]:.2f}秒；历史FNO参考成本另外保留。TS共9,601,208参数，FNO参考12,245,894，容量与骨干精度不等，比较的是实际方案。','',
        '实际fresh TS全参数字节和四类RNG与FP32原实验一致。官方原版5层、宽256、8头、32 slices，完整262144个R64 token参与整体attention，三个实际全节点参数/context/latent梯度检查通过，资格不更新模型或Adam。TRAIN物理3份各读一次、常驻端点复用9次；encoded、原CSV、DEV、FINAL、scaler拟合与父检查点读取均0。FNO和FP32参考只复用不可变标量，不重新推理或读取GT。','',
        '这是TRAIN拟合和成本诊断，不能证明新CAD泛化或宣布最终solver。原版DeepJEB GRID对齐、全原节点signed U(mm)/S(MPa)。查询接口C0不代表独立离节点FEM准确率。云图共用truth展示样本|S|P99色标，范围外只在显示时饱和，全节点指标和符号不变。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,reference_binding=up['shared_binding'],FP32_binding=up['fp32_binding'],
        FNO_endpoints=reference,FP32_partial_endpoints=fp32,differences_pp=diff,precision_diff100_pp=precision_diff100,learning_interval_cost_ratio=newcost/oldcost,
        old_FP32_stage_seconds=failure['stage_epoch_seconds'],two_TS_actual_stage_seconds=e['elapsed_epoch_seconds']+failure['stage_epoch_seconds'],
        capacity_matched=False,arithmetic_matched_across_backbones=False,fitting_only=True,final_solver_selected=False,new_DEV_accuracy=None))
    with (out/'per_field.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['backbone_precision','field','passes','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for name,endpoint in [('TS_BF16',points),('FNO_FP32',reference),('TS_FP32_partial',fp32)]:
            for step in sorted(endpoint,key=int):
                for key in keys:
                    z=endpoint[step]['per_case'][key];writer.writerow([name,key,step,z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    shutil.copyfile(image,out/'representative_stress.png');assert d.sha(out/'representative_stress.png')==plot['identity']['sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for label,endpoint,color in [('TS BF16 + Hash',points,'#3274a1'),('FNO FP32 + Hash',reference,'#e1812c'),('TS FP32 partial',fp32,'#777777')]:
        steps=sorted(map(int,endpoint));axes[0].plot(steps,[100*endpoint[str(t)][FIELDS[1]]for t in steps],'-o',label=label,color=color)
        if '1000'in endpoint:axes[1].plot(range(3),[100*endpoint['1000']['per_case'][k]['stress_relative_l2']for k in keys],'-o',label=label,color=color)
    axes[0].set_xticks([0,100,1000]);axes[0].set_xlabel('Full-node exposures per field');axes[0].set_ylabel('Physical stress relative L2 (%)')
    axes[1].set_xticks(range(3),keys,rotation=15);axes[1].set_title('Shared3 final1000 / FP32 TS unavailable')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_field.csv','comparison.png','representative_stress.png'];assert all((out/f).is_file()for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        actual_CPU_seconds=time.perf_counter()-began,files=files,PT_body_reads=0,new_GT_array_reads=0,GPU_calls=0))
    print(dict(status='completed',TS_minus_FNO_S_pp=diff[FIELDS[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
