"""Scalar-only report of actual fresh paired three-field decoder fitting."""
import argparse
import time

OWN='tools/export_geobc_single_decoder_v60.py'
REVIEW='notes/geobc_single_decoder_v60_exporter_root_review.json'
PLAN='notes/geobc_single_decoder_v60_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.single_decoder_pair_contract_v60 import terminal,complete_physical,check_identity
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['selected_TRAIN_keys'];families=s['families']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==6000
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==3 and a['resident_physical_target_reuses']==18
    assert all(a[k]==0 for k in ('encoded_TRAIN_reads','raw_GT_reads','DEV_FINAL_reads','scaler_fits','parent_checkpoint_hash_reads','parent_checkpoint_loads'))
    assert a['input_array_attempts']==a['input_array_reads']
    assert a['learning_interval_epoch_seconds']<=900 and a['learning_interval_monotonic_seconds']<=900
    assert a['cache_setup_epoch_seconds']<=180 and a['cache_setup_monotonic_seconds']<=180
    assert all(a[k]<=s['memory_limit_bytes'] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))
    backbones=[];costs={}
    for family in families:
        init=a['model_initialization'][family];backbones.append(init['shared_backbone_digests'])
        assert init['backbone_identity_passed'] and not init['full_model_same_across_families']
        assert a['family_profiles'][family]['parameter_count']==s['parameter_counts'][family]
        costs[family]=sum(a['models'][family][key]['segment_epoch_seconds']for key in keys)
        for key in keys:
            row=a['models'][family][key];q=row['actual_fullN_gradient_check']
            assert row['status']=='completed' and row['optimizer_step']==1000 and row['case_exposure_counts']=={key:1000}
            assert row['fresh_Adam0'] and row['fresh_model_digest']==init['model_digest'] and row['fresh_RNG_digest']==init['RNG_digest']
            assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored'] and q['optimizer_updates']==0
            assert all(v>0 for v in q['groups_squared_L2'].values()) and q['leaves']['context']>0 and q['leaves']['latent']>0
            for metric in row['endpoints'].values():complete_physical(metric,[key])
            d.stat_matches(row['checkpoint_identity'])
        for metric in a['aggregate_endpoints'][family].values():complete_physical(metric,keys)
    assert backbones[0]==backbones[1]
    plot=a['representative_field_plot'];assert plot['key']==s['representative_field']
    assert plot['metrics_all_original_nodes'] and plot['original_stress_sign_unchanged'] and plot['no_additional_GT_or_inference']
    assert plot['display_sample_nodes']<=20000
    image=check_identity(plot['identity']);assert image.suffix=='.png' and image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    endpoint={family:a['aggregate_endpoints'][family]['1000']for family in families}
    gains={k:endpoint['FNO9'][k]-endpoint['FNO_GLOBAL_HASH'][k]for k in fields}
    paired={key:{k:endpoint['FNO9']['per_case'][key][k]-endpoint['FNO_GLOBAL_HASH']['per_case'][key][k]
        for k in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2')}for key in keys}
    below1={family:sum(endpoint[family]['per_case'][key]['stress_relative_l2']<.01 for key in keys)for family in families}
    out=d.path('results/geobc_single_decoder_v60');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    names={'FNO9':'普通FNO解码器','FNO_GLOBAL_HASH':'FNO条件Hash解码器'}
    text=['# 同3个困难场：fresh1000步解码器对照','',
        '**两套方案均从头初始化，每个场独立训练1000次、每次用全部原始节点。共6000次更新；这里只判断已知场拟合，不是新CAD求解器排名。**','',
        '| 解码器包 | 位移误差 | 应力误差 | 热点误差 | 三场学习区间 |','|---|---:|---:|---:|---:|']
    for family in families:
        metric=endpoint[family]
        text.append(f'| {names[family]} | {100*metric[fields[0]]:.2f}% | {100*metric[fields[1]]:.2f}% | {100*metric[fields[2]]:.2f}% | {costs[family]:.1f}秒 |')
    text+=['',f'Hash相对普通FNO：应力降低{100*gains[fields[1]]:.2f}个百分点、位移{100*gains[fields[0]]:.2f}、热点{100*gains[fields[2]]:.2f}个百分点（负值为变差）。应力低于1%：FNO {below1["FNO9"]}/3，Hash {below1["FNO_GLOBAL_HASH"]}/3。','',
        f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒；连续学习区间{a["learning_interval_epoch_seconds"]:.2f}秒，包含初始化、梯度资格、训练、读出和检查点保存，表中区间不是纯训练秒数。启动前失败成本单独保留，CPU报告另计。','',
        '相同几何与载荷输入、scaler、seed2026、Adam.001、物理位移与signed应力联合损失；真实PointNet/FNO骨干初值完全相同。FNO约4.53M参数、FP32；Hash约12.25M参数、FP32骨干和官方TCNN FP16解码器。结果比较解码器包、容量与精度，不能单独归因于Hash编码。','',
        '固定53_482/ver、229_200/ver、56_293/hor，按此前共享模型的困难TRAIN场选定。每个模型真实全节点参数和context/latent梯度检查通过；逐场恢复相同fresh初值与四类RNG、Adam从0开始。不是随机总体、共享训练或未见几何检验。','',
        'TRAIN物理标签3份各读一次，18次端点常驻复用；encoded标签、原CSV、DEV、FINAL、scaler拟合及父检查点读取均0。最终每模型仅该场曝光1000次。之前warm1000的16.68%仅是背景诊断，不与fresh方案混作公平排名。','',
        '应力云图复用229_200/ver合法端点值，最多20000个固定原节点、相同镜头与signed MPa色标；所有数字用全部原节点。没有额外GT读取或推理。','',
        '原slides少量场拟合<1%不代表共享几何与边界条件迁移；原配置和样本不可完整恢复。本对照不是原数值精确复现，也不据此宣称架构上限或最终solver。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,aggregate_endpoints=a['aggregate_endpoints'],
        Hash_gain_pp={k:100*v for k,v in gains.items()},paired_per_field=paired,below1percent=below1,
        segment_epoch_seconds=costs,fitting_only=True,final_solver_selected=False,capacity_precision_matched=False))
    import csv,shutil
    with (out/'per_field.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['family','field','updates','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for family in families:
            for key in keys:
                for step in ('0','100','1000'):
                    row=a['models'][family][key]['endpoints'][step]['per_case'][key]
                    writer.writerow([family,key,step]+[row[k]for k in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2')])
    shutil.copyfile(image,out/'representative_stress.png')
    assert d.sha(out/'representative_stress.png')==plot['identity']['sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.5));colors={'FNO9':'#3b6cb4','FNO_GLOBAL_HASH':'#b75632'}
    for family in families:
        x=[0,100,1000]
        axes[0].plot(x,[100*a['aggregate_endpoints'][family][str(step)][fields[1]]for step in x],'-o',label=family,color=colors[family])
        axes[1].plot(range(3),[100*endpoint[family]['per_case'][key]['stress_relative_l2']for key in keys],'-o',label=family,color=colors[family])
    axes[0].set_xticks([0,100,1000]);axes[0].set_xlabel('Fresh full-node updates per field');axes[0].set_title('Same three fields / mean signed stress')
    axes[1].set_xticks(range(3),keys,rotation=15);axes[1].set_title('Endpoint1000 / all original nodes')
    axes[0].set_ylabel('Physical stress relative L2 (%)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_field.csv','comparison.png','representative_stress.png']
    assert all((out/k).is_file()for k in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        actual_CPU_seconds=time.perf_counter()-began,files=files,new_GT_array_reads=0,PT_body_reads=0,GPU_calls=0))
    print(dict(status='completed',Hash_S_gain_pp=100*gains[fields[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
