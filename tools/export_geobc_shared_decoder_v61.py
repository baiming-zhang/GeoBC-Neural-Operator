"""Scalar-only actual shared3 vs independent3 decoder-package report."""
import argparse
import time

OWN='tools/export_geobc_shared_decoder_v61.py'
REVIEW='notes/geobc_shared_decoder_v61_exporter_root_review.json'
PLAN='notes/geobc_shared_decoder_v61_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.shared_decoder_pair_contract_v61 import terminal,complete_physical,check_identity
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['selected_TRAIN_keys'];families=s['families']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==6000
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==3 and a['resident_physical_target_reuses']==18
    assert all(a[k]==0 for k in ('encoded_TRAIN_reads','raw_GT_reads','DEV_FINAL_reads','scaler_fits','parent_checkpoint_hash_reads','parent_checkpoint_loads'))
    assert a['input_array_attempts']==a['input_array_reads']
    assert max(a['learning_interval_epoch_seconds'],a['learning_interval_monotonic_seconds'])<=900
    assert max(a['cache_setup_epoch_seconds'],a['cache_setup_monotonic_seconds'])<=180
    assert all(a[k]<=s['memory_limit_bytes']for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))
    costs={};backbones=[]
    for family in families:
        init=a['model_initialization'][family];expected=s['expected_actual_initialization'][family]
        assert init['ownv60_exact_initial_match'] and init['backbone_identity_passed']
        for field in ('model_digest','RNG_digest','shared_backbone_digests'):assert init[field]==expected[field]
        backbones.append(init['shared_backbone_digests'])
        assert a['family_profiles'][family]['parameter_count']==s['parameter_counts'][family]
        row=a['models'][family]
        assert row['status']=='completed' and row['optimizer_step']==3000 and row['case_exposure_counts']=={key:1000 for key in keys}
        assert row['fresh_Adam0'] and row['fresh_model_digest']==init['model_digest'] and row['fresh_RNG_digest']==init['RNG_digest'] and row['initial_full3_scalar_match']
        assert set(row['gradient_checks'])==set(keys)
        for key in keys:
            q=row['gradient_checks'][key]
            assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored'] and q['optimizer_updates']==0
            assert all(v>0 for v in q['groups_squared_L2'].values()) and q['leaves']['context']>0 and q['leaves']['latent']>0
        for metric in row['endpoints'].values():complete_physical(metric,keys)
        assert row['endpoints']==a['aggregate_endpoints'][family]
        d.stat_matches(row['checkpoint_identity']);costs[family]=row['segment_epoch_seconds']
    assert backbones[0]==backbones[1]
    plot=a['representative_field_plot'];assert plot['key']==s['representative_field']
    assert plot['metrics_all_original_nodes'] and plot['original_stress_sign_unchanged'] and plot['no_additional_GT_or_inference']
    assert plot['display_sample_nodes']<=20000 and plot['visual_range_rule'].startswith('Truth display sample absoluteP99')
    image=check_identity(plot['identity']);assert image.suffix=='.png' and image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    shared={f:a['aggregate_endpoints'][f]['1000']for f in families}
    independent={f:s['independent_endpoint_reference'][f]['1000']for f in families}
    for f in families:complete_physical(independent[f],keys)
    sharing_changes={f:{k:100*(shared[f][k]-independent[f][k])for k in fields}for f in families}
    package_gains={k:100*(shared['FNO9'][k]-shared['FNO_GLOBAL_HASH'][k])for k in fields}
    below1={f:sum(shared[f]['per_case'][key]['stress_relative_l2']<.01 for key in keys)for f in families}
    out=d.path('results/geobc_shared_decoder_v61');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True);names={'FNO9':'共享FNO','FNO_GLOBAL_HASH':'共享FNO条件Hash'}
    text=['# 同3场共享一个模型：解码器对照','',
        '**每套方案一个fresh模型共同学习固定3个场，每场1000次全原节点曝光、每模型Adam3000步。与上一轮三个独立模型使用相同自身初值和逐场曝光；这是TRAIN拟合诊断。**','',
        '| 共享模型 | 位移误差 | 应力误差 | 热点误差 | 学习区间 |','|---|---:|---:|---:|---:|']
    for f in families:
        metric=shared[f];text.append(f'| {names[f]} | {100*metric[fields[0]]:.2f}% | {100*metric[fields[1]]:.2f}% | {100*metric[fields[2]]:.2f}% | {costs[f]:.1f}秒 |')
    text+=['',f'Hash相对共享FNO：应力降低{package_gains[fields[1]]:.2f}个百分点、位移{package_gains[fields[0]]:.2f}、热点{package_gains[fields[2]]:.2f}（负值为变差）；应力低于1%：FNO {below1["FNO9"]}/3、Hash {below1["FNO_GLOBAL_HASH"]}/3。','',
        f'共享相对各自独立拟合，应力变化：FNO {sharing_changes["FNO9"][fields[1]]:+.2f}个百分点、Hash {sharing_changes["FNO_GLOBAL_HASH"][fields[1]]:+.2f}个百分点；独立应力分别18.27%/0.60%，独立位移5.12%/4.95%。正值表示共享更差。共享一条Adam3000轨迹和一个模型、独立三条Adam1000轨迹和三倍模型总容量，因此不能单独归因梯度冲突。','',
        f'整轮{e["elapsed_epoch_seconds"]:.2f}秒，连续学习区间{a["learning_interval_epoch_seconds"]:.2f}秒；区间含初始化、资格、读出与保存，不是纯训练时间。CPU报告另计，上一轮独立拟合428.31秒成本保留。','',
        '真实初始完整model/RNG和各family自身独立模型一致，encoder/operator跨family相同；所有六个逐场全节点参数/context/latent梯度检查通过，模型/Adam/RNG无资格更新。相同seed2026、scaler、Adam.001、几何载荷输入和物理U(mm)/signed S(MPa)联合损失，固定平衡顺序，不挑best。','',
        'FNO约4.53M参数/FP32，条件Hash约12.25M参数/FP32骨干+官方TCNN FP16解码器。解码器包、容量与精度不同，不能单独归因Hash编码。固定53_482/ver、229_200/ver、56_293/hor是已知困难TRAIN场，不能当新CAD或最终solver排名。','',
        'TRAIN物理3份各读一次、18次常驻端点复用，encoded/raw/DEV/FINAL/scaler拟合及父PT读取均0；两份共享检查点只保存在本地。云图同原节点sample/镜头，色标为truth展示样本|S|的99分位，超范围只在显示时饱和，MPa与符号不变；所有指标仍用全部原节点且不截断，不额外读取GT或推理。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,aggregate_endpoints=a['aggregate_endpoints'],
        independent_reference=independent,sharing_changes_pp=sharing_changes,shared_Hash_gain_pp=package_gains,below1percent=below1,
        fitting_only=True,final_solver_selected=False,capacity_precision_matched=False))
    import csv,shutil
    with (out/'per_field.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['family','regime','field','passes','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for f in families:
            for regime,endpoints in [('shared',a['aggregate_endpoints'][f]),('independent',s['independent_endpoint_reference'][f])]:
                for step in ('0','100','1000'):
                    for key in keys:
                        row=endpoints[step]['per_case'][key];writer.writerow([f,regime,key,step]+[row[k]for k in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2')])
    shutil.copyfile(image,out/'representative_stress.png');assert d.sha(out/'representative_stress.png')==plot['identity']['sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.6));colors={'FNO9':'#3b6cb4','FNO_GLOBAL_HASH':'#b75632'}
    for f in families:
        axes[0].plot([0,100,1000],[100*a['aggregate_endpoints'][f][str(step)][fields[1]]for step in (0,100,1000)],'-o',label=f,color=colors[f])
        for regime,metric,line in [('shared',shared[f],'-'),('independent',independent[f],'--')]:
            axes[1].plot(range(3),[100*metric['per_case'][key]['stress_relative_l2']for key in keys],line+'o',label=f+' '+regime,color=colors[f])
    axes[0].set_xticks([0,100,1000]);axes[0].set_xlabel('Full-node exposures per field');axes[0].set_title('Shared3 / mean signed stress')
    axes[1].set_xticks(range(3),keys,rotation=15);axes[1].set_title('Matched exposure1000 / shared vs independent')
    axes[0].set_ylabel('Physical stress relative L2 (%)')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_field.csv','comparison.png','representative_stress.png'];assert all((out/k).is_file()for k in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        actual_CPU_seconds=time.perf_counter()-began,files=files,new_GT_array_reads=0,PT_body_reads=0,GPU_calls=0))
    print(dict(status='completed',shared_Hash_S_gain_pp=package_gains[fields[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
