"""Terminal scalar-only joint TRAIN/knownDEV report; no arrays or inference."""
import argparse
import time

OWN = 'tools/export_geobc_global_hash_physical_v18.py'
PLAN = 'notes/geobc_global_hash_physical_readout_v18_plan.json'
REVIEW = 'notes/geobc_global_hash_physical_v18_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash_physical_readout_contract_v18 import approved_contract, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN)
    plan, _, _, _, qualified = approved_contract(PLAN)
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output']); science = plan['scientific']
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status']=='completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['exit_code']==0 and external['within_external_budget'] and external['closed'] and not external['active']
    assert not external['external_timeout'] and all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    assert summary['execution']['run_process']==external['child_process']
    assert summary['execution']['guard']['guard_process']==external['guard_process']
    assert summary['execution']['scientific_digest']==external['scientific_digest']==plan['scientific_digest']
    assert summary['committed_finite_predictions']==60 and summary['owned_GT_array_reads']==summary['owned_GT_read_attempts']==60
    assert summary['raw_GT_reads']==summary['FINAL_numeric_reads']==summary['training_updates']==summary['scaler_fits']==0
    assert max(summary['elapsed_epoch_seconds'],summary['elapsed_monotonic_seconds'],external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds'])<=science['whole_budget_seconds']
    training = qualified['GLOBALHASH100_PHYSICAL_1500']
    variant = training['summary']; train_science = training['plan']['scientific']
    baseline = d.read(train_science['baseline_terminal']['summary']['path'])
    base_external = d.read(train_science['baseline_terminal']['external']['path'])
    points=[]
    for name,label,upstream,clock in [('FNO100_PHYSICAL_1500','FNO MLP',baseline,base_external),
                                    ('GLOBALHASH100_PHYSICAL_1500','FNO Global Hash',variant,training['external'])]:
        train = upstream['endpoints']['1500']['full_TRAIN']; dev = summary['candidates'][name]['full60']
        complete_physical(train,train_science['TRAIN_keys']); complete_physical(dev,science['case_keys'])
        fields = ('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
        points.append(dict(name=name,label=label,TRAIN={f:train[f] for f in fields},DEV={f:dev[f] for f in fields},
            training_epoch_seconds=upstream['training_epoch_seconds'],training_monotonic_seconds=upstream['training_monotonic_seconds'],
            stage_epoch_seconds=clock['elapsed_epoch_seconds'],stage_monotonic_seconds=clock['elapsed_monotonic_seconds']))
    old,new=points
    du = new['DEV']['case_mean_u_relative_l2']-old['DEV']['case_mean_u_relative_l2']
    ds = new['DEV']['case_mean_stress_relative_l2']-old['DEV']['case_mean_stress_relative_l2']
    gate = science['DEV_gate']; passed = .5*(du+ds)<=-gate['primary_absolute_improvement_at_least'] and max(du,ds)<=gate['maximum_either_field_absolute_deterioration']
    assert passed == summary['DEV_gate']['passed']
    out = d.path('results/geobc_global_hash_physical_v18'); assert not out.exists(),'No report replay'
    out.mkdir(parents=True)
    cost=variant['technical_recovery_cost']
    assert cost['original_optimizer_updates']==0 and cost['original_absolute_deadlines_retained']
    assert variant['actual_initial_FNO_backbone_identity']['passed']
    lines=['# 同物理损失／100CAD／5遍：MLP与全局Hash解码器','',
        '**两者同300训练病例、1500更新、FNO全局骨干与物理损失；下表为knownDEV20CAD×三载荷全节点误差。解码器容量与TCNN精度不同。**','',
        '| 解码器 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |','|---|---:|---:|---:|---:|']
    for point in points:
        m=point['DEV']
        lines.append(f"| {point['label']} | {100*m['case_mean_u_relative_l2']:.2f}% | {100*m['case_mean_stress_relative_l2']:.2f}% | {100*m['case_mean_stress_hotspot_relative_l2']:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    dh=new['DEV']['case_mean_stress_hotspot_relative_l2']-old['DEV']['case_mean_stress_hotspot_relative_l2']
    lines+=['',f"Hash减MLP：位移{du*100:+.2f}、应力{ds*100:+.2f}、热点{dh*100:+.2f}个百分点。完整60例联合U/S最佳：{', '.join(summary['eligible_best'])}。预设新增候选优先门槛{'通过' if passed else '未通过'}，不自动延长训练或宣布最终求解器。",'',
        'TRAIN300的U/S：MLP'+f"{100*old['TRAIN']['case_mean_u_relative_l2']:.2f}%/{100*old['TRAIN']['case_mean_stress_relative_l2']:.2f}%，Hash"+f"{100*new['TRAIN']['case_mean_u_relative_l2']:.2f}%/{100*new['TRAIN']['case_mean_stress_relative_l2']:.2f}%。",'',
        f"MLP完整阶段{old['stage_epoch_seconds']/60:.2f}分钟；Hash成功恢复段{new['stage_epoch_seconds']/60:.2f}分钟，另保留首0更新接口失败{cost['original_attempt_setup_cost_seconds']:.2f}秒与修复gap {cost['between_attempt_wall_epoch_seconds']:.2f}秒。原起点至训练child终态UTC跨度{cost['original_campaign_elapsed_epoch_seconds']/60:.2f}分钟，原截止未重置；epoch/monotonic分别列。新开发读出{external['elapsed_epoch_seconds']:.2f}秒。",'',
        '旧接口错误来自配置getter，恢复仅复用既有修复版；失败阶段0更新、无检查点，重建同seed原模型，不重播已有训练。新实际最大全节点反传通过；encoder/operator初始化与同seed原FNO构造器逐张量核对，不是从历史检查点提取初始化hash。', '',
        '两者原损失、seed2026、Adam .001、病例顺序、oldTRAIN40-only scaler相同；Hash仍为原固定12层网格，未微调结构或超参数。MLP约4.53M/FP32；GlobalHash约12.25M，FP32骨干＋TCNN FP16解码计算/FP32主参数，因此是实用解码器包比较，不是单因素容量／精度匹配消融。', '',
        '60份新有限预测提交并重验后共享knownGT60一次；MLP物理5遍只复用已有完整60例标量，没有基线推理或数组重播。rawCSV/FINAL/scalerfit为0；全部300/60、每载荷20、配对20CAD、最坏案例和成本见metrics.json。', '',
        'knownDEV已用于迭代，单seed、可用官方TRAIN来源；不是新盲测或无偏官方测试。连续接口不等于独立离节点FEM精度。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,TRAIN_variant=variant['endpoints']['1500']['full_TRAIN'],
        TRAIN_baseline=baseline['endpoints']['1500']['full_TRAIN'],DEV_candidates=summary['candidates'],paired_CAD20=summary['paired_CAD20'],
        TRAIN_gate=variant['TRAIN_gate'],DEV_gate=summary['DEV_gate'],physical_TRAIN_total_array_reads=variant['physical_TRAIN_total_array_reads'],
        actual_fresh_initial_model_sha256=variant['actual_fresh_initial_model_sha256'],technical_recovery_cost=cost,actual_initial_FNO_backbone_identity=variant['actual_initial_FNO_backbone_identity'],model_profiles={'FNO':baseline['model_profile'],'GlobalHash':variant['model_profile']},peak_memory={k:variant[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.0),sharey=True)
    fields=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
    for axis,split,title in zip(axes,('TRAIN','DEV'),('TRAIN100 x 3','Known DEV20 x 3')):
        for j,point in enumerate(points):
            axis.bar([i+(j-.5)*.32 for i in range(3)],[100*point[split][f] for f in fields],width=.32,label=point['label'])
        axis.set_xticks([0,1,2],['Displacement','Signed stress','Hotspot']);axis.set_title(title);axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Same physical objective / FNO100 / 5 passes / decoder comparison')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=plan['scientific_digest'],source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True),
        actual_CPU_seconds=time.monotonic()-began,numerical_array_reads=0,PT_reads=0,raw_GT_reads=0,GPU_calls=0,
        DEV_gate_passed=passed,final_solver_selected=False,files=['BRIEF.md','metrics.json','comparison.png']))
    print(dict(status='completed',points=points,DEV_gate_passed=passed,CPU_seconds=time.monotonic()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:
        run()
