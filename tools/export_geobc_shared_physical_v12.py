"""Terminal scalar-only joint TRAIN/knownDEV report; no arrays or inference."""
import argparse
import time

OWN = 'tools/export_geobc_shared_physical_v12.py'
PLAN = 'notes/geobc_physical100_readout_v12_plan.json'
REVIEW = 'notes/geobc_shared_physical_v12_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.physical_readout_contract_v12 import approved_contract, complete_physical
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
    training = qualified['FNO100_PHYSICAL_1500']
    variant = training['summary']; train_science = training['plan']['scientific']
    baseline = d.read(train_science['baseline_terminal']['summary']['path'])
    base_external = d.read(train_science['baseline_terminal']['external']['path'])
    points=[]
    for name,label,upstream,clock in [('FNO100_1500','Encoded MSE',baseline,base_external),
                                    ('FNO100_PHYSICAL_1500','Physical relative',variant,training['external'])]:
        train = upstream['endpoints']['1500']['full_TRAIN']; dev = summary['candidates'][name]['full60']
        complete_physical(train,train_science['TRAIN_keys']); complete_physical(dev,science['case_keys'])
        fields = ('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
        points.append(dict(name=name,label=label,TRAIN={f:train[f] for f in fields},DEV={f:dev[f] for f in fields},
            training_epoch_seconds=upstream['training_epoch_seconds'],training_monotonic_seconds=upstream['training_monotonic_seconds'],
            stage_epoch_seconds=clock['elapsed_epoch_seconds'],stage_monotonic_seconds=clock['elapsed_monotonic_seconds']))
    old,new=points
    du = new['DEV']['case_mean_u_relative_l2']-old['DEV']['case_mean_u_relative_l2']
    ds = new['DEV']['case_mean_stress_relative_l2']-old['DEV']['case_mean_stress_relative_l2']
    gate = science['DEV_gate']; passed = ds<=-gate['stress_absolute_improvement_at_least'] and du<=gate['U_absolute_deterioration_at_most']
    assert passed == summary['DEV_gate']['passed']
    out = d.path('results/geobc_shared_physical_v12'); assert not out.exists(),'No report replay'
    out.mkdir(parents=True)
    lines=['# FNO100／5遍：共享训练损失对照','',
        '**同架构、同100CAD／300训练病例、同1500更新；下表为20个已知开发CAD×三载荷的全节点物理误差。**','',
        '| 损失 | 位移误差 | 应力误差 | 热点误差 | 训练分钟 |','|---|---:|---:|---:|---:|']
    for point in points:
        m=point['DEV']
        lines.append(f"| {point['label']} | {100*m['case_mean_u_relative_l2']:.2f}% | {100*m['case_mean_stress_relative_l2']:.2f}% | {100*m['case_mean_stress_hotspot_relative_l2']:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    lines+=['',f"物理损失减原损失：位移{du*100:+.2f}、应力{ds*100:+.2f}个百分点。预设开发门槛{'通过' if passed else '未通过'}；这只影响下一阶段优先级，不是最终求解器或新盲测。",
        '', 'TRAIN300的U/S：原损失'+f"{100*old['TRAIN']['case_mean_u_relative_l2']:.2f}%/{100*old['TRAIN']['case_mean_stress_relative_l2']:.2f}%，新损失"+f"{100*new['TRAIN']['case_mean_u_relative_l2']:.2f}%/{100*new['TRAIN']['case_mean_stress_relative_l2']:.2f}%。",
        '', f"原／新完整训练阶段{old['stage_epoch_seconds']/60:.2f}／{new['stage_epoch_seconds']/60:.2f}分钟，含缓存、最大全节点梯度、检查点及TRAIN读出。新开发评估{external['elapsed_epoch_seconds']:.2f}秒；60份新预测提交并重验后共享owned已知GT60一次。基线复用v11既有完整60例标量，不重跑其推理，不重读原CSV。",
        '', '同seed2026、病例顺序、Adam、FP32模型、oldTRAIN40-only scaler，未重新拟合scaler。新损失使用原TRAIN物理场、固定全节点范数及已有FP64点式逆变换；不clamp、不伪造符号。基线原始初始化hash未记录；初始化源码／种子制度一致，新初始化hash已保存。',
        '', '损失同时改变病例幅值权重、U/S平衡、应力变换后的梯度及点式精度，不能单独归因某一项。单seed／已知availability开发池；FINAL未读取或用于选型。全部300／60例、每载荷20例、配对20CAD与完整成本见metrics.json。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,TRAIN_variant=variant['endpoints']['1500']['full_TRAIN'],
        TRAIN_baseline=baseline['endpoints']['1500']['full_TRAIN'],DEV_candidates=summary['candidates'],paired_CAD20=summary['paired_CAD20'],
        TRAIN_gate=variant['TRAIN_gate'],DEV_gate=summary['DEV_gate'],physical_TRAIN_total_array_reads=variant['physical_TRAIN_total_array_reads'],
        actual_fresh_initial_model_sha256=variant['actual_fresh_initial_model_sha256'],peak_memory={k:variant[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.0),sharey=True)
    fields=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
    for axis,split,title in zip(axes,('TRAIN','DEV'),('TRAIN100 x 3','Known DEV20 x 3')):
        for j,point in enumerate(points):
            axis.bar([i+(j-.5)*.32 for i in range(3)],[100*point[split][f] for f in fields],width=.32,label=point['label'])
        axis.set_xticks([0,1,2],['Displacement','Signed stress','Hotspot']);axis.set_title(title);axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Same FNO100 / 5 passes / training objective comparison')
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
