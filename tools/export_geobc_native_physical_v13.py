"""Terminal scalar-only joint TRAIN/knownDEV report; no arrays or inference."""
import argparse
import time

OWN = 'tools/export_geobc_native_physical_v13.py'
PLAN = 'notes/geobc_native_physical100_readout_v13_plan.json'
REVIEW = 'notes/geobc_native_physical_v13_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.native_physical_readout_contract_v13 import approved_contract, complete_physical, check_identity
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
    training=qualified['PP_NATIVE100_PHYSICAL_1500']
    variant=training['summary']; train_science=training['plan']['scientific']
    baseline=d.read(train_science['baseline_terminal']['summary']['path'])
    base_external=d.read(train_science['baseline_terminal']['external']['path'])
    fnorow=summary['candidates']['FNO100_PHYSICAL_1500']
    fb=fnorow['upstream_terminal_bindings']
    ftrain=d.read(check_identity(fb['summary'])); fexternal=d.read(check_identity(fb['external']))
    assert ftrain['status']=='completed' and ftrain['closed'] and not ftrain['active']
    assert fexternal['exit_code']==0 and fexternal['within_external_budget'] and fexternal['closed'] and not fexternal['active']
    assert all(d.exact_process_exited(fexternal[k]) for k in ('child_process','guard_process'))
    points=[]
    triples=[('PP_NATIVE100_1500','Native++ encoded',baseline,base_external),
             ('PP_NATIVE100_PHYSICAL_1500','Native++ physical',variant,training['external']),
             ('FNO100_PHYSICAL_1500','FNO physical',ftrain,fexternal)]
    fields=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
    for name,label,upstream,clock in triples:
        train=upstream['endpoints']['1500']['full_TRAIN']; dev=summary['candidates'][name]['full60']
        complete_physical(train,train_science['TRAIN_keys']); complete_physical(dev,science['case_keys'])
        points.append(dict(name=name,label=label,TRAIN={f:train[f] for f in fields},DEV={f:dev[f] for f in fields},
            training_epoch_seconds=upstream['training_epoch_seconds'],training_monotonic_seconds=upstream['training_monotonic_seconds'],
            stage_epoch_seconds=clock['elapsed_epoch_seconds'],stage_monotonic_seconds=clock['elapsed_monotonic_seconds'],
            inference_epoch_seconds=summary['candidates'][name]['inference_epoch_seconds'],
            inference_monotonic_seconds=summary['candidates'][name]['inference_seconds']))
    old,new,fno=points
    du=new['DEV'][fields[0]]-old['DEV'][fields[0]];ds=new['DEV'][fields[1]]-old['DEV'][fields[1]]
    th=science['DEV_gate'];passed=ds<=-th['stress_absolute_improvement_at_least'] and du<=th['U_absolute_deterioration_at_most']
    assert passed==summary['DEV_gate']['passed']
    gu=new['DEV'][fields[0]]-fno['DEV'][fields[0]];gs=new['DEV'][fields[1]]-fno['DEV'][fields[1]]
    out=d.path('results/geobc_native_physical_v13');assert not out.exists(),'No report replay';out.mkdir(parents=True)
    winners=summary['eligible_best']; assert winners
    lines=['# 100CAD／5遍：Native Transolver++ 与 FNO','',
        '同100个训练CAD×三载荷、1500更新，knownDEV20CAD×三载荷，全原节点物理误差。三行同时给出原++、新++和同物理损失的FNO。','',
        '| 方法与损失 | 位移误差 | 应力误差 | 热点误差 | 训练分钟 |','|---|---:|---:|---:|---:|']
    for point in points:
        m=point['DEV'];lines.append(f"| {point['label']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    lines+=['',f"新++减原++：位移{du*100:+.2f}、应力{ds*100:+.2f}个百分点；预设开发改善门槛{'通过' if passed else '未通过'}。",
        '',f"相同物理损失下，新++减FNO：位移{gu*100:+.2f}、应力{gs*100:+.2f}个百分点。联合U/S均值最优：{', '.join(winners)}。这决定后续优先级，不是最终求解器或架构上限。",
        '',f"原++／新++／FNO完整训练阶段{old['stage_epoch_seconds']/60:.2f}／{new['stage_epoch_seconds']/60:.2f}／{fno['stage_epoch_seconds']/60:.2f}分钟。新++本轮knownDEV读出{external['elapsed_epoch_seconds']:.2f}秒；60份新预测提交并重验后，读取既有已知开发场60次。原++和FNO仅复用不可变全60例标量，无推理重播、无原CSV重读。",
        '', 'TRAIN300、每载荷20例、配对20CAD、节点数／热点与物理单位指标和完整成本见metrics.json。same seed2026/order/Adam/oldTRAIN40-only scaler；两模型FP32最高精度，容量约4.53M但空间表示不同。',
        '', '物理损失同时改变病例幅值权重、U/S平衡、应力变换后的梯度及FP64点式精度，不能单独归因于某项。原++初始化hash未记录；初始化制度一致，新hash保存。新++最大完整N梯度实际通过。',
        '', '单seed、已知availability开发池；未来不可将本组当盲测。旧FINAL已经揭盲，本阶段FINAL读取0。signed U(mm)/S(MPa)原符号及全节点分母保留，连续查询资格不等于独立离节点FEM精度。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,TRAIN_native_variant=variant['endpoints']['1500']['full_TRAIN'],
        TRAIN_native_baseline=baseline['endpoints']['1500']['full_TRAIN'],TRAIN_FNO_physical=ftrain['endpoints']['1500']['full_TRAIN'],
        DEV_candidates=summary['candidates'],paired_CAD20=summary['paired_CAD20'],eligible_best=winners,
        TRAIN_gate=variant['TRAIN_gate'],DEV_gate=summary['DEV_gate'],physical_TRAIN_total_array_reads=variant['physical_TRAIN_total_array_reads'],
        actual_fresh_initial_native_model_sha256=variant['actual_fresh_initial_model_sha256'],
        native_peak_memory={k:variant[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.3),sharey=True)
    for axis,split,title in zip(axes,('TRAIN','DEV'),('TRAIN100 x 3','Known DEV20 x 3')):
        for j,point in enumerate(points):axis.bar([i+(j-1)*.24 for i in range(3)],[100*point[split][f] for f in fields],width=.24,label=point['label'])
        axis.set_xticks([0,1,2],['Displacement','Signed stress','Hotspot']);axis.set_title(title);axis.legend(fontsize=7)
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Same100CAD / 5 passes / native Transolver++ and FNO')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=plan['scientific_digest'],source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True),
        actual_CPU_seconds=time.monotonic()-began,numerical_array_reads=0,PT_reads=0,raw_GT_reads=0,GPU_calls=0,
        DEV_gate_passed=passed,final_solver_selected=False,files=['BRIEF.md','metrics.json','comparison.png']))
    print(dict(status='completed',DEV_gate_passed=passed,eligible_best=winners,CPU_seconds=time.monotonic()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
