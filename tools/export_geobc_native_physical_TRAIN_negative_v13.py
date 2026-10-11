"""Actual TRAIN gate negative; scalar-only report, no development inference."""
import argparse
import time

OWN='tools/export_geobc_native_physical_TRAIN_negative_v13.py'
REVIEW='notes/geobc_native_physical_TRAIN_negative_v13_exporter_root_review.json'


def run():
    began=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.native_physical_contract_v13 import approved_contract
    from geobc.physical_readout_contract_v12 import approved_contract as fno_readout,complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    plan,_,manifest,_=approved_contract('notes/geobc_macro100_PP256_NATIVE_physical5_v13_plan.json')
    assert marker['scientific_digest']==plan['scientific_digest']
    out=d.path(plan['output']);summary=d.read(out/'summary.json');external=d.read(out/'external_terminal.json')
    assert summary==d.read(out/'status.json') and summary['status']=='completed' and summary['training_complete']
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code']==0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('guard_process','child_process'))
    assert summary['execution']['run_process']==external['child_process'] and summary['execution']['guard']['guard_process']==external['guard_process']
    assert summary['execution']['scientific_digest']==external['scientific_digest']==plan['scientific_digest']
    assert not summary['TRAIN_gate']['passed'] and summary['DEV_FINAL_reads']==summary['raw_GT_reads']==summary['scaler_fits']==0
    assert summary['step']==summary['optimizer_step_attempts']==summary['optimizer_step_completions']==1500
    assert summary['case_exposure_counts']=={k:5 for k in plan['scientific']['TRAIN_keys']}
    assert not summary['possible_partial_optimizer_update'] and summary['actual_largest_fullN_gradient_check']['passed']
    assert summary['physical_TRAIN_total_array_reads']==600
    budget=plan['scientific']
    assert max(summary['training_epoch_seconds'],summary['training_monotonic_seconds'])<=budget['training_budget_seconds']
    assert max(summary['cache_setup_epoch_seconds'],summary['cache_setup_monotonic_seconds'])<=budget['cache_setup_budget_seconds']
    assert max(summary['elapsed_epoch_seconds'],summary['elapsed_monotonic_seconds'],external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds'])<=budget['whole_budget_seconds']
    assert max(summary[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=budget['memory_limit_bytes']
    assert max(summary['endpoints']['1500'][k] for k in ('epoch_seconds','monotonic_seconds'))<=budget['TRAIN_readout_budget_seconds']
    _,_,fm,_,fq=fno_readout('notes/geobc_physical100_readout_v12_plan.json');assert fm==manifest
    fno=fq['FNO100_PHYSICAL_1500']
    baseline=d.read(plan['scientific']['baseline_terminal']['summary']['path'])
    be=d.read(plan['scientific']['baseline_terminal']['external']['path'])
    fields=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
    points=[]
    for name,ss,ee in [('Native++ encoded',baseline,be),('Native++ physical',summary,external),('FNO physical',fno['summary'],fno['external'])]:
        metric=ss['endpoints']['1500']['full_TRAIN'];complete_physical(metric,plan['scientific']['TRAIN_keys'])
        points.append(dict(name=name,metrics={k:metric[k] for k in fields},training_epoch_seconds=ss['training_epoch_seconds'],
            training_monotonic_seconds=ss['training_monotonic_seconds'],whole_epoch_seconds=ee['elapsed_epoch_seconds'],whole_monotonic_seconds=ee['elapsed_monotonic_seconds']))
    thresholds=plan['scientific']['TRAIN_gate'];du=points[1]['metrics'][fields[0]]-points[0]['metrics'][fields[0]];ds=points[1]['metrics'][fields[1]]-points[0]['metrics'][fields[1]]
    assert not (ds<=-thresholds['stress_absolute_improvement_at_least'] and du<=thresholds['U_absolute_deterioration_at_most'])
    directory=d.path('results/geobc_native_physical_TRAIN_v13');assert not directory.exists(),'No report replay';directory.mkdir(parents=True)
    lines=['# 100CAD／5遍：Native++物理损失未通过训练门槛','',
        '**下表均为TRAIN100CAD×三载荷、300病例的全原节点误差，不能解释为新几何验证误差。**','',
        '| 方法与损失 | 位移误差 | 应力误差 | 热点误差 | 训练分钟 |','|---|---:|---:|---:|---:|']
    for p in points:
        m=p['metrics'];lines.append(f"| {p['name']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {p['training_epoch_seconds']/60:.2f} |")
    lines+=['',f"新++减原++：位移{100*du:+.2f}、应力{100*ds:+.2f}个百分点。预设应力至少降低5个百分点、位移恶化不超过2个百分点的门槛未通过。训练正常完成，属于科学负结果；不改变门槛、不重播、不追加++训练。",
        '',f"Native++新完整阶段{external['elapsed_epoch_seconds']/60:.2f}分钟，含缓存、完整N梯度检查、1500次Adam、检查点和TRAIN300读出。原checkpoint／四RNG／曝光及双时钟保留。",
        '', '新Native++ knownDEV预测、读标签、导出均未执行；预先准备的DEV程序不等于新结果。FNO物理5遍已有knownDEV位移／应力31.61%／56.10%，该结果见FNO简报，不与本表训练误差混列。',
        '', '因此当前优先继续FNO物理损失5→10遍，保留其模型、Adam、四RNG和cursor，检验额外5遍的收益；尚无本简报中的10遍指标。TF32速度探针暂缓，提速本身不能解决当前++的拟合问题。',
        '', '同100CAD／5遍／seed2026／病例顺序／Adam／FP32最高精度和oldTRAIN40-only scaler，容量约4.53M但FNO与native空间表示不同。该配方下拟合差，不能判定Transolver架构上限；不是作者原版DeepJEB精确配置。signed U(mm)/S(MPa)保持；全部300病例分母与成本见metrics.json；未读取或重用FINAL。']
    (directory/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(directory/'metrics.json',dict(points=points,native_TRAIN300=summary['endpoints']['1500']['full_TRAIN'],native_encoded_TRAIN300=baseline['endpoints']['1500']['full_TRAIN'],
        FNO_physical_TRAIN300=fno['summary']['endpoints']['1500']['full_TRAIN'],TRAIN_gate=summary['TRAIN_gate'],new_native_DEV_actual_execution=0,
        native_actual_fresh_initial_model_sha256=summary['actual_fresh_initial_model_sha256'],peak_memory={k:summary[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(8.3,4.4))
    for j,p in enumerate(points):axis.bar([i+(j-1)*.24 for i in range(3)],[100*p['metrics'][f] for f in fields],width=.24,label=p['name'])
    axis.set_xticks([0,1,2],['Displacement','Signed stress','Hotspot']);axis.set_ylabel('Case mean relative L2 (%)')
    axis.set_title('TRAIN100 x 3 / same5passes / not unseen-CAD accuracy');axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(directory/'comparison.png',dpi=160);plt.close(fig)
    assert all((directory/n).is_file() for n in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(directory/'manifest.json',dict(status='completed',SCI=plan['scientific_digest'],source_sha256=d.sha(OWN),summary_identity=d.identity(out/'summary.json',hashed=True),
        external_identity=d.identity(out/'external_terminal.json',hashed=True),actual_CPU_seconds=time.monotonic()-began,numerical_array_reads=0,PT_reads=0,raw_GT_reads=0,GPU_calls=0,
        scientific_TRAIN_gate_negative=True,new_native_DEV_actual_execution=0,files=['BRIEF.md','metrics.json','comparison.png']))
    print(dict(status='completed',native_TRAIN_gate_passed=False,CPU_seconds=time.monotonic()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
