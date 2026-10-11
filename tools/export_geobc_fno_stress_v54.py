"""One scalar-only TRAIN300 matched-loss report; no checkpoint or arrays."""
import argparse
import time

OWN='tools/export_geobc_fno_stress_v54.py'
REVIEW='notes/geobc_fno_stress_v54_exporter_root_review.json'
PLAN='notes/geobc_fno_stress_v54_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.fno_stress_contract_v54 import terminal,metadata_inputs
    from geobc.global_hash_physical_readout_contract_v18 import complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];up=metadata_inputs()
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==1500
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==3000
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    assert a['case_exposure_counts']=={k:10 for k in s['TRAIN_keys']}
    assert a['actual_largest_fullN_gradient_check']['passed']
    assert a['actual_largest_fullN_gradient_check']['model_unchanged'] and a['actual_largest_fullN_gradient_check']['Adam_unchanged']
    assert a['actual_largest_fullN_gradient_check']['fourRNG_restored']
    assert a['parent_checkpoint_loads']==a['parent_checkpoint_hash_reads']==1
    assert a['physical_TRAIN_total_array_reads']==a['resident_TRAIN_physical_target_reuses']==300
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['cached_TRAIN_physical_reads']==0
    base=up['control_summary']['endpoints']['3000']['full_TRAIN'];new=a['endpoints']['3000']['full_TRAIN']
    for metric in [base,new]:complete_physical(metric,s['TRAIN_keys'])
    fs='case_mean_stress_relative_l2';fh='case_mean_stress_hotspot_relative_l2'
    ds=new[fs]-base[fs];dh=new[fh]-base[fh]
    passed=ds<=-.03 and dh<=.02
    assert passed==a['TRAIN_gate']['passed']
    out=d.path('results/geobc_fno_stress_v54');assert not out.exists(),'One immutable report'
    out.mkdir(parents=True)
    decision='值得进入独立的knownDEV读出，仍需另行释放' if passed else '未达到预设收益，不追加训练或自动读取DEV'
    text=['# 同父检查点：移除位移损失是否改善应力拟合？','',
        '**100个TRAIN CAD×三载荷，全原节点physical signed S(MPa)。从同一个5遍检查点追加5遍，只移除U项；复用既有联合10遍对照。**','',
        '| 总10遍目标 | TRAIN应力误差 | TRAIN热点误差 | 追加训练秒 |','|---|---:|---:|---:|']
    for label,m,seconds in [('联合U/S',base,up['control_summary']['training_epoch_seconds']),('仅应力',new,a['training_epoch_seconds'])]:
        text.append(f'| {label} | {100*m[fs]:.2f}% | {100*m[fh]:.2f}% | {seconds:.2f} |')
    text+=['',f'仅应力减联合：应力{100*ds:+.2f}、热点{100*dh:+.2f}个百分点。{decision}。','',
        '当前结果只检验这个共享FNO训练状态下两种监督的影响；不能据此宣布架构上限、排除其他优化问题或证明新CAD精度。单样本Hash的小误差对应已知场拟合。','',
        f'实际新训练1500次更新；整个新阶段{e["elapsed_epoch_seconds"]:.2f}秒。Adam、四类RNG、病例顺序、scaler、表示及原应力项0.5权重保持；真实最大全节点反传已通过，检查时model/Adam不变且RNG恢复。','',
        'TRAIN标签只加载一次300份，端点复用GPU常驻标签300次；新DEV／FINAL／原CSV读取及scaler拟合均0。仅应力分支的原始位移输出未在新增训练中受监督，不能当成联合求解器结果。','',
        '完整300例应力／热点／最差及原始位移诊断、原始epoch与monotonic成本见metrics.json。不自动追加40遍或24000步。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(control_TRAIN300=base,specialist_TRAIN300=new,TRAIN_gate=a['TRAIN_gate'],
        specialist_cost=a,control_terminal_binding=up['control_binding'],specialist_terminal_binding=b,
        scope='TRAIN-only shared fitting diagnostic; raw specialist U is not a composite solver',DEV_readout_completed=False,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.7,3.6))
    for i,(label,m) in enumerate([('Joint U/S',base),('Stress only',new)]):
        positions=[j+(i-.5)*.32 for j in range(2)];values=[100*m[k] for k in [fs,fh]]
        bars=ax.bar(positions,values,width=.32,label=label)
        ax.bar_label(bars,fmt='%.2f',fontsize=9)
    ax.set_xticks([0,1],['Signed stress','Stress hotspot']);ax.set_ylabel('Mean relative L2 (%)')
    ax.set_title('TRAIN100 x 3 / same parent, same 10 exposures');ax.legend()
    ax.set_ylim(0,max(100*m[k]for m in [base,new]for k in [fs,fh])*1.23)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/k).is_file()for k in ['BRIEF.md','metrics.json','comparison.png'])
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),
        actual_CPU_seconds=time.perf_counter()-began,terminal_binding=b,TRAIN_gate_passed=passed,
        files=['BRIEF.md','metrics.json','comparison.png'],GPU_calls=0,new_GT_array_reads=0,PT_body_reads=0))
    print(dict(status='completed',TRAIN_gate_passed=passed,CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
