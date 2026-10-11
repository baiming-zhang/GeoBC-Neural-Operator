"""Scalar-only actual three-field100/300/1000 learning curve export."""
import argparse
import time

OWN='tools/export_geobc_single_field_v59.py'
REVIEW='notes/geobc_single_field_v59_exporter_root_review.json'
PLAN='notes/geobc_single_field_v59_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.single_field_learning_contract_v59 import terminal,complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['selected_TRAIN_keys']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==2700 and not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==a['parent_checkpoint_hash_reads']==a['parent_checkpoint_loads']==3
    assert a['encoded_TRAIN_reads']==a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    for key in keys:
        row=a['fields'][key];q=row['actual_fullN_gradient_check']
        assert row['status']=='completed' and row['additional_updates']==900 and row['optimizer_step']==7000
        assert row['total_additional_updates_since_shared20']==1000 and row['baseline_parent_scalar_match']
        assert row['restored_model_Adam_fourRNG'] and row['parent_checkpoint_unchanged']
        assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored']
        assert row['case_exposure_counts']=={k:(1020 if k==key else 20)for k in s['parent_TRAIN_keys']}
        assert set(row['checkpoint_identities'])=={'300','1000'}
    metrics=a['aggregate_endpoints'];fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    for metric in metrics.values():complete_physical(metric,keys)
    gains={k:metrics['100'][k]-metrics['1000'][k]for k in fields}
    below1={step:sum(metric['per_case'][key]['stress_relative_l2']<.01 for key in keys)for step,metric in metrics.items()}
    out=d.path('results/geobc_single_field_v59');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    text=['# 固定3个难场：单场学习到1000次更新','',
        '**从各自独立100检查点恢复model／Adam6100／四类RNG，每场再900次完整节点更新；同FNO9、物理U(mm)/signed S(MPa)联合损失。100／300／1000均为原共享20遍之后的额外单场更新。**','',
        '| 每场累计额外更新 | 位移误差 | 应力误差 | 热点误差 |','|---|---:|---:|---:|---:|']
    for step in ('100','300','1000'):
        m=metrics[step];text.append(f'| {step} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% |')
    text+=['',f'100→1000：应力改善{100*gains[fields[1]]:.2f}个百分点，位移{100*gains[fields[0]]:.2f}、热点{100*gains[fields[2]]:.2f}个百分点；应力低于1%的场{below1["1000"]}/3。','',
        f'新增2700次全节点更新，完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，连续学习区间{a["training_epoch_seconds"]:.2f}秒（包括恢复、资格检查、读出与保存），CPU报告另计。原共享0→20及此前诊断成本完整保留。','',
        '固定53_482/ver、229_200/ver、56_293/hor，按原共享20遍训练指标选择；不是验证／测试或随机样本。三个模型分别通过真实完整节点梯度与自身100端点重放核对；最终Adam7000、该场曝光1020、其余299场历史曝光20。原检查点不变，固定端点不选best。','',
        'TRAIN物理3数组各读一次、端点常驻复用；父PT hash/load各3，encoded／DEV／FINAL／原CSV读取及scaler拟合均0。独立模型不具备未见CAD算子资格，也不代表共享训练效果。','',
        '这只检验固定单场的学习趋势，不证明收敛或架构上限。与原slides单场1000次采样更新的任务／曝光不同，不能据此宣称精确复现；网格指标与原节点指标保持分离。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,
        aggregate_endpoints=metrics,gains100to1000=gains,below1percent=below1,
        fitting_only=True,DEV_FINAL_reads=0,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.5))
    x=[100,300,1000]
    for key in keys:
        axes[0].plot(x,[100*a['fields'][key]['endpoints'][str(step)]['per_case'][key]['stress_relative_l2']for step in x],'-o',label=key)
    axes[0].set_title('Selected individual signed stress');axes[0].legend(fontsize=8)
    for field,label in zip(fields,['Displacement','Signed stress','Stress hotspot']):
        axes[1].plot(x,[100*metrics[str(step)][field]for step in x],'-o',label=label)
    axes[1].set_title('Same three fields / mean');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x);ax.set_xlabel('Extra full-node updates after shared20');ax.grid(alpha=.2)
    axes[0].set_ylabel('Physical relative L2 (%)');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/k).is_file()for k in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        actual_CPU_seconds=time.perf_counter()-began,files=['BRIEF.md','metrics.json','comparison.png'],new_GT_array_reads=0,PT_body_reads=0,GPU_calls=0))
    print(dict(status='completed',S_gain_pp=100*gains[fields[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
