"""Scalar-only actual shared/independent fitting diagnostic report."""
import argparse
import time

OWN='tools/export_geobc_shared_fit_v58.py'
REVIEW='notes/geobc_shared_fit_v58_exporter_root_review.json'
PLAN='notes/geobc_shared_fit_v58_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.shared_fit_contract_v58 import terminal,complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['selected_TRAIN_keys']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==1800 and not a['optimizer_step_in_progress']
    assert not a['possible_partial_optimizer_update'] and a['baseline_parent_scalar_match'] and a['parent_checkpoint_unchanged']
    assert a['actual_largest_selected_fullN_gradient_check']['passed']
    assert a['physical_TRAIN_attempts']==a['physical_TRAIN_reads']==9
    assert a['parent_checkpoint_hash_reads']==a['parent_checkpoint_loads']==1
    assert a['encoded_TRAIN_reads']==a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    full=s['parent_TRAIN_keys'];shared=a['arms']['shared9']['shared9']
    assert shared['status']=='completed' and shared['additional_updates']==900 and shared['optimizer_step']==6900
    assert shared['case_exposure_counts']=={k:(120 if k in keys else 20)for k in full}
    for key in keys:
        row=a['arms']['independent9'][key]
        assert row['status']=='completed' and row['additional_updates']==100 and row['optimizer_step']==6100
        assert row['case_exposure_counts']=={k:(120 if k==key else 20)for k in full}
        assert row['parent_model_digest']==shared['parent_model_digest']==a['parent_model_digest']
        assert row['parent_Adam_digest']==shared['parent_Adam_digest']==a['parent_Adam_digest']
    metrics=[a['baseline_selected9'],a['shared9_final'],a['independent9_final']]
    for metric in metrics:complete_physical(metric,keys)
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    gap={k:metrics[1][k]-metrics[2][k]for k in fields}
    below1={label:sum(metric['per_case'][key]['stress_relative_l2']<.01 for key in keys)
        for label,metric in zip(['parent20','shared9','independent9'],metrics)}
    paired={cad:{field:sum(a['shared9_final']['per_case'][cad+'/'+l][field]-a['independent9_final']['per_case'][cad+'/'+l][field]for l in ('ver','hor','tor'))/3
        for field in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2')}for cad in s['selected_CADs']}
    out=d.path('results/geobc_shared_fit_v58');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    text=['# 高误差训练场：共享与独立拟合','',
        '**固定TRAIN20应力平均误差最高的3个CAD、三载荷共9场。同FNO20父model／Adam／四类RNG、相同完整节点物理U/S损失；两组每场额外100遍、各900次更新。**','',
        '| 配置 | 位移误差 | 应力误差 | 热点误差 | 追加更新 |','|---|---:|---:|---:|---:|']
    for label,m,count in zip(['原共享100CAD模型20遍','仅这9场共享拟合','9场分别独立拟合'],metrics,[0,900,900]):
        text.append(f'| {label} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {count} |')
    text+=['',f'独立相对共享应力改善{100*gap[fields[1]]:.2f}个百分点，位移{100*gap[fields[0]]:.2f}、热点{100*gap[fields[2]]:.2f}个百分点。应力低于1%的场：共享{below1["shared9"]}/9，独立{below1["independent9"]}/9。','',
        f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，其中连续训练阶段{a["training_epoch_seconds"]:.2f}秒（含模型重置、固定端点读出与保存）；父0→20训练成本另保留，不按9个模型重复计算历史共享训练。','',
        '实际242,936节点完整梯度检查通过，model／Adam不变、四类RNG恢复，初始9场预测与父20遍标量一致。物理TRAIN标签9份各读一次；端点复用常驻标签，DEV／FINAL／原CSV／encoded标签读取及scaler拟合均0。','',
        '这是高误差已知场拟合诊断：独立组有9倍模型总容量与9条Adam轨迹，不能单独归因于梯度冲突；共享9场也改变了训练范围，不能当作全100CAD的改进。100追加遍的非收敛结果不代表单场容量上限，独立模型不具备未见CAD求解器资格。','',
        '原slides约0.08%是已知少场拟合；当前固定100追加遍若未达到该水平，不能宣称复现。模型、表示与训练曝光仍不同。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,
        shared_minus_independent=gap,below1percent=below1,paired_CAD3=paired,
        fitting_only=True,DEV_FINAL_reads=0,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.5))
    for ax,field,title in zip(axes,fields[1:],['Signed stress','Stress hotspot']):
        for arm,label in [('shared9','Shared9'),('independent9','Independent9')]:
            points=[metrics[0][field]]
            for step in ('20','100'):
                if arm=='shared9':value=shared['endpoints'][step][field]
                else:value=sum(a['arms'][arm][k]['endpoints'][step]['per_case'][k][field.replace('case_mean_','')]for k in keys)/9
                points.append(value)
            ax.plot([0,20,100],[100*x for x in points],'-o',label=label)
        ax.set_title(title);ax.set_xticks([0,20,100]);ax.set_xlabel('Additional complete case exposures');ax.grid(alpha=.2)
    axes[0].set_ylabel('Selected TRAIN9 mean relative L2 (%)');axes[1].legend();fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/k).is_file()for k in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),
        terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        files=['BRIEF.md','metrics.json','comparison.png'],new_GT_array_reads=0,PT_body_reads=0,GPU_calls=0))
    print(dict(status='completed',shared_minus_independent_S_pp=100*gap[fields[1]],CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
