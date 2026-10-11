"""Scalar-only brief for the independently scoped same12k comparison."""
import argparse
from geobc import available_cad_pilot_data as d


def run():
    plan=d.read('notes/geobc_short_screen_readout_plan.json'); folder=d.path(plan['output'])
    summary=d.read(folder/'summary.json'); outer=d.read(folder/'external_terminal.json')
    assert summary['status']=='completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert outer['closed'] and not outer['active'] and outer['exit_code']==0 and outer['within_external_budget']
    assert all(d.exact_process_exited(outer[k]) for k in ('guard_process','child_process'))
    assert summary['execution']['scientific_digest']==outer['scientific_digest']==plan['scientific_digest']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    assert summary['committed_finite_predictions']==120 and summary['owned_GT_array_reads']==60
    assert summary['training_updates']==summary['raw_GT_reads']==summary['scaler_fits']==0
    output=d.path('results/geobc_short_screen'); output.mkdir(parents=True,exist_ok=True)
    assert not (output/'manifest.json').exists(), 'One actual export'
    rows=summary['candidates']
    for row in rows.values(): assert row['full60']['all_cases_complete']
    # Key names follow the genuine aggregate writer; no recomputed arrays/inference.
    d.write(output/'metrics.json',summary)
    names=['FNO9_step12000','PP256_step12000']
    labels=['FNO','Transolver++']
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    lines=['# 同12000更新：已知开发20 CAD×三载荷','',
        '两方法均100遍TRAIN120，完整60病例、全部原节点、signed U(mm)/S(MPa)。这是已知开发池、单seed结果，不是最终测试或架构上限。原双24000实验主动终止，未冒充完成。','',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 60例推理秒 |','|---|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        row=rows[name]; values=[100*row['full60'][f] for f in fields]
        lines.append(f"| {label} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% | {row['inference_seconds']:.2f} |")
    gap=100*(rows[names[1]]['full60']['selection_score']-rows[names[0]]['full60']['selection_score'])
    paired={cad:sum(.5*(rows[names[1]]['full60']['per_case'][cad+'/'+load]['u_relative_l2']+
        rows[names[1]]['full60']['per_case'][cad+'/'+load]['stress_relative_l2']-
        rows[names[0]]['full60']['per_case'][cad+'/'+load]['u_relative_l2']-
        rows[names[0]]['full60']['per_case'][cad+'/'+load]['stress_relative_l2']) for load in ('ver','hor','tor'))/3
        for cad in sorted({key.split('/')[0] for key in plan['scientific']['case_keys']})}
    better=sum(value>0 for value in paired.values())
    lines+=['',f'++减FNO的联合误差差值：**{gap:.2f}个百分点**；FNO较低的CAD为{better}/20。原始全60／每载荷20／最差／热点标量见metrics.json。',
        '', '下一步比较FNO、原生节点Transolver++、条件化Hash的10遍短筛，每候选训练上限30分钟；只有有竞争力的候选再加训练预算。',
        '',f"本次读出UTC耗时{summary['elapsed_epoch_seconds']:.2f}秒，monotonic耗时{summary['elapsed_monotonic_seconds']:.2f}秒；新增训练0，完整120份预测验证后共享已知GT60一次。部分原训练中断成本未知，保留不填零。"]
    (output/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    for j,(name,label) in enumerate(zip(names,labels)):
        ax.bar([i+(j-.5)*.32 for i in range(3)], [100*rows[name]['full60'][f] for f in fields],width=.32,label=label)
    ax.set_xticks(range(3),['Displacement','Signed stress','Stress hotspot']);ax.set_ylabel('Case mean relative L2 (%)')
    ax.set_title('Known DEV: 20 CAD x 3 loads; both 12,000 updates');ax.legend();fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    d.write(output/'manifest.json',dict(status='completed',scientific_digest=plan['scientific_digest'],
        full_case_denominator=60, prediction_denominator=120, paired_CAD_gaps=paired,
        PP_minus_FNO_score_percentage_points=gap,FNO_better_CAD=better,
        original_24000_campaign_completed=False,final_solver_selected=False,
        files=['BRIEF.md','metrics.json','comparison.png','terminal_receipt.json']))
    print({'joint_gap_pp':gap,'FNO_better_CAD':better,'readout_seconds':summary['elapsed_epoch_seconds']})
    d.write(output/'terminal_receipt.json',dict(scientific_digest=plan['scientific_digest'],
        summary_identity=d.identity(folder/'summary.json',hashed=True),
        external_identity=d.identity(folder/'external_terminal.json',hashed=True),
        exact_processes_exited=True, actual_new_training_updates=0, raw_GT_reads=0,
        scalar_export_UTC=d.utc(), original_24000_campaign_completed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export: run()
