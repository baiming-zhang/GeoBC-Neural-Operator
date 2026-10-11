"""Scalar-only whole-context Hash comparison and head-reliance diagnostics."""
import argparse
import time

OWN='tools/export_geobc_operator_hash_v5.py'
PLAN='notes/geobc_operator_hash_readout_v5_plan.json'
REVIEW='notes/geobc_operator_hash_v5_exporter_review.json'


def run():
    start=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.operator_hash_readout_contract_v5 import approved_contract
    from tools.export_geobc_exposure_comparison import paired
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256']==d.sha(OWN)
    plan,_,_,_,qualified=approved_contract(PLAN)
    assert marker['scientific_digest']==plan['scientific_digest']
    folder=d.path(plan['output']);summary=d.read(folder/'summary.json');outer=d.read(folder/'external_terminal.json')
    assert summary['status']=='completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert outer['exit_code']==0 and outer['within_external_budget'] and outer['closed'] and not outer['active']
    assert all(d.exact_process_exited(outer[k]) for k in ('child_process','guard_process'))
    assert summary['execution']['scientific_digest']==outer['scientific_digest']==plan['scientific_digest']
    assert summary['committed_finite_predictions']==240 and summary['owned_GT_array_reads']==60
    assert summary['raw_GT_reads']==summary['FINAL_numeric_reads']==summary['training_updates']==summary['scaler_fits']==0
    reference_plan=d.read('notes/geobc_fno_coverage_readout_v4_plan.json')
    assert reference_plan['scientific']['case_keys']==plan['scientific']['case_keys']
    assert reference_plan['scientific']['scaler']==plan['scientific']['scaler']
    assert reference_plan['scientific']['GT_manifest_identity']==plan['scientific']['GT_manifest_identity']
    reference=d.read(d.path(reference_plan['output'])/'summary.json')
    assert reference['status']=='completed' and reference['protocol_qualified']
    keys=plan['scientific']['case_keys'];full={'FNO100_4800':reference['candidates']['FNO100_step4800']}
    full.update({n:summary['candidates'][n] for n in ('HASH_PN100_4800','GLOBAL_HASH100_4800')})
    points=[];costs={}
    labels={'FNO100_4800':'FNO','HASH_PN100_4800':'Geometry Hash','GLOBAL_HASH100_4800':'Whole-BC operator Hash'}
    for name,row in full.items():
        metric=row['full60'];assert metric['all_cases_complete'] and metric['fixed_denominator']==60 and set(metric['per_case'])==set(keys)
        assert row['step']==4800 and row['train_CAD_count']==100 and row['case_exposure_count']==16
        if name=='FNO100_4800':
            cost=dict(isolated_4800_campaign_cost_exact=False,whole_stage_UTC_seconds=None,
                      reason='Intermediate checkpoint of previously completed12000 campaign; exact isolatedwholecost unknown')
        else:
            stage=qualified[name];native=stage['summary'];external=stage['external']
            cost=dict(isolated_4800_campaign_cost_exact=True,whole_stage_UTC_seconds=external['elapsed_epoch_seconds'],
                      whole_stage_monotonic_seconds=external['elapsed_monotonic_seconds'],
                      training_UTC_seconds=native['training_epoch_seconds'])
            if 'technical_recovery_cost' in native:
                recovery=native['technical_recovery_cost'];old=d.read(recovery['original_failure_binding']['external']['path'])
                cost.update(original_failed_attempt_outer_UTC_seconds=old['elapsed_epoch_seconds'],
                            recovery_attempt_outer_UTC_seconds=external['elapsed_epoch_seconds'],
                            both_active_attempts_UTC_seconds=old['elapsed_epoch_seconds']+external['elapsed_epoch_seconds'],
                            whole_stage_UTC_seconds=external['original_campaign_elapsed_epoch_seconds'],
                            whole_stage_monotonic_seconds=external['original_campaign_elapsed_monotonic_seconds'],
                            repair_gap_UTC_seconds=recovery['between_attempt_wall_epoch_seconds'],
                            repair_gap_monotonic_seconds=recovery['between_attempt_wall_monotonic_seconds'],
                            original_absolute_deadlines_retained=True)
        costs[name]=cost
        points.append(dict(name=name,label=labels[name],U_percent=100*metric['case_mean_u_relative_l2'],
                           S_percent=100*metric['case_mean_stress_relative_l2'],
                           hot_percent=100*metric['case_mean_stress_hotspot_relative_l2'],
                           joint_score=metric['selection_score'],inference_UTC_seconds=row['inference_epoch_seconds'],
                           parameters=row['model_profile']['parameter_count']))
    leader=min(p['joint_score'] for p in points);leaders=[p['name'] for p in points if p['joint_score']==leader]
    base=full['GLOBAL_HASH100_4800']['full60'];ablations=[]
    for name in ('GLOBAL_HASH_head_latent_zero','GLOBAL_HASH_head_geometry_zero'):
        row=summary['candidates'][name];assert not row['eligible_for_selection']
        metric=row['full60'];assert metric['all_cases_complete'] and metric['fixed_denominator']==60
        ablations.append(dict(name=name,mode=row['head_condition_ablation'],eligible_for_selection=False,
                              U_percent=100*metric['case_mean_u_relative_l2'],S_percent=100*metric['case_mean_stress_relative_l2'],
                              U_change_points=100*(metric['case_mean_u_relative_l2']-base['case_mean_u_relative_l2']),
                              S_change_points=100*(metric['case_mean_stress_relative_l2']-base['case_mean_stress_relative_l2'])))
    pairs={a+':'+b:paired(full[a]['full60'],full[b]['full60'],keys)
           for i,a in enumerate(full) for b in list(full)[i+1:]}
    output=d.path('results/geobc_operator_hash_v5');output.mkdir(parents=True,exist_ok=True)
    assert not (output/'manifest.json').exists(),'No silent CPU export replay'
    text=['# 全局几何／BC控制连续Hash：是否比现有方法更好','',
          '同100TRAIN CAD×三载荷、4800更新／16遍、seed2026、old40-only scaler；knownDEV20×三载荷全原节点，未读FINAL。','',
          '| 方法 | 位移误差 | 应力误差 | 60例推理秒 |','|---|---:|---:|---:|']
    for p in points:text.append(f"| {p['label']} | {p['U_percent']:.2f}% | {p['S_percent']:.2f}% | {p['inference_UTC_seconds']:.2f} |")
    text+=['',f"同4800联合评分最低：{', '.join(labels[n] for n in leaders)}。当前更充分曝光的FNO100／12000更新23.17%／50.85%仍单独作主候选参考；不与4800冒称相同曝光。",
           '', '全局Hash复用FNO完整几何／BC／材料空间latent，Hash配置沿用原参数。总容量和混合精度不同，不能称纯容量匹配消融；真实初始化／最大全节点梯度与资源均在训练预算内核验。',
           '', '两个推理消融只将解码head的16维latent或32维几何descriptor置零，编码器／算子及训练权重保持；geometry仍可能通过其余路径存在。它们是分布外依赖诊断，未重训、不参与求解器选型。']
    for a in ablations:text.append(f"{a['mode']}：位移／应力误差变化{a['U_change_points']:+.2f}／{a['S_change_points']:+.2f}个百分点。")
    text+=['', '成本JSON保留初始元数据故障、原失败126秒、恢复尝试和原campaign全部wall（含修复间隔），原绝对预算不重置。FNO4800为既有中间端点，隔离完整成本unknown=null，不冒用12000成本。',
           '', '240份有限预测先保存／重验，再knownGT60共享一次；原CSV／FINAL／训练更新／新scalerfit均0。物理U(mm)和signed S(MPa)保留；原节点精度与连续查询接口一致性不代表独立离节点FEM准确性。']
    (output/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(output/'metrics.json',dict(points=points,costs=costs,paired_CAD20=pairs,
              ablations=ablations,new_candidates=summary['candidates'],FNO_reference=reference['candidates']['FNO100_step4800']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for j,p in enumerate(points):ax.bar([i+(j-1)*.24 for i in range(2)],[p['U_percent'],p['S_percent']],width=.24,label=p['label'])
    ax.set_xticks([0,1],['Displacement','Signed stress']);ax.set_ylabel('Case mean relative L2 (%)')
    ax.set_title('Known DEV20 x 3: same100 TRAIN CAD / 4,800 updates');ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    d.write(output/'manifest.json',dict(status='completed',scientific_digest=plan['scientific_digest'],
              exporter_source_sha256=d.sha(OWN),summary_identity=d.identity(folder/'summary.json',hashed=True),
              external_identity=d.identity(folder/'external_terminal.json',hashed=True),points=points,costs=costs,
              ablations=ablations,eligible_same4800_leaders=leaders,actual_CPU_export_seconds=time.monotonic()-start,
              final_solver_selected=False,files=['BRIEF.md','metrics.json','comparison.png']))
    print(dict(status='completed',points=points,ablations=ablations,actual_CPU_export_seconds=time.monotonic()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
