"""CPU scalar comparison at matched100 CAD/12000 exposure; default inert."""
import argparse
import time

OWN='tools/export_geobc_global_hash_matched_v8.py'
PLAN='notes/geobc_operator_hash_matched_readout_v8_plan.json'
REVIEW='notes/geobc_global_hash_matched_v8_exporter_review.json'


def run():
    entry=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.operator_hash_matched_readout_contract_v8 import approved_contract,complete_physical,check_identity
    from tools.export_geobc_exposure_comparison import paired
    marker=d.read(REVIEW);assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,_,_,_,qualified=approved_contract(PLAN);assert p['scientific_digest']==marker['scientific_digest']
    folder=d.path(p['output']);s=d.read(folder/'summary.json');e=d.read(folder/'external_terminal.json')
    assert s['status']=='completed' and s['protocol_qualified'] and s['closed'] and not s['active']
    assert e['exit_code']==0 and e['within_external_budget'] and e['closed'] and not e['active']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert s['committed_finite_predictions']==s['owned_GT_array_reads']==60
    assert s['raw_GT_reads']==s['FINAL_numeric_reads']==s['training_updates']==s['scaler_fits']==0
    check_identity(marker['FNO_reference_summary_identity'])
    refp=d.read('notes/geobc_fno_coverage_readout_v4_plan.json');refs=refp['scientific']
    assert all(refs[k]==p['scientific'][k] for k in ('case_keys','scaler','GT_manifest_identity','dataset_identity'))
    ref=d.read(marker['FNO_reference_summary_identity']['path']);assert ref['status']=='completed' and ref['protocol_qualified']
    rows={'FNO100_12000':ref['candidates']['FNO100_step12000'],'GLOBAL_HASH100_12000':s['candidates']['GLOBAL_HASH100_12000']}
    keys=p['scientific']['case_keys'];points=[]
    for name,row in rows.items():
        complete_physical(row['full60'],keys)
        assert row['step']==12000 and row['case_exposure_count']==40 and row['train_CAD_count']==100
        m=row['full60'];points.append(dict(name=name,U_percent=100*m['case_mean_u_relative_l2'],S_percent=100*m['case_mean_stress_relative_l2'],hot_percent=100*m['case_mean_stress_hotspot_relative_l2'],joint_score=m['selection_score'],parameters=row['model_profile']['parameter_count'],inference_UTC_seconds=row['inference_epoch_seconds']))
    stage=qualified['GLOBAL_HASH100_12000'];train=stage['plan'];native=stage['summary'];outer=stage['external']
    parent=d.read(train['scientific']['warm_parent_training_plan_identity']['path']);parent_outer=d.read(d.path(parent['output'])/'external_terminal.json');parent_native=d.read(d.path(parent['output'])/'summary.json')
    cost=dict(global_parent_campaign_UTC_seconds=parent_outer['original_campaign_elapsed_epoch_seconds'],
              global_additional_stage_UTC_seconds=outer['elapsed_epoch_seconds'],global_sum_stage_wall_UTC_seconds=parent_outer['original_campaign_elapsed_epoch_seconds']+outer['elapsed_epoch_seconds'],
              global_train_UTC_seconds=parent_native['training_epoch_seconds']+native['training_epoch_seconds'],
              global_original_failed_and_recovery_cost=parent_native['technical_recovery_cost'],
              DEV_readout_UTC_seconds=e['elapsed_epoch_seconds'],between_successful_stage_analysis_wall_not_in_stage_sum=True,
              FNO_train_UTC_seconds=rows['FNO100_12000']['terminal_campaign_training_epoch_seconds'])
    fno_train_plan=d.read(rows['FNO100_12000']['training_plan_identity']['path']);cost['FNO_whole_stage_UTC_seconds']=d.read(d.path(fno_train_plan['output'])/'external_terminal.json')['elapsed_epoch_seconds']
    delta={name:points[1][name]-points[0][name] for name in ('U_percent','S_percent','hot_percent')}
    pairs=paired(rows['GLOBAL_HASH100_12000']['full60'],rows['FNO100_12000']['full60'],keys)
    best=min(x['joint_score'] for x in points);leaders=[x['name'] for x in points if x['joint_score']==best]
    output=d.path('results/geobc_global_hash_matched_v8');output.mkdir(parents=True,exist_ok=True)
    assert not (output/'manifest.json').exists(),'No silent CPU export replay'
    lines=['# 同100CAD、40遍：全局Hash与FNO','',
           'same100 TRAIN CAD×三载荷／12000更新／seed2026／oldTRAIN40-only scaler；knownDEV20×三载荷全原节点物理误差。FINAL数值未读。','',
           '| 方法 | 位移误差 | 应力误差 | 热点误差 |','|---|---:|---:|---:|---:|']
    for x in points:lines.append(f"| {x['name']} | {x['U_percent']:.2f}% | {x['S_percent']:.2f}% | {x['hot_percent']:.2f}% |")
    lines+=['',f"全局Hash相对FNO位移／应力／热点变化：{delta['U_percent']:+.2f}／{delta['S_percent']:+.2f}／{delta['hot_percent']:+.2f}个百分点，负值更好。联合评分最低：{', '.join(leaders)}。",
            '', '全局Hash从自身4800恢复模型／Adam／四RNG／曝光cursor，仅追加7200，原元数据失败和修复成本保留；容量12.25M vs4.53M、FP16 TCNN vsFP32 MLP，非纯容量或精度匹配的架构因果比较。',
            '', '这轮仅追加一个有竞争力候选，未进行Hash小参数调整。此前4800端点headlatentzero与geomzero仅OOD依赖诊断，不重复训练／推理或用于排名。',
            '', '60份新完整finite预测先保存／重验，再已知GT60共享一次；FNO指标从已有提交的同60评分标量读取，不重跑FNO。knownDEV单seed／可用性条件池不等于最终盲测；实际CUDA连续查询一致性不证明独立离节点FEM准确性。']
    (output/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(output/'metrics.json',dict(points=points,cost=cost,delta_global_minus_FNO_percentage_points=delta,paired_CAD20=pairs,rows=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.8))
    for j,x in enumerate(points):ax.bar([i+(j-.5)*.3 for i in range(3)],[x['U_percent'],x['S_percent'],x['hot_percent']],width=.3,label=x['name'])
    ax.set_xticks([0,1,2],['Displacement','Signed stress','Stress hotspot']);ax.set_ylabel('Case mean relative L2 (%)');ax.legend(fontsize=8);ax.set_title('Known DEV20 x 3 / same100 TRAIN CAD / 12,000 updates');fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    d.write(output/'manifest.json',dict(status='completed',scientific_digest=p['scientific_digest'],exporter_source_sha256=d.sha(OWN),actual_CPU_export_seconds=time.monotonic()-entry,points=points,cost=cost,eligible_joint_leaders=leaders,delta_global_minus_FNO_percentage_points=delta,summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True),final_solver_selected=False))
    print(dict(status='completed',points=points,delta=delta,CPU_seconds=time.monotonic()-entry))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
