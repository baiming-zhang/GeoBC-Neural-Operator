"""One complete TRAIN learning curve and all three segment costs, scalar only."""
import argparse
import time

OWN='tools/export_geobc_fno_learning20_v56.py'
REVIEW='notes/geobc_fno_learning20_v56_exporter_root_review.json'
PLAN='notes/geobc_fno_learning20_v56_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.fno_learning20_contract_v56 import terminal,metadata_inputs,check_identity
    from geobc.global_hash_physical_readout_contract_v18 import complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);up=metadata_inputs();s=p['scientific']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==3000
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==6000
    assert a['case_exposure_counts']=={k:20 for k in s['TRAIN_keys']}
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    q=a['actual_largest_fullN_gradient_check']
    assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored']
    assert a['strict_loaded_checkpoint_tensor_state_verified'] and a['parent_checkpoint_loads']==a['parent_checkpoint_hash_reads']==1
    assert a['physical_TRAIN_total_array_reads']==a['resident_TRAIN_physical_target_reuses']==300
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['cached_TRAIN_physical_reads']==0
    original=d.read(check_identity(up['original5_binding']['summary']))
    original_external=d.read(check_identity(up['original5_binding']['external']))
    segments=[(5,original,original_external),(10,up['parent_summary'],up['parent_external']),(20,a,e)]
    points=[];costs={k:0. for k in ['training_epoch','training_monotonic','stage_epoch','stage_monotonic']}
    for passes,native,outer in segments:
        step=passes*300;metric=native['endpoints'][str(step)]['full_TRAIN'];complete_physical(metric,s['TRAIN_keys'])
        costs['training_epoch']+=native['training_epoch_seconds'];costs['training_monotonic']+=native['training_monotonic_seconds']
        costs['stage_epoch']+=outer['elapsed_epoch_seconds'];costs['stage_monotonic']+=outer['elapsed_monotonic_seconds']
        points.append(dict(passes=passes,step=step,full_TRAIN=metric,cumulative_cost=dict(costs),
            segment_training_epoch_seconds=native['training_epoch_seconds'],segment_training_monotonic_seconds=native['training_monotonic_seconds'],
            segment_stage_epoch_seconds=outer['elapsed_epoch_seconds'],segment_stage_monotonic_seconds=outer['elapsed_monotonic_seconds']))
    # The v56 native trajectory fields cover its immediate parent segment
    # plus this segment. Rebuild all0-to20 costs from all three actual receipts.
    assert abs(a['trajectory_training_epoch_seconds']-(up['parent_summary']['training_epoch_seconds']+a['training_epoch_seconds']))<1e-6
    base=points[1]['full_TRAIN'];new=points[2]['full_TRAIN'];fu='case_mean_u_relative_l2';fs='case_mean_stress_relative_l2';fh='case_mean_stress_hotspot_relative_l2'
    du=new[fu]-base[fu];ds=new[fs]-base[fs];dh=new[fh]-base[fh];dp=.5*(du+ds)
    passed=dp<=-.03 and max(du,ds)<=.02 and dh<=.02
    assert passed==a['TRAIN_gate']['passed']
    out=d.path('results/geobc_fno_learning20_v56');assert not out.exists(),'One immutable report'
    out.mkdir(parents=True)
    text=['# FNO：有证据支持的10→20遍趋势检查','',
        '**同100个TRAIN CAD×三载荷、同模型和训练轨迹，完整原节点physical U(mm)/signed S(MPa)。5→10时TRAIN和knownDEV都改善，才单独释放本次固定20端点；没有自动40遍或24000步。**','',
        '| 总曝光遍数 | TRAIN位移误差 | TRAIN应力误差 | TRAIN热点误差 |','|---|---:|---:|---:|']
    for point in points:
        m=point['full_TRAIN'];text.append(f'| {point["passes"]} | {100*m[fu]:.2f}% | {100*m[fs]:.2f}% | {100*m[fh]:.2f}% |')
    decision='TRAIN推进门槛通过，值得单独knownDEV60读出；尚未执行，不代表泛化收益' if passed else 'TRAIN推进门槛未通过，不自动读取DEV或延长训练'
    text+=['',f'20减10：位移{100*du:+.2f}、应力{100*ds:+.2f}、热点{100*dh:+.2f}个百分点。{decision}。','',
        f'本次新增3000次完整病例更新、实际训练{a["training_epoch_seconds"]:.2f}秒、完整阶段{e["elapsed_epoch_seconds"]:.2f}秒。三段纯训练累计{costs["training_epoch"]/60:.2f}分钟，完整训练阶段累计{costs["stage_epoch"]/60:.2f}分钟；原始epoch／monotonic分别记录，阶段间分析等待排除。','',
        '累计成本从真实5、10、20三份native/external收据合并；v56原生trajectory字段只覆盖最近两段，未将该字段误称完整0→20成本。所有旧成本保留。','',
        '恢复自身10遍model、Adam、四类RNG、cursor和曝光量；原scaler、病例顺序、FP32、模型、物理联合目标与Adam配方不变。实际最大全节点资格通过，model/Adam不变且RNG恢复。','',
        'TRAIN物理标签300份只加载一次、端点复用常驻标签；新DEV／FINAL／原CSV读取及scaler拟合均0。当前只是增加曝光量的TRAIN趋势，不能作为等预算架构比较、收敛证明或最终求解器结论。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,complete0to20_cost=costs,TRAIN_gate=a['TRAIN_gate'],
        actual_native=a,actual_external=e,terminal_binding=b,original5_binding=up['original5_binding'],parent10_binding=up['parent_binding'],
        raw_native_trajectory_scope='Only5to10+10to20; full0to20 cost reconstructed above',DEV_readout_completed=False,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    for field,label in [(fu,'Displacement'),(fs,'Signed stress'),(fh,'Stress hotspot')]:
        ax.plot([q['passes']for q in points],[100*q['full_TRAIN'][field]for q in points],marker='o',label=label)
    ax.set_xticks([5,10,20]);ax.set_xlabel('Complete case exposures');ax.set_ylabel('TRAIN mean relative L2 (%)')
    ax.set_title('Same joint physical FNO / TRAIN100 x 3');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/k).is_file()for k in ['BRIEF.md','metrics.json','comparison.png'])
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),actual_CPU_seconds=time.perf_counter()-began,
        terminal_binding=b,TRAIN_gate_passed=passed,files=['BRIEF.md','metrics.json','comparison.png'],GPU_calls=0,new_GT_array_reads=0,PT_body_reads=0))
    print(dict(status='completed',TRAIN_gate_passed=passed,CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
