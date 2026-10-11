"""One scalar-only exposure-matched TRAIN300 recipe report, no arrays/PT."""
import argparse
import time

OWN='tools/export_geobc_fno_case_batch_v55.py'
REVIEW='notes/geobc_fno_case_batch_v55_exporter_root_review.json'
PLAN='notes/geobc_fno_case_batch_v55_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.fno_case_batch_contract_v55 import terminal,metadata_inputs
    from geobc.global_hash_physical_readout_contract_v18 import complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific'];up=metadata_inputs()
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==375
    assert a['additional_case_exposure_completions']==a['case_backward_attempts']==a['case_backward_completions']==1500
    assert a['cursor']==3000 and a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==1875
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    assert a['case_exposure_counts']=={k:10 for k in s['TRAIN_keys']}
    q=a['actual_largest_fullN_gradient_check']
    assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored']
    assert a['qualification_case_backward_attempts']==a['qualification_case_backward_completions']==4
    assert a['saved_Adam_steps_equal_optimizer_step'] and not a['saved_Adam_steps_equal_exposure_cursor']
    assert a['parent_checkpoint_loads']==a['parent_checkpoint_hash_reads']==1
    assert a['physical_TRAIN_total_array_reads']==a['resident_TRAIN_physical_target_reuses']==300
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['cached_TRAIN_physical_reads']==0
    base=up['control_summary']['endpoints']['3000']['full_TRAIN'];new=a['endpoints']['3000']['full_TRAIN']
    for metric in [base,new]:complete_physical(metric,s['TRAIN_keys'])
    fu='case_mean_u_relative_l2';fs='case_mean_stress_relative_l2';fh='case_mean_stress_hotspot_relative_l2'
    du=new[fu]-base[fu];ds=new[fs]-base[fs];dh=new[fh]-base[fh];dp=.5*(du+ds)
    passed=dp<=-.03 and max(du,ds)<=.02 and dh<=.02
    assert passed==a['TRAIN_gate']['passed']
    out=d.path('results/geobc_fno_case_batch_v55');assert not out.exists(),'One immutable report'
    out.mkdir(parents=True)
    decision='值得进入单独的knownDEV60读出，仍需另行释放' if passed else '未达到推进门槛，不延长这套配方或自动读取DEV'
    text=['# 四病例平均梯度是否改善共享拟合？','',
        '**100个TRAIN CAD×三载荷、全原节点physical U(mm)/signed S(MPa)。同一个5遍父检查点、同病例顺序与累计10遍曝光，改变每次Adam更新合并的病例数。**','',
        '| 每次更新病例数 | TRAIN位移误差 | TRAIN应力误差 | TRAIN热点误差 |','|---|---:|---:|---:|']
    for label,m in [('1（既有对照）',base),('4（本次）',new)]:
        text.append(f'| {label} | {100*m[fu]:.2f}% | {100*m[fs]:.2f}% | {100*m[fh]:.2f}% |')
    text+=['',f'4病例减1病例：位移{100*du:+.2f}、应力{100*ds:+.2f}、热点{100*dh:+.2f}个百分点；联合U/S变化{100*dp:+.2f}个百分点。{decision}。','',
        f'本次新增1500个完整病例曝光、375次Adam更新；既有对照同曝光新增1500次更新。因此检查点曝光cursor=3000、真实optimizer step=1875。两种计数没有混同。实际新训练{a["training_epoch_seconds"]:.2f}秒，完整阶段{e["elapsed_epoch_seconds"]:.2f}秒；原始epoch／monotonic及父训练成本独立存档。','',
        '模型、scaler、原物理联合损失、Adam配方及恢复的四类RNG相同。各病例用自身完整U/S范数，四份梯度等权平均，GPU每次只保留一个病例计算图，不采样原节点。资格包含真实最大全节点病例，model/Adam不变且RNG恢复。','',
        'TRAIN物理标签300份只加载一次，端点复用常驻标签；新DEV／FINAL／原CSV读取及scaler拟合均0。既有对照只复用完整标量，未重跑训练或推理。','',
        '病例分组与更新次数共同变化，当前短配方结果不能单因果证明或排除梯度冲突，也不是架构上限或最终求解器结论。完整300例、最差、曝光／更新／资格／失败成本细节见metrics.json。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(control_TRAIN300=base,case_batch4_TRAIN300=new,TRAIN_gate=a['TRAIN_gate'],
        case_batch4_cost=a,control_terminal_binding=up['control_binding'],case_batch4_terminal_binding=b,
        scope='Same-case-exposure TRAIN-only shared recipe diagnostic; grouping and Adam update count both change',
        DEV_readout_completed=False,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.3,3.6))
    for i,(label,m)in enumerate([('1 case / update',base),('4 cases / update',new)]):
        bars=ax.bar([j+(i-.5)*.32 for j in range(3)],[100*m[k]for k in [fu,fs,fh]],width=.32,label=label)
        ax.bar_label(bars,fmt='%.2f',fontsize=9)
    ax.set_xticks([0,1,2],['Displacement','Signed stress','Stress hotspot']);ax.set_ylabel('Mean relative L2 (%)')
    ax.set_title('TRAIN100 x 3 / same parent, same 10 exposures');ax.legend()
    ax.set_ylim(0,max(100*m[k]for m in [base,new]for k in [fu,fs,fh])*1.23)
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
