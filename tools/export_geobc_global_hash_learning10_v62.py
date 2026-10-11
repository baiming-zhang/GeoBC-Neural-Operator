"""One immutable full-TRAIN scalar report; no arrays, PT or inference."""
import argparse
import csv
import time

OWN='tools/export_geobc_global_hash_learning10_v62.py'
PLAN='notes/geobc_global_hash_learning10_v62_plan.json'
REVIEW='notes/geobc_global_hash_learning10_v62_exporter_root_review.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash_learning10_contract_v62 import terminal,approved_contract,complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,_,_,_,up=approved_contract(PLAN)
    _,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['TRAIN_keys']
    assert marker['SCI']==p['scientific_digest'] and a['training_complete']
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==1500
    assert a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==3000
    assert a['case_exposure_counts']=={k:10 for k in keys}
    assert not a['possible_partial_optimizer_update'] and not a['optimizer_step_in_progress']
    q=a['actual_largest_fullN_gradient_check']
    assert q['passed'] and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored']
    assert a['typed_model_Adam_restore_verified'] and a['restored_model_Adam_fourRNG_cursor_exposure']
    assert a['parent_preupdate_fullN_replay']['passed'] and a['parent_checkpoint_stat_unchanged']
    assert a['parent_checkpoint_hash_attempts']==a['parent_checkpoint_hash_reads']==a['parent_checkpoint_load_attempts']==a['parent_checkpoint_loads']==1
    assert a['physical_TRAIN_total_array_reads']==300 and a['resident_TRAIN_physical_target_reuses']==301
    c=a['cache_receipt']
    assert c['physical_TRAIN_target_read_attempts']==c['physical_TRAIN_target_reads']==300
    assert c['encoded_TRAIN_target_read_attempts']==c['encoded_TRAIN_target_reads']==0
    assert c['input_array_read_attempts']==c['input_array_reads'] and c['source_CSV_H5_reads']==c['scaler_fits']==c['DEV_FINAL_target_reads']==0
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    assert max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    old=s['parent_TRAIN300'];new=a['endpoints']['3000']['full_TRAIN'];fno=s['same10_FNO_TRAIN300']
    rows=[('GlobalHash 5',old),('GlobalHash 10',new),('FNO 10',fno)]
    for _,metric in rows:complete_physical(metric,keys)
    fu='case_mean_u_relative_l2';fs='case_mean_stress_relative_l2';fh='case_mean_stress_hotspot_relative_l2'
    du=new[fu]-old[fu];ds=new[fs]-old[fs];dh=new[fh]-old[fh]
    passed=ds<=-.03 and du<=.02
    assert passed==a['TRAIN_gate']['passed'] and not a['DEV_evaluation_actual_released']
    out=d.path('results/geobc_global_hash_learning10_v62');assert not out.exists(),'One immutable report'
    out.mkdir(parents=True)
    decision='TRAIN门槛通过，可另行准备新CAD预测先提交的验证；尚未发生泛化读出' if passed else 'TRAIN门槛未通过，停止追加长Hash训练，不读取新DEV'
    text=['# 100CAD：条件Hash从5遍追加至10遍','',
        '**同100个TRAIN CAD×三载荷、全部原FEM节点、物理位移与signed应力；恢复原模型和Adam，固定追加1500次病例更新。**','',
        '| 方法／总曝光遍数 | TRAIN位移误差 | TRAIN应力误差 | TRAIN热点误差 |',
        '|---|---:|---:|---:|']
    for label,m in rows:text.append(f'| {label} | {100*m[fu]:.2f}% | {100*m[fs]:.2f}% | {100*m[fh]:.2f}% |')
    text+=['',f'Hash10减自身5：位移{100*du:+.2f}、应力{100*ds:+.2f}、热点{100*dh:+.2f}个百分点。{decision}。','',
        f'新增训练{a["training_epoch_seconds"]:.2f}秒，完整阶段{e["elapsed_epoch_seconds"]:.2f}秒；Hash两段累计训练{a["trajectory_training_epoch_seconds"]/60:.2f}分钟。原5遍技术失败、恢复和阶段间等待成本单独保留，没有用累计纯训练时间抹掉失败成本。','',
        '这是全局几何／边界条件控制的连续Hash场，与FNO相比容量和精度不同；同10遍曝光不等于等参数或等成本，TRAIN排名不是最终求解器或新CAD泛化结论。少量共享3场可低于1%，不能推断100CAD同样已收敛。','',
        '实际恢复model、Adam、四类RNG及曝光cursor；最大节点案例完整梯度通过且资格检查不改变状态。更新前同种子原案例物理指标重算一致。旧检查点仅有原stat记录：本次首次观察的文件SHA不能冒充原实验保存时的SHA。','',
        'TRAIN物理标签300份各读一次，常驻目标复用301次含恢复见证；冗余编码标签在读取前移除。原5遍缓存实际读过未使用编码标签，历史记录不改写。新DEV／FINAL／原CSV／scaler拟合均0，旧5遍负结果保留，不自动20、40遍或24000步。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['method','case','nodes','U_relative_L2','S_relative_L2','hotspot_relative_L2'])
        for label,m in rows:
            for key in keys:
                z=m['per_case'][key];writer.writerow([label,key,z['nodes'],z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    d.write(out/'metrics.json',dict(rows={k:v for k,v in rows},TRAIN_gate=a['TRAIN_gate'],actual_native=a,actual_external=e,terminal_binding=b,
        parent_binding=up['parent_binding'],FNO_binding=up['FNO_binding'],shared3_binding=up['shared_binding'],DEV_readout_completed=False,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for field,label in [(fu,'Displacement'),(fs,'Signed stress'),(fh,'Stress hotspot')]:
        axes[0].plot([5,10],[100*old[field],100*new[field]],marker='o',label=label)
    axes[0].set_xticks([5,10]);axes[0].set_xlabel('GlobalHash complete case exposures');axes[0].set_ylabel('TRAIN mean relative L2 (%)');axes[0].legend();axes[0].grid(alpha=.2)
    for index,field in enumerate([fu,fs,fh]):
        axes[1].bar(index-.18,100*new[field],width=.36,color='#3274a1',label='GlobalHash 10' if index==0 else None)
        axes[1].bar(index+.18,100*fno[field],width=.36,color='#e1812c',label='FNO 10' if index==0 else None)
    axes[1].set_xticks([0,1,2],['U','S','Hotspot']);axes[1].set_ylabel('TRAIN mean relative L2 (%)');axes[1].legend()
    fig.suptitle('Full-node physical TRAIN100 x 3 / capacity and precision differ');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','comparison.png']
    assert all((out/k).is_file()for k in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),actual_CPU_seconds=time.perf_counter()-began,
        terminal_binding=b,TRAIN_gate_passed=passed,files=files,GPU_calls=0,new_GT_array_reads=0,PT_body_reads=0))
    print(dict(status='completed',TRAIN_gate_passed=passed,CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
