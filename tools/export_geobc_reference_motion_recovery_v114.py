"""Scalar-only reference pose, reciprocity and energy evidence; no field reads."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_reference_motion_recovery_v114_plan.json')
    r=d.read('notes/geobc_reference_motion_recovery_v114_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest']
    assert r['source_sha256']==d.sha('tools/export_geobc_reference_motion_recovery_v114.py') and r['actual_export']==0
    assert a['prediction_barrier_complete'] and a['committed_pose_requests']==12 and a['committed_pose_components']==72 and a['committed_energy_values']==12
    assert a['input_cache_reads']==2 and a['basis_array_reads']==12 and a['independent_mixed_array_reads']==6 and a['actual_force6_gradient_points']==4
    assert a['GT_reads']==a['K_PCG_replays']==a['raw_H5_deck_GRID_reads']==a['neural_PT_reads']==a['training_updates']==0
    o=d.path('results/geobc_reference_motion_recovery_v114');o.mkdir(exist_ok=False);rows=[]
    lines=['# RBE参考点六维运动与能量接口','',f'固定门：{"PASS"if a["reference_motion_gate_passed"]else"FAIL"}；整阶段UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。',
        '', '`ReferenceMotionResponse.query_force6(force6)`返回前三分量平移mm、后三分量转角rad与0.5 f·pose能量Nmm。参考点使用真实RBE3加权刚体拟合Hbar定义，不通过实体点定位器投影。原始柔度矩阵C未对称化、投影或修正。',
        '', '| CAD | 原节点 | 原始互易性相对差 | 能量最小特征值（mm/N） | 能量条件数 | 独立混合运动最大相对差 | 缓存／构造／查询秒 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        rec=a['case_records'][cad];m=rec['constructor_metadata'];mix=rec['independent_mixed_metrics']
        maximum=max(v['scaled_pose_mm']['relative']for v in mix.values())
        lines.append(f'| {cad} | {rec["actual_fullN"]} | {m["reciprocity_relative"]:.3g} | {m["energy_normalized_minimum_eigenvalue"]:.3g} | {m["energy_normalized_condition"]} | {maximum:.3g} | {rec["predictor_cache_seconds"]:.3f}／{rec["API_constructor_seconds"]:.3f}／{rec["query_chunk_properties_seconds"]:.4f} |')
        checks={k:rec[k]for k in ('chunk','homogeneity','superposition','canonical_derived_translation_witness')}
        for name,v in mix.items():
            for component in ('scaled_pose_mm','translation_mm','rotation_rad'):checks['mix'+name+'_'+component]=v[component]
        for j,v in enumerate(rec['gradient_checks']):
            checks['gradient'+str(j)+'_Jacobian']=v['Jacobian_vs_rawC'];checks['gradient'+str(j)+'_energy']=v['energy_gradient_vs_motion']
        for name,v in checks.items():rows.append(dict(CAD=cad,check=name,relative=v['relative'],maximum_absolute=v['maximum_absolute'],passed=v['passed']))
    lines+=['',f'原技术失败UTC {e["original_guard"]["elapsed_epoch_seconds"]:.6f}秒；恢复UTC {e["elapsed_epoch_seconds"]:.6f}秒；原campaign总UTC {e["whole_original_campaign_epoch_seconds"]:.6f}秒，原截止未延长。', '源reference GRID没有H5输出行，derived_node_ids为空，原规范输出节点平移见证不可得，明确null／FAIL；完整预声明总门因此FAIL，不造节点或放宽门。其余运动／能量性质独立列出。', '完整2CAD／12运动请求／72分量／12能量先有限值、hash、body、Cartesian屏障，再读取6个原独立零初值混合场各一次。原规范参考点输出平移见证只用已驻留六基解，不读取GT。4个实际六力矩自动微分点全部保留；Jacobian直接与原始C比较，能量梯度直接与实际运动比较。',
        '运动综合误差使用[t_mm,L_fit·theta_rad]同单位坐标；平移mm与转角rad误差另列，避免混单位L2。互易性和能量特征值使用D C D（D=diag(1,1,1,L_fit,L_fit,L_fit)）；对称部分只做正定诊断，未替换预测矩阵。',
        '**这是已验收近似P2方程内的参考运动／能量一致性，不是新取得的原OptiStruct旋转或新载荷GT精度。**近似curvedP2／四Q／weightedLS RBE3与未知DEBUG／PSOLID控制局限保留。两样本已揭盲；不证明任意支撑、任意载荷位置、C1、形状导数或最终求解器。signedS仍未解决。',
        '本阶段只加载Hbar／member IDs／拟合长度／derived IDs与六基解。缓存读取、构造、查询、梯度、保存、独立参考与退出全部计入双时轴。原新几何源→六基解77.94036／63.43762秒仅继承真实成本，未知完整工程冷启动null；缓存接口延迟不当新几何全冷部署。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'checks.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,new_OptiStruct_reference_rotation_GT=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));cads=p['scientific']['selection']['CADs']
    vals=[a['case_records'][cad]['constructor_metadata']['reciprocity_relative']for cad in cads]
    bars=axes[0].bar(cads,vals);axes[0].bar_label(bars,fmt='%.2e');axes[0].axhline(1e-5,color='r',linestyle='--',label='fixed gate');axes[0].set_yscale('log');axes[0].set_ylabel('Raw normalized reciprocity defect');axes[0].legend()
    vals=[a['case_records'][cad]['query_chunk_properties_seconds']for cad in cads]
    bars=axes[1].bar(cads,vals);axes[1].bar_label(bars,fmt='%.4fs');axes[1].set_ylabel('12 motion/energy queries + properties (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Reference motion / energy '+('PASS'if a['reference_motion_gate_passed']else'FAIL')+'; original rotation GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        reference_motion_gate_passed=a['reference_motion_gate_passed'],new_GT_PT_arrays_GPU=0,CSV_rows=len(rows),PNG_visual_review_pending=True,
        files=['BRIEF.md','checks.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=a['reference_motion_gate_passed'],CSV_rows=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
