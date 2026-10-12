"""Scalar-only exact motion boundary evidence; no scientific bodies or APIs."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_reference_prescribed_motion_v116_plan.json');r=d.read('notes/geobc_reference_prescribed_motion_v116_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['actual_export']==0
    assert r['source_sha256']==d.sha('tools/export_geobc_reference_prescribed_v116.py')
    assert a['GT_reads']==a['raw_H5_deck_GRID_reads']==a['geometry_K_coarse_rebuilds']==a['neural_PT_reads']==a['training_updates']==0
    if a['prescribed_motion_gate_passed']:
        assert a['prediction_barrier_complete']and a['committed_finite_endpoints']==6 and a['input_cache_reads']==2 and a['basis_array_reads']==12
        assert a['operator_array_reads']==6 and a['coarse_array_reads']==20 and a['owned_array_reads']==6
        assert all(v['passed']and v['physics_basis_reads_before_independent_solve']==0 for v in a['case_records'].values())
    o=d.path('results/geobc_reference_prescribed_motion_v116');o.mkdir(exist_ok=False);rows=[]
    lines=['# 精确参考点运动边界：独立约束零空间求解','',f'固定两CAD／六解门：{"PASS"if a["prescribed_motion_gate_passed"]else"FAIL"}；整阶段UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。',
        '', '真实Hbar约束规定加载RBE参考点的平移／转角，原其他零支撑／材料／几何保持。独立求解使用几何抬升u0=Rᵀ solve(RRᵀ,target)与约束零空间Π，不读取物理基解作为抬升或warmstart；不是大弹簧近似。',
        '', '| CAD | 原节点 | 三路独立零空间PCG秒 | 紧凑更新＋全N输出秒 | 最大U相对差 | 最大约束相对差 |',
        '|---|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        rec=a['case_records'].get(cad,{});metrics=rec.get('compact_vs_independent_constraint',{})
        if metrics:
            umax=max(v['fullN_U']['relative']for v in metrics.values());cmax=max(v['prescribed_constraint']['relative']for v in metrics.values())
            lines.append(f'| {cad} | {rec["actual_allN"]} | {rec["independent_group_seconds"]:.3f} | {rec["compact_update_and_fullfield_seconds"]:.5f} | {umax:.3g} | {cmax:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — |')
        for j in range(3):
            record=a.get('endpoint_records',{}).get(cad+'/prescribed'+str(j),{});v=metrics.get(str(j),{})
            rows.append(dict(CAD=cad,load='prescribed'+str(j),all_original_nodes=record.get('all_original_nodes'),initial_nullspace_guess=record.get('initial_nullspace_guess'),
                geometry_only_lift=record.get('geometry_only_lift'),numerical_state=record.get('numerical_state'),projected_true_relative_residual=record.get('true_projected_relative_residual'),
                projected_RHS_norm=record.get('projected_RHS_norm'),constraint_relative=record.get('reference_constraint_relative'),
                projected_applied_reference_force_relative=record.get('projected_applied_reference_force_relative'),fullN_U_relative=v.get('fullN_U',{}).get('relative'),
                normalized_reference_reaction_relative=v.get('normalized_reference_reaction',{}).get('relative'),body_energy_relative=v.get('body_energy',{}).get('relative'),
                translation_mm_relative=v.get('translation_mm',{}).get('relative'),rotation_rad_relative=v.get('rotation_rad',{}).get('relative'),
                reference_reaction_force6=record.get('reference_reaction_force6'),body_strain_energy_Nmm=record.get('body_strain_energy_Nmm'),
                net_force_with_constraint_fixed_reaction_N=record.get('net_force_including_constraint_and_fixed_reaction_N'),net_moment_with_constraint_fixed_reaction_Nmm=record.get('net_moment_including_constraint_and_fixed_reaction_Nmm')))
    lines+=['', 'Π=I−Rᵀ(RRᵀ)⁻¹R、v初值0，独立方程ΠKΠv=Πb−ΠKu0。原固定32步预条件器按ΠMΠ应用；完整投影空间有零方向，SPD只声称在约束零空间。每路alpha／beta／残差停止独立，真残差用实际非零投影RHS归一化，绝不用接近零的Πb归一化。每25步原子保存z／r／p／lift／rhs与真实成本。',
        '六个完整原节点字段、约束、真平衡残差及独立反力先finite、hash、body、Cartesian屏障；之后才读12个原六基解body各一次，构造未改原始柔度的紧凑响应。抬升与独立PCG的基解body读取0。',
        '同一reference全部六运动被指定时，施加于该reference的外载影响约束反力，实体位移由目标运动决定；Πb近零实际见证另列，没有伪构一般实体bodyload。独立lambda=solve(RRᵀ,R(Ku−b))，物理反力Dlambda，平衡含reference约束及原固定支撑反力。',
        '紧凑totalforce= D solve(D rawC D,Dtarget)，reaction=totalforce−appliedforce。能量0.5 totalforce·target必须包含约束反力功，不用appliedforce单独冒总能量。平移mm、转角rad与反力N／Nmm按实际拟合长度规范，分量分别列出。',
        '**验证的是声明的新精确运动边界近似方程，不是新取得的原OptiStruct支撑／转角／离节点GT。**近似curvedP2四Q／weighted-LS RBE3与未知DEBUG／PSOLID控制保留；reference GRID无原H5输出行，物理定义来自Hbar，不生成替代节点。两样本已揭盲，signedS仍未提供。',
        '原K／diag／粗矩阵不重建，旧8fine／粗能量guards不重播；实际新投影K／M与标量及geometry-only抬升资格单列。缓存加载、资格、PCG、保存、屏障、基解读取／API、对照、失败与退出全部计入双时轴；未知工程全冷null。继承源到六基解77.94036／63.43762秒不是本阶段新成本，缓存紧凑响应与独立PCG也不是NN同预算排名。']
    if not a['prescribed_motion_gate_passed']:lines+=['',f'完整2CAD／6终态分母保留，未验收项null；固定终态：{a.get("reason",a.get("scientific_outcome"))}。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'endpoints.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,new_original_support_GT_accuracy=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));cads=p['scientific']['selection']['CADs']
    vals=[a['case_records'].get(cad,{}).get('independent_group_seconds')for cad in cads]
    compact=[a['case_records'].get(cad,{}).get('compact_update_and_fullfield_seconds')for cad in cads]
    if all(v is not None for v in vals):
        bars=axes[0].bar(cads,vals);axes[0].bar_label(bars,fmt='%.2fs')
    if all(v is not None for v in compact):
        bars=axes[1].bar(cads,compact);axes[1].bar_label(bars,fmt='%.4fs')
    axes[0].set_ylabel('Geometry lift + independent nullspace PCG (s)');axes[1].set_ylabel('Existing basis exact-motion update + fullN U (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Exact reference motion '+('PASS'if a['prescribed_motion_gate_passed']else'FAIL')+'; original new-support GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        prescribed_motion_gate_passed=a['prescribed_motion_gate_passed'],CSV_rows=len(rows),PNG_visual_review_pending=True,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','endpoints.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=a['prescribed_motion_gate_passed'],CSV_rows=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
