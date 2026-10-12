"""Scalar-only evidence for fixed spring BC adaptation; independent equations."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_reference_spring_BC_v115_plan.json');r=d.read('notes/geobc_reference_spring_BC_v115_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['actual_export']==0
    assert r['source_sha256']==d.sha('tools/export_geobc_reference_spring_v115.py')
    assert a['GT_reads']==a['raw_H5_deck_GRID_reads']==a['geometry_K_coarse_rebuilds']==a['neural_PT_reads']==a['training_updates']==0
    if a['spring_BC_gate_passed']:
        assert a['prediction_barrier_complete']and a['committed_finite_endpoints']==6 and a['input_cache_reads']==2 and a['basis_array_reads']==12
        assert a['operator_array_reads']==6 and a['coarse_array_reads']==20 and a['owned_array_reads']==6
        assert all(v['passed']for v in a['case_records'].values())
    o=d.path('results/geobc_reference_spring_BC_v115');o.mkdir(exist_ok=False);rows=[]
    lines=['# 六维参考点弹簧边界与紧凑响应更新','',f'固定两CAD／六解门：{"PASS"if a["spring_BC_gate_passed"]else"FAIL"}。实际整阶段UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。',
        '', '在原加载RBE参考点增加真实六维grounded弹簧，原实体几何／材料／零支撑保持。新增K项是Hbarᵀ Sn Hbar的物理边界刚度，与数值Kshift不同；原fineK和固定32步粗预条件器不重建或改动。',
        '', '| CAD | 原节点 | 独立修改方程三解秒 | 紧凑更新＋全节点U秒 | 最大U相对差 | 最大缩放运动差 |',
        '|---|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        rec=a['case_records'].get(cad,{});metrics=rec.get('compact_vs_independent_modified_equation',{})
        if metrics:
            umax=max(v['fullN_U']['relative']for v in metrics.values());pmax=max(v['scaled_reference_pose']['relative']for v in metrics.values())
            lines.append(f'| {cad} | {rec["actual_allN"]} | {rec["independent_group_seconds"]:.3f} | {rec["compact_update_and_fullfield_seconds"]:.5f} | {umax:.3g} | {pmax:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — |')
        for j in range(3):
            key=cad+'/spring_mix'+str(j);record=a.get('endpoint_records',{}).get(key,{});v=metrics.get(str(j),{});balance=rec.get('equilibrium_including_grounded_spring',{}).get(str(j),{})
            rows.append(dict(CAD=cad,load='spring_mix'+str(j),all_original_nodes=record.get('all_original_nodes'),initial_guess=record.get('initial_guess'),
                numerical_state=record.get('numerical_state'),true_relative_residual=record.get('true_relative_residual'),
                fullN_U_relative=v.get('fullN_U',{}).get('relative'),scaled_pose_relative=v.get('scaled_reference_pose',{}).get('relative'),
                translation_mm_relative=v.get('translation_mm',{}).get('relative'),rotation_rad_relative=v.get('rotation_rad',{}).get('relative'),
                energy_identity_relative=v.get('body_plus_spring_energy_vs_external_work',{}).get('relative'),
                compact_total_energy_relative=v.get('compact_total_energy_vs_independent',{}).get('relative'),
                body_strain_energy_Nmm=balance.get('body_strain_energy_Nmm'),spring_strain_energy_Nmm=balance.get('spring_strain_energy_Nmm'),
                net_force_with_spring_N=balance.get('net_force_including_grounded_spring_N'),net_moment_with_spring_Nmm=balance.get('net_moment_including_grounded_spring_Nmm')))
    lines+=['', '固定Sn=I／mean(diag(D rawC D))，仅一个预声明配置。q=[t_mm,L_fit theta_rad]、p=D⁻¹f；q=solve(I+Cn Sn,Cn p)，有效原载荷f_eff=f−D Sn q，再用原六基解响应。原始柔度未对称化／投影／修正，弹簧强度和容差未扫描。',
        '两组三RHS各独立零初值FP64 PCG，每路独立alpha／beta／残差停止，真残差≤1e−8、最多10000步、每25步原子状态保存。端点op.original绑定修改后标量K；六个全原节点字段／平衡先finite、hash、body、Cartesian屏障，再与紧凑全节点位移和参考点运动比较。',
        '平衡显式包含grounded弹簧反力；未计弹簧的中间合力／矩单列，不用它冒充真实平衡。应变能用实体K与弹簧分别计算，与实际外力功和紧凑总能量对照。综合运动误差使用[t_mm,L_fit theta_rad]同单位范数，平移mm／转角rad分列。',
        '**这是修改后近似物理方程的独立数值对照，不是原DeepJEB支撑配置或新的原OptiStruct支撑／载荷GT。**近似curvedP2四Q／weighted-LS RBE3及未知DEBUG／PSOLID控制边界保留，两个样本已揭盲，signedS仍未提供。',
        '每CAD只加载原own NPZ／3 operator数组／10粗CSR数组／6基解各一次，不读取rawH5／deck／GRID／旧mixed场／GT／神经PT，不重建K／diag／粗矩阵或重播旧8guards。必要修改K资格单独计费。全import、缓存、资格、PCG、保存、屏障、对照与退出均计入双时轴。',
        '紧凑更新复用现成六基解，独立PCG复用现成K和原粗预条件器；这些时延表示缓存边界适配与验证成本，不是新几何全冷部署，也不是与点云NN同预算架构排名。原源→六基解77.94036／63.43762秒继承单列，未知完整工程冷启动null。']
    if not a['spring_BC_gate_passed']:lines+=['',f'完整2CAD／6终态分母保留，未验收项为null；固定配置终态：{a.get("reason",a.get("scientific_outcome"))}。']
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
    axes[0].set_ylabel('Independent modified K, 3 zero PCG RHS (s)');axes[1].set_ylabel('Existing 6-basis BC update + fullN U (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Physical reference spring BC '+('PASS'if a['spring_BC_gate_passed']else'FAIL')+'; original new-support GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        spring_BC_gate_passed=a['spring_BC_gate_passed'],CSV_rows=len(rows),PNG_visual_review_pending=True,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','endpoints.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=a['spring_BC_gate_passed'],CSV_rows=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
