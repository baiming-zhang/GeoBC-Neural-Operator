"""Scalar-only four-support rotation evidence, without source or field rereads."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d
def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_all_support_rigid_motion_v120_plan.json');r=d.read('notes/geobc_all_support_rigid_motion_v120_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['actual_export']==0
    assert r['source_sha256']==d.sha('tools/export_geobc_all_support_rigid_motion_v120.py')and r['actual_conditions_verified_at_utc']
    assert a['GT_reads']==a['raw_H5_GRID_reads']==a['geometry_K_coarse_rebuilds']==a['neural_PT_reads']==a['training_updates']==0
    gate=a['support_rigid_motion_gate_passed']
    if gate:
        assert a['prediction_barrier_complete']and a['committed_finite_endpoints']==30 and a['basis_array_reads']==36 and a['owned_array_reads']==30
        assert a['input_cache_reads']==2 and a['operator_array_reads']==6 and a['coarse_array_reads']==20 and a['deck_coordinate_passes']==a['deck_coordinate_hash_reads']==2
    o=d.path('results/geobc_all_support_rigid_motion_v120');o.mkdir(exist_ok=False);rows=[];cads=p['scientific']['selection']['CADs'];names=p['scientific']['implementation']['new_rotation_basis_names']+p['scientific']['implementation']['independent_mixed_names']
    lines=['# 四个真实RBE2支撑转角与30方向物理U响应','',f'固定两CAD／30新独立场门：{"PASS"if gate else"FAIL"}。实际外UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。','',
        '| CAD | 真实参考坐标扫描秒 | 新十二旋转基解秒 | 三独立混合秒 | 全N缓存更新秒 | 混合U最大相对差 | 刚体小转角相对差 |','|---|---:|---:|---:|---:|---:|---:|']
    basis_cost=[];mixed_cost=[];query_cost=[]
    for cad in cads:
        rec=a['case_records'].get(cad,{});metrics=rec.get('compact_vs_independent_mixed',{});groups=rec.get('groups',{});rigid=rec.get('rigid_rotation_numeric_compatibility',{})
        basis=sum(groups['rotation_port'+str(j)]['wall_seconds']for j in range(4))if all('rotation_port'+str(j)in groups for j in range(4))else None
        mixed=groups.get('independent_mixed',{}).get('wall_seconds');query=rec.get('compact_force6_support24_update_fullfield_seconds');source=rec.get('authentic_source_support_coordinates',{})
        basis_cost.append(basis);mixed_cost.append(mixed);query_cost.append(query)
        if metrics:
            umax=max(v['relative']for v in metrics.values());lines.append(f'| {cad} | {source["source_coordinate_seconds"]:.3f} | {basis:.3f} | {mixed:.3f} | {query:.5f} | {umax:.3g} | {rigid["relative"]:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — | — |')
        for name in names:
            v=a.get('endpoint_records',{}).get(cad+'/'+name,{});u=metrics.get(name,{})
            rows.append(dict(CAD=cad,load=name,all_original_nodes=v.get('all_original_nodes'),numerical_state=v.get('numerical_state'),
                initial_free_guess=v.get('initial_free_guess'),true_free_relative_residual=v.get('true_free_relative_residual'),actual_nonzero_free_RHS_norm=v.get('actual_nonzero_free_RHS_norm'),
                all_fixed_boundary_relative=v.get('all_fixed_boundary_relative'),all_fixed_boundary_maximum_absolute_mm=v.get('all_fixed_boundary_maximum_absolute_mm'),
                compact_mixed_relative=u.get('relative'),body_energy_virtualwork_relative=v.get('body_energy_virtualwork_relative'),
                all_support_translations_mm=v.get('all_support_translations_mm'),all_support_rotations_rad=v.get('all_support_rotations_rad'),
                authentic_source_reference_xyz_mm=v.get('authentic_source_reference_xyz_mm'),all_support_reaction_forces_N=v.get('all_support_reaction_forces_N'),
                all_support_local_moments_Nmm=v.get('all_support_local_moments_Nmm'),body_strain_energy_Nmm=v.get('body_strain_energy_Nmm'),
                external_force_work_Nmm=v.get('external_force_work_Nmm'),all_support_motion_work_Nmm=v.get('all_support_motion_work_Nmm'),
                net_force_N=v.get('net_force_including_all_fixed_reaction_N'),net_moment_about_loaded_reference_Nmm=v.get('net_moment_about_loaded_reference_including_all_fixed_reaction_Nmm')))
    lines+=['','每CAD原deck严格hash一次、目标无关卡片流扫描一次。四支撑参考XYZ来自真实GRID／CP=CD=0及GRDSET展开，RBE2 CM123456和成员GRID与原缓存region_support_0..3精确一致；不使用成员中心／定位投影或虚构reference H5行。全卡片计数、原未知DEBUG效果、原支持范围与源身份保留。',
        '每个真实端口新增xyz三方向 .0001rad旋转、外力0，两CAD共24新旋转基解；再六个declared force6＋四端口平移／转角混合解。十组三RHS均独立零自由初值FP64 PCG，每路alpha／beta／真实残差停止独立，原fineK与32步预条件器不改。',
        '几何lift=t_mm＋theta_rad×(vertices_mm−真实reference XYZ_mm)，free0；RHS=b_free−(K·lift)_free，以这个真实非零RHS归一化。完整物理u保留非零固定lift，不以原基解warmstart或大罚刚度近似。每25步原子保存z/r/p/lift/rhs/4×6目标／真实XYZ／端口与成本。',
        '完整30场／真平衡／四端口反力与力矩先finite/hash/body/Cartesian屏障，之后原12force＋24translation基解各一次共享读取；24新旋转基解从屏障常驻复用，未参与独立mixed warmstart或RHS。真实30方向API提供force6＋四端口motion6(mm/rad)到全原节点物理U(mm)。',
        '端口F_N=sum固定反力，M_Nmm=sum((x−真实referenceXYZ)×反力)。bodyenergy=0.5外部nodalforce·u＋0.5Σ(t_mm·F_N＋theta_rad·M_Nmm)；整体力／矩平衡包括所有固定反力，不混单位，也不把未知支撑参考坐标或局部力矩填0。',
        '无外力、四端口相同小转角，平移按theta×(referenceXYZ−加载参考原点)设置，实际缓存响应与全N theta×(vertices−原点)比较；这是线性化刚体转角数值兼容，不是有限旋转／新FEM GT或近零能量归一化。',
        '**两CAD已揭盲，原OptiStruct新支撑／载荷／转角或独立offnodeGT未测。**近似curvedP2四Q／weightedLS RBE3与未知DEBUG／PSOLID限制保留，signedS仍未提供，未证明任意支撑选择或工程资格。',
        '新24旋转基解、6独立混合、真实source XYZ扫描、cache/API/readout与继承旧force6／translation12成本分列。全部import/hash/IO/资格/PCG/checkpoint/屏障/对照/退出计入真实双时轴，不缩放差异，不将并行lane时间相加；既有K／基解缓存不是新几何工程全冷或NN等预算排名，未知完整engineeringcold为null。']
    if not gate:lines+=['',f'完整2CAD／30分母保留，缺失null；固定终态：{a.get("reason",a.get("scientific_outcome"))}。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'endpoints.csv').open('w',encoding='utf-8',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,new_original_support_GT_accuracy=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    if all(v is not None for v in basis_cost+mixed_cost):
        axes[0].bar(cads,basis_cost,label='New support rotation12 basis');axes[0].bar(cads,mixed_cost,bottom=basis_cost,label='Independent mixed3');axes[0].legend();axes[0].set_ylabel('Independent zero PCG groups (s)')
    if all(v is not None for v in query_cost):
        bars=axes[1].bar(cads,query_cost);axes[1].bar_label(bars,fmt='%.4fs')
    axes[1].set_ylabel('Existing force6 + support24 fullN U (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Authentic four support rotations '+('PASS'if gate else'FAIL')+'; original new-BC GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,support_rigid_motion_gate_passed=gate,CSV_rows=len(rows),PNG_visual_review_pending=True,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','endpoints.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=gate,CSV_rows=len(rows)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
