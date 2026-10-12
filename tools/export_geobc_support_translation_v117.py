"""Scalar-only source support port evidence; all new fields retain denominators."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_support_translation_port_v117_plan.json');r=d.read('notes/geobc_support_translation_port_v117_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['actual_export']==0
    assert r['source_sha256']==d.sha('tools/export_geobc_support_translation_v117.py')
    assert a['GT_reads']==a['raw_H5_deck_GRID_reads']==a['geometry_K_coarse_rebuilds']==a['neural_PT_reads']==a['training_updates']==0
    if a['support_translation_gate_passed']:
        assert a['prediction_barrier_complete']and a['committed_finite_endpoints']==12 and a['input_cache_reads']==2 and a['basis_array_reads']==12
        assert a['operator_array_reads']==6 and a['coarse_array_reads']==20 and a['owned_array_reads']==12
        assert all(v['passed']and v['physics_basis_reads_before_independent_solve']==0 for v in a['case_records'].values())
    o=d.path('results/geobc_support_translation_port_v117');o.mkdir(exist_ok=False);rows=[]
    lines=['# 实际RBE2支撑平移端口与加载参考点六维力矩响应','',f'固定两CAD／12新场门：{"PASS"if a["support_translation_gate_passed"]else"FAIL"}；外部UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。',
        '', '选择原源RBE2中reference GRID最小的支撑，三平移非零、转角0，其他三支撑仍0；加载RBE3六维力矩保持独立。端口来自真实源reference排序和region_support_0成员，不按文件名或误差选取，H5没有master行时不生成替代节点。',
        '', '| CAD | 支撑reference／成员行 | 新三支撑基解秒 | 三独立混合解秒 | 九方向更新＋全N秒 | 混合U最大相对差 |',
        '|---|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        rec=a['case_records'].get(cad,{});metrics=rec.get('compact_vs_independent_mixed',{});port=rec.get('source_support_port',{})
        if metrics:
            umax=max(v['relative']for v in metrics.values())
            lines.append(f'| {cad} | {port["source_reference_GRID_ID"]}／{port["physical_member_rows"]} | {rec["groups"]["support_basis"]["wall_seconds"]:.3f} | {rec["groups"]["independent_mixed"]["wall_seconds"]:.3f} | {rec["compact_force6_support3_update_fullfield_seconds"]:.5f} | {umax:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — |')
        for name in p['scientific']['selection']['new_support_basis_names']+p['scientific']['selection']['independent_mixed_names']:
            record=a.get('endpoint_records',{}).get(cad+'/'+name,{});v=metrics.get(name,{})
            rows.append(dict(CAD=cad,load=name,all_original_nodes=record.get('all_original_nodes'),initial_free_guess=record.get('initial_free_guess'),
                geometry_only_fixed_lift=record.get('geometry_only_fixed_lift'),numerical_state=record.get('numerical_state'),true_free_relative_residual=record.get('true_free_relative_residual'),
                actual_nonzero_free_RHS_norm=record.get('actual_nonzero_free_RHS_norm'),selected_fixed_boundary_relative=record.get('selected_fixed_boundary_relative'),
                other_fixed_maximum_absolute_mm=record.get('other_fixed_maximum_absolute_mm'),compact_mixed_fullN_relative=v.get('relative'),
                body_energy_virtualwork_relative=record.get('body_energy_virtualwork_relative'),selected_support_translation_mm=record.get('selected_support_translation_mm'),
                selected_support_reaction_force_N=record.get('selected_support_reaction_force_N'),selected_support_local_moment_Nmm=record.get('selected_support_local_moment_Nmm'),
                body_strain_energy_Nmm=record.get('body_strain_energy_Nmm'),external_force_work_Nmm=record.get('external_force_work_Nmm'),selected_support_motion_work_Nmm=record.get('selected_support_motion_work_Nmm'),
                net_force_including_all_fixed_reaction_N=record.get('net_force_including_all_fixed_reaction_N'),net_moment_about_loaded_reference_Nmm=record.get('net_moment_about_loaded_reference_including_all_fixed_reaction_Nmm')))
    lines+=['', '每CAD三新支撑单位基解(.01mm)的外力为0，再三个source force6＋支撑运动的独立混合解。四组三RHS各独立零初值自由未知量、FP64／每路alpha beta残差停止，真实RHS=b_free−K·lift_free；纯支撑运动用非零−K·lift归一化，不用零外力范数。每25步原子保存z/r/p/lift/rhs/目标与成本。',
        '实际完整物理场保留固定节点的非零平移，再填自由位移，不用旧零边界restore丢掉lift。真实region0所有成员属于原fixed、与其他regions无交叠；master仅已有H5行才加入。实体几何、材料、free/fixed分区、fineK与原32步预条件器保持；不重建K／diag／粗矩阵或重播旧8guards。',
        '完整12新字段／真平衡／支撑反力先finite、hash、body、Cartesian屏障，之后12旧六力矩基解各一次；新支撑基解没有用于独立混合解warmstart或RHS。真实九方向API=sum(forceB⁻¹f·旧6基解)+sum(支撑平移/.01·新3基解)，固定六混合场全原节点对照。',
        'bodyenergy=0.5外部nodalforce·u+0.5所选supporttranslation·portreaction，其余不动支撑做功0；平衡含所有固定支撑反力。局部支撑master力矩没有实际参考XYZ时明确null，整体力矩仍关于已有加载reference XYZ；不假称新支撑六旋转方向已支持。',
        '**这是声明的新非零支撑方程及多端口数值响应，不是原OptiStruct新支撑／载荷GT。**近似curvedP2四Q／weightedLS RBE3及未知DEBUG／PSOLID效果保留，两样本已揭盲，不证明任意支撑选择或工程资格，signedS仍未提供。',
        '新三支撑基解、独立混合验证、cache／API时间与继承源到旧六力矩基解77.94036／63.43762秒分列。所有import、缓存hash／IO、资格、PCG、保存、屏障、对照、失败退出计入双时轴；未知完整工程冷启动null，缓存接口不是NN同预算排名。']
    if not a['support_translation_gate_passed']:lines+=['',f'完整2CAD／12终态分母保留，未验收项null；固定终态：{a.get("reason",a.get("scientific_outcome"))}。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'endpoints.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,new_original_support_GT_accuracy=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));cads=p['scientific']['selection']['CADs']
    basis=[a['case_records'].get(cad,{}).get('groups',{}).get('support_basis',{}).get('wall_seconds')for cad in cads]
    mixed=[a['case_records'].get(cad,{}).get('groups',{}).get('independent_mixed',{}).get('wall_seconds')for cad in cads]
    if all(v is not None for v in basis+mixed):
        axes[0].bar([0,1],basis,label='New support3 basis');axes[0].bar([0,1],mixed,bottom=basis,label='Independent mixed3')
        axes[0].set_xticks([0,1],cads);axes[0].legend();axes[0].set_ylabel('Independent zero PCG groups (s)')
    compact=[a['case_records'].get(cad,{}).get('compact_force6_support3_update_fullfield_seconds')for cad in cads]
    if all(v is not None for v in compact):
        bars=axes[1].bar(cads,compact);axes[1].bar_label(bars,fmt='%.4fs')
    axes[1].set_ylabel('Existing force6 + new support3, fullN U (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Source support translation port '+('PASS'if a['support_translation_gate_passed']else'FAIL')+'; original new-BC GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        support_translation_gate_passed=a['support_translation_gate_passed'],CSV_rows=len(rows),PNG_visual_review_pending=True,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','endpoints.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=a['support_translation_gate_passed'],CSV_rows=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
