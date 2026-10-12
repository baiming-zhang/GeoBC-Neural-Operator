"""Scalar evidence for all actual support translations; no scientific body reads."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d
def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_all_support_translation_v118_plan.json');r=d.read('notes/geobc_all_support_translation_v118_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['actual_export']==0
    assert r['source_sha256']==d.sha('tools/export_geobc_all_support_translation_v118.py')and r['actual_conditions_verified_at_utc']
    assert a['GT_reads']==a['raw_H5_deck_GRID_reads']==a['geometry_K_coarse_rebuilds']==a['neural_PT_reads']==a['training_updates']==0
    if a['support_translation_gate_passed']:
        assert a['prediction_barrier_complete']and a['committed_finite_endpoints']==24 and a['basis_array_reads']==18 and a['owned_array_reads']==24
        assert a['input_cache_reads']==2 and a['operator_array_reads']==6 and a['coarse_array_reads']==20
    o=d.path('results/geobc_all_support_translation_v118');o.mkdir(exist_ok=False);rows=[];cads=p['scientific']['selection']['CADs'];names=p['scientific']['implementation']['new_support_basis_names']+p['scientific']['implementation']['independent_mixed_names']
    gate=a['support_translation_gate_passed'];lines=['# 四个实际源支撑平移端口与加载参考点六维力矩','',f'固定两CAD／24新独立场：{"PASS"if gate else"FAIL"}。实际外UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。','',
        '| CAD | 真实四支撑reference GRID | 剩余九基解秒 | 三独立混合秒 | 全N缓存更新秒 | 混合U最大相对差 | 刚体平移相对差 |','|---|---|---:|---:|---:|---:|---:|']
    basis_cost=[];mixed_cost=[];query_cost=[]
    for cad in cads:
        rec=a['case_records'].get(cad,{});metrics=rec.get('compact_vs_independent_mixed',{});groups=rec.get('groups',{});rigid=rec.get('rigid_translation_numeric_compatibility',{})
        basis=sum(groups['support_port'+str(j)]['wall_seconds']for j in (1,2,3))if all('support_port'+str(j)in groups for j in (1,2,3))else None
        mixed=groups.get('independent_mixed',{}).get('wall_seconds');query=rec.get('compact_force6_support12_update_fullfield_seconds')
        basis_cost.append(basis);mixed_cost.append(mixed);query_cost.append(query)
        if metrics:
            ports=rec['source_support_ports']['ports'];umax=max(v['relative']for v in metrics.values())
            lines.append(f'| {cad} | {[v["source_reference_GRID_ID"]for v in ports]} | {basis:.3f} | {mixed:.3f} | {query:.5f} | {umax:.3g} | {rigid["relative"]:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — | — |')
        for name in names:
            v=a.get('endpoint_records',{}).get(cad+'/'+name,{});u=metrics.get(name,{})
            rows.append(dict(CAD=cad,load=name,all_original_nodes=v.get('all_original_nodes'),numerical_state=v.get('numerical_state'),
                initial_free_guess=v.get('initial_free_guess'),true_free_relative_residual=v.get('true_free_relative_residual'),
                actual_nonzero_free_RHS_norm=v.get('actual_nonzero_free_RHS_norm'),all_fixed_boundary_relative=v.get('all_fixed_boundary_relative'),
                all_fixed_boundary_maximum_absolute_mm=v.get('all_fixed_boundary_maximum_absolute_mm'),compact_mixed_relative=u.get('relative'),
                body_energy_virtualwork_relative=v.get('body_energy_virtualwork_relative'),all_support_translations_mm=v.get('all_support_translations_mm'),
                all_support_reaction_forces_N=v.get('all_support_reaction_forces_N'),all_support_local_moments_Nmm=v.get('all_support_local_moments_Nmm'),
                body_strain_energy_Nmm=v.get('body_strain_energy_Nmm'),external_force_work_Nmm=v.get('external_force_work_Nmm'),
                all_support_motion_work_Nmm=v.get('all_support_motion_work_Nmm'),net_force_N=v.get('net_force_including_all_fixed_reaction_N'),
                net_moment_about_loaded_reference_Nmm=v.get('net_moment_about_loaded_reference_including_all_fixed_reaction_Nmm')))
    lines+=['','实际源RBE2按reference GRID排序对应region_support_0..3。每个端口真实fixed成员和已存在master输出行均核验；四端口互不交叠、并集恰等于原fixed分区，转角始终0，未生成缺失reference节点。',
        '仅剩余端口1／2／3各三方向(.01mm)新求解，旧port0不重跑。每CAD九新支撑基解及三四端口＋force6混合解，共八组三RHS／24独立零初值FP64场；每路alpha、beta、真实残差停止独立，固定32步原预条件器及原fineK不改。',
        '几何lift在真实各fixed端口赋对应运动、自由行0；RHS=b_free−(K·lift)_free，用此真实非零RHS归一化，包括纯支撑运动外力0的情况。完整物理u保留非零fixedlift；每25步原子保存z/r/p/lift/rhs/四端口目标与成本。',
        '全部24场、真平衡和四端口反力完成finite/hash/body/Cartesian屏障后，12旧force6基解与6旧port0基解各一次共享读取。新18基解从屏障常驻复用，未用于独立混合warmstart或RHS。实际18方向API由force6和四支撑translation4×3求全N物理U。',
        '能量包含0.5外部nodalforce·u＋0.5所有支撑motion·reaction，整体平衡纳入全部固定反力。局部master力矩没有真实参考XYZ时明确null。四支撑同.01mm x平移、无外力的刚体见证是数值兼容性，不是新FEM真值精度。',
        '**两CAD已揭盲；这是近似curvedP2／四Q／weightedLS RBE3的数值响应，未知DEBUG／PSOLID仍保留。** 未测原OptiStruct新支撑或载荷GT，不证明任意支撑选择、支撑旋转或工程资格，signedS未解决。',
        '剩余18基解、6独立混合、cache／API成本分别记录；继承旧force6源到基解77.94036／63.43762秒、旧port0基解34.76535／34.90065秒，均不重跑。全stage包括import/hash/IO/资格/PCG/checkpoint/屏障/对照/退出；双时轴不缩放、未知工程全cold为null，缓存接口不是NN等预算排序。']
    if not gate:lines+=['',f'完整2CAD／24分母保留，缺失null，固定终态：{a.get("reason",a.get("scientific_outcome"))}。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'endpoints.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,new_original_support_GT_accuracy=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    if all(v is not None for v in basis_cost+mixed_cost):
        axes[0].bar(cads,basis_cost,label='Remaining support9 basis');axes[0].bar(cads,mixed_cost,bottom=basis_cost,label='Independent mixed3');axes[0].legend();axes[0].set_ylabel('Independent zero PCG groups (s)')
    if all(v is not None for v in query_cost):
        bars=axes[1].bar(cads,query_cost);axes[1].bar_label(bars,fmt='%.4fs')
    axes[1].set_ylabel('Existing force6 + support12 fullN U (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('All four source support translations '+('PASS'if gate else'FAIL')+'; new original-BC GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,support_translation_gate_passed=gate,CSV_rows=len(rows),PNG_visual_review_pending=True,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','endpoints.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=gate,CSV_rows=len(rows)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
