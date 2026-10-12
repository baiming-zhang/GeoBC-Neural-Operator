"""Scalar-only twoCAD paired precision, canonical GT compatibility and cost."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physics_crossCAD_paired_v103_plan.json');r=d.read('notes/geobc_physics_crossCAD_paired_v103_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest'] and r['source_sha256']==d.sha('tools/export_geobc_physics_crossCAD_v103.py')
    assert a['neural_PT_reads']==a['neural_inference']==a['training_updates']==a['scaler_fits']==a['FINAL_reads']==a['raw_CSV_H5_nodal_target_reads']==0
    full=a['protocol_qualified'] and a['committed_finite_endpoints']==18;precision=full and a['physics_reference_compatible'];cost=full and a['cost_gate_passed'];whole=max(e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])
    o=d.path('results/geobc_physics_crossCAD_paired_v103');o.mkdir(exist_ok=False);rows=[];table=[]
    lines=['# 两个额外CAD的物理精度与粗预条件器成本','', '固定131_345／131_351，来自已迭代开发CAD、按字典序选取，未按误差挑例。FP64曲面P2／四点积分／近似RBE3，原三载荷blockJacobi与固定32步粗空间配对，另求三混合载荷；每CAD共享一个K，每个载荷独立零初值和PCG系数。','',f'完整阶段{whole:.2f}秒；数值端点{a["committed_finite_endpoints"]}/18，GT读取{a["GT_reads"]}/6。全精度／原GT兼容门 '+('PASS'if precision else'FAIL')+'；两CAD均至少25%提速门 '+('PASS'if cost else'FAIL')+'。']
    if full:
        assert a['prediction_barrier_complete'] and a['GT_attempts']==a['GT_reads']==6
        assert a['H5_input_reads']==a['GRID_map_reads']==a['deck_parses']==a['deck_header_reads']==2 and a['combined_solve_seconds']<=1200 and a['readout_seconds']<=120
        assert len(a['endpoint_records'])==18 and len(a['canonical_original_GT_U'])==6 and len(a['same_solution_metrics'])==12
        lines+=['','| CAD | 三规范U最大误差 | 普通三载荷耗时 | 粗构造＋粗三载荷 | 提速 |','|---|---:|---:|---:|---:|']
        for cad in p['scientific']['selection']['CADs']:
            c=a['case_records'][cad];assert c['algebra_guards']['K_applications']==8 and c['parallel_K_qualification']['passed'] and c['parallel_M_qualification']['passed'] and c['coarse_algebra_guards']['passed']
            vals=[a['canonical_original_GT_U'][cad+'/'+load]['coarse_canonical']['U_relative_L2']for load in ('ver','hor','tor')];maxU=max(vals)if all(v is not None for v in vals)else None
            table.append(dict(CAD=cad,maximum_canonical_U=maxU,block_seconds=c['paired_primary_baseline_seconds'],coarse_seconds=c['paired_primary_coarse_seconds'],reduction_fraction=c['paired_reduction_fraction'],precision_compatible=all(v is not None and v<=.01 for v in vals),cost_passed=c['paired_cost_gate_passed']))
            lines.append(f'| {cad} | '+(f'{100*maxU:.5f}%'if maxU is not None else'undefined')+f' | {c["paired_primary_baseline_seconds"]:.2f}s | {c["paired_primary_coarse_seconds"]:.2f}s | {100*c["paired_reduction_fraction"]:.2f}% |')
        for key,record in a['endpoint_records'].items():
            parts=key.split('/');cad,role,name=parts
            gt=a['canonical_original_GT_U'].get(cad+'/'+name,{}).get(role,{})
            numeric_key=cad+('/mixed/'if role=='coarse_mixed'else'/canonical/')+name
            numerical=a['same_solution_metrics'].get(numeric_key,{})if role!='block_canonical'else{}
            rows.append(dict(key=key,role=role,U_error_vs_original_GT=gt.get('U_relative_L2'),U_same_system_relative=numerical.get('U_relative_L2'),true_relative_residual=record['true_relative_residual'],iterations=record['iterations'],numerical_state=record['numerical_state']))
        assert len(rows)==18
    else:lines+=['','协议未完成，不按子集宣布精度或成本赢家；固定18数值／6GT分母保留。原因：'+str(a.get('reason',a.get('scientific_outcome')))+'。']
    lines+=['','主要成本为同三规范载荷的配对：粗构造、必要粗代数／三列资格、缓存保存与粗三载荷求解，比较同K实测普通三载荷求解。第二组三混合载荷两边都不计入配对主指标，但完整阶段成本始终计入。共同原源／K／8guard每CAD一次单列，不把共享准备摊成0或称fresh全冷请求；不相加并行三列的时间。','原GT仅规范三载荷位移：所有18全原节点有限字段和真实平衡先提交、完整复核后才GT6共享一次。粗规范场与真实独立零初值普通解对照；粗混合场与同方程三规范基解的线性组合对照，后者明确是数值派生参考，**不是独立原OptiStruct混合载荷真值**。','固定32次零初值阻尼Jacobi，omega=1/Gamma，加性系数均1；corner／midedge／约束mask的PtAP保持原曲面P2数学。全N三列K／M与逐列资格、两个Galerkin向量、受约束中点的非嵌套事实、每个字段的残差／反力／力矩和全成本见JSON；不是全谱或工程资格证明。','两个已迭代CAD、可用性条件选样，不代表总体盲测或最终架构赢家。原DEBUG LGSHPSNS／PSOLID数值设置未复现，四点积分／加权RBE3保留近似；应力出口契约仍未解决，不生成signedS参考。负结果保留，不调步数／lambda／容差／预算或更换CAD挽救。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        names=['key','role','U_error_vs_original_GT','U_same_system_relative','true_relative_residual','iterations','numerical_state'];w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,precision_gate_passed=precision,cost_gate_passed=cost,whole_seconds=whole,paired_table=table,freshdeployment_fullcold_seconds=None,new_OptiStruct_mixedGT=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    if full:
        for offset,col,label in [(-.18,'block_seconds','Block Jacobi /3 loads'),(.18,'coarse_seconds','Coarse setup +3 loads')]:
            bars=axes[0].bar([i+offset for i in range(2)],[v[col]for v in table],width=.36,label=label);axes[0].bar_label(bars,fmt='%.1f',fontsize=8)
        axes[0].set_xticks(range(2),[v['CAD']for v in table]);axes[0].set_ylabel('Actual paired seconds');axes[0].legend(fontsize=7)
        for i,cad in enumerate(p['scientific']['selection']['CADs']):
            vals=[a['canonical_original_GT_U'][cad+'/'+load]['coarse_canonical']['U_relative_L2']for load in ('ver','hor','tor')]
            if all(v is not None for v in vals):axes[1].plot(range(3),[100*v for v in vals],'o-',label=cad)
        axes[1].set_xticks(range(3),['ver','hor','tor']);axes[1].set_ylabel('U error vs original canonical GT (%)');axes[1].legend(fontsize=8)
    else:
        for ax in axes:ax.axis('off');ax.text(.5,.5,'Fixed two-CAD protocol negative\nNo subset accuracy/cost winner',ha='center',va='center')
    fig.suptitle('Two iterated DEV CADs / precision '+('PASS'if precision else'FAIL')+' / paired cost '+('PASS'if cost else'FAIL'));fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed'if full else'completed_scientific_negative',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,precision_gate_passed=precision,cost_gate_passed=cost,new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','metrics.json','per_case.csv','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,precision_gate=precision,cost_gate=cost))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
