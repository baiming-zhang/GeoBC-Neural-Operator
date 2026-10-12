"""Scalar-only independent physics endpoint/reference report, after exact exit."""
import argparse,time
from geobc import available_cad_pilot_data as d


def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_load_linear_independent_physics_v100_plan.json');r=d.read('notes/geobc_load_linear_independent_physics_v100_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['source_sha256']==d.sha('tools/export_geobc_load_linear_physics_v100.py')
    assert a['training_updates']==a['scaler_fits']==a['neural_PT_reads']==a['neural_inference']==a['FINAL_reads']==0
    o=d.path('results/geobc_load_linear_independent_physics_v100');o.mkdir(exist_ok=False);rows=[]
    lines=['# 一个开发CAD的独立近似物理六载荷对照','','固定120_153，三规范载荷＋三新混合载荷，两GPU并行组；每个载荷独立零初值PCG。原始K／几何只建一次。近似curvedP2／fourQ／RBE3，**不是原OptiStruct新载荷真值或工程资格**。','',f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒；数值与来源结果：{a.get("scientific_outcome")}; 固定分母6，有限输出{a["committed_finite_endpoints"]}/6，既有原physicalGT读取{a["GT_reads"]}次。']
    full=a['protocol_qualified']and a['committed_finite_endpoints']==6
    if full:
        assert a['prediction_barrier_complete']and a['GT_reads']==a['GT_attempts']==3
        q=a['parallel_K_qualification'];assert q['passed']and q['actual_all_original_nodes']==188196 and q['actual_RHS_lanes']==3
        assert a['algebra_guards']['K_applications']==8 and a['H5_input_reads']==a['GRID_map_reads']==a['deck_parses']==a['deck_header_reads']==1
        assert a['input_operator_seconds']<=300 and a['combined_solve_seconds']<=900 and a['final_readout_seconds']<=300
        lines+=['','| 独立物理规范载荷 | 对原GT的U误差 | 真残差 |','|---|---:|---:|']
        for load in ('ver','hor','tor'):
            v=a['canonical_original_GT_U']['120_153/'+load];ep=a['endpoint_records'][load];x=v['U_relative_L2'];rows.append(dict(role='physics_canonical_vs_originalGT',candidate='P2',key=load,U_relative_L2=x,true_relative_residual=ep['true_relative_residual'],numerical_state=ep['numerical_state']))
            lines.append(f'| {load} | '+(f'{100*x:.5f}%'if x is not None else'undefined')+f' | {ep["true_relative_residual"]:.3g} |')
        lines+=['','| 神经骨干 | 原模型混合U误差 | 线性接口混合U误差 |','|---|---:|---:|']
        for name,label in [('FNO20','FNO'),('FNO_HASH20','FNO+Hash'),('TS20','Transolver')]:
            c=a['mixed_neural_U_against_independent_approx_P2'][name]
            lines.append(f'| {label} | '+(f'{100*c["raw"]["mean_U"]:.2f}%'if c['raw']['mean_U']is not None else'undefined')+' | '+(f'{100*c["adapter"]["mean_U"]:.2f}%'if c['adapter']['mean_U']is not None else'undefined')+' |')
            for kind,v in c.items():
                for key,metric in v['per_case'].items():rows.append(dict(role='neural_'+kind+'_vs_independent_approxP2',candidate=name,key=key,U_relative_L2=metric['U_relative_L2'],true_relative_residual=None,numerical_state=None))
        for load in ('mix0','mix1','mix2'):
            v=a['endpoint_records'][load];rows.append(dict(role='physics_mixed_equilibrium',candidate='P2',key=load,U_relative_L2=None,true_relative_residual=v['true_relative_residual'],numerical_state=v['numerical_state']))
        assert len(rows)==24
        lines+=['',f'预声明兼容门：'+('PASS'if a['physics_reference_compatible']else'FAIL')+'。必须全6真残差≤1e−8且原3规范U误差各≤1%，才增强独立近似参考的证据；不满足时，上表混合误差仅有限端点诊断，不能当精度验收。', '', '三个新混合载荷参照是独立零初值求得的近似物理场，未使用原GT线性组合求解，和v99的条件派生参考分开。旧神经预测18份复用，无PT／推理／训练。一个已迭代开发CAD、仅3组合，不确定总体求解器；原signedS不生成新参考。']
    else:
        lines+=['','完整物理参考未成立，保留全部6分母，无子集精度／补零排名。原因：'+str(a.get('reason',a.get('scientific_outcome')))+'。不自动重试、换CAD、放宽残差或扫配置。']
    lines+=['','实际输入、GRID映射、材料／约束／每节点LOAD动作、力／矩／虚功、几何／8次旧K guards、3列实际fullN并行K一致性、每组状态／残差／成本及所有控制排除见JSON。额外并行一致性验证3次旧K＋1次三列K如实记账，未重复原8guards。10秒终点预留仍包含在每组300秒内；并行组wall按独占实测计，不把并行三列时间累加成串行成本。','', '四点曲面二次元积分与RBE3加权拟合有明确近似；原DEBUG LGSHPSNS／PSOLID数值设置未复现。既有五例ver约0.016%只是先前参考，未重新运行或推断本次必然通过；本报告不读GT或重算场。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import csv
    with(o/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        names=['role','candidate','key','U_relative_L2','true_relative_residual','numerical_state'];w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,independent_original_OptiStruct_newloadGT=False,approximate_P2_reference_compatible=a['physics_reference_compatible']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4))
    if full:
        names=['FNO20','FNO_HASH20','TS20'];labels=['FNO','FNO+Hash','Transolver']
        for off,kind,label in [(-.18,'raw','Raw model'),(.18,'adapter','Load-linear API')]:
            vals=[a['mixed_neural_U_against_independent_approx_P2'][n][kind]['mean_U']for n in names]
            if all(v is not None for v in vals):
                bars=ax.bar([i+off for i in range(3)],[100*v for v in vals],width=.36,label=label);ax.bar_label(bars,fmt='%.2f',fontsize=8)
        ax.set_xticks(range(3),labels);ax.set_ylabel('U error vs independent approximate P2 (%)');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title('One iterated DEV CAD /3 mixtures; reference gate '+('PASS'if a['physics_reference_compatible']else'FAIL'))
    else:
        ax.axis('off');ax.text(.5,.5,'Fixed six-load physics protocol negative\nNo complete reference/no subset accuracy ranking',ha='center',va='center')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed'if full else'completed_scientific_negative',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','per_case.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,physics_reference_compatible=a['physics_reference_compatible']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
