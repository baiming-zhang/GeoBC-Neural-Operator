"""CPU scalar-only report after real reusable API and108-prediction run."""
import argparse
import csv
import time
from geobc import available_cad_pilot_data as d


def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_load_linear_adapter_v96_plan.json');s=p['scientific']
    r=d.read('notes/geobc_load_linear_adapter_v96_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['source_sha256']==d.sha('tools/export_geobc_load_linear_v96.py')
    assert a['protocol_qualified']and a['prediction_barrier_complete']and a['committed_finite_predictions']==108 and a['property_rows']==81
    assert a['GT_reads']==a['CSV_H5_deck_reads']==a['training_updates']==a['scaler_fits']==0 and a['checkpoint_hash_reads']==a['checkpoint_loads']==3
    assert a['input_tensor_bytes_unchanged']and a['input_array_attempts']==a['input_array_reads']==33
    out=d.path('results/geobc_load_linear_adapter_v96');out.mkdir(exist_ok=False)
    names=['FNO20','FNO_HASH20','TS20'];labels=['FNO','FNO+Hash','Transolver'];rows=[]
    lines=['# 显式载荷线性位移接口：真实调用验证','','**同三个冻结20遍骨干、旧三CAD×三载荷；真实可复用接口生成108份全节点预测，GT读取／训练／scaler拟合0。通过是接口与物理结构一致性，尚不是新载荷GT准确率改善。**','',
        '| 骨干 | 原幅值响应偏差 | 接口幅值响应偏差 | 规范载荷U变化 |','|---|---:|---:|---:|']
    old=d.read(s['physics_metadata_identity']['path'])
    for name,label in zip(names,labels):
        row=a['candidates'][name];q=row['qualification'];v=row['properties']
        assert row['strict_checkpoint_verified']and row['model_bytes_unchanged']and row['predictions']==36
        assert q['passed']and q['original_nodes']==242936 and q['actual_queries']==64 and q['coordinate_gradient_L2']>0 and q['in_span_three_coefficient_gradient_L2']>0
        assert v['all_cases_complete']and v['amplitude_fixed_denominator']==18 and v['mixture_fixed_denominator']==9 and v['canonical_fixed_denominator']==9 and v['stress_fixed_denominator']==27
        lines.append(f'| {label} | {100*old["candidates"][name]["properties"]["mean_relative_U_defect"]:.2f}% | {100*v["mean_amplitude_defect"]:.3g}% | {100*v["worst_canonical_U_difference"]:.3g}% |')
        for kind,field in [('amplitude','amplitude_per_case'),('mixture','mixture_per_case'),('canonical_U','canonical_per_case'),('raw_encodedS','raw_encodedS_per_case')]:
            for key,value in v[field].items():rows.append(dict(candidate=name,kind=kind,key=key,value=value,metric='relative_defect'if kind!='raw_encodedS'else'max_absolute_encodedS_difference'))
    lines+=['',f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，全部预声明接口不变量：{"PASS"if a["all_model_invariants_passed"]else"FAIL"}。参数保持原骨干；新增模型权重／优化更新0，加载自身原检查点3次，旧输入33数组复用。实际准备／多基响应／推理／参考预测读取成本分别保留在JSON。',
        '', '每CAD的三个真实规范力／力矩列组成秩3矩阵，在此子空间求系数并组合先解码后的物理单位U。原signedS继续使用请求载荷下的原模型输出，不施加应力叠加、符号或缩放规律。原规范载荷位移与原encodedS逐案例对照旧不可变预测；全部幅值／叠加缺陷已降到FP32输出舍入量级。',
        '', '实际最大242936节点几何上，64个真实坐标查询及三载荷系数方向的梯度非零／有限、chunk一致性、模型／输入字节保持通过。此处不是六独立载荷方向全部有效，也不是独立离节点FEM准确率证明。',
        '', '同geometry／MAT1／零支撑的输入证据来自上一阶段。原DEBUG／PSOLID等完整数值控制效果未重建，保留局限；内部线性结构由构造保证，不能用零缺陷声称位移GT误差归零、signed应力正确或骨干最终胜出。下一步在原CAD隔离开发划分检验真实物理单位指标与条件性派生位移参考，单独冻结后再读GT。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len(rows)==189
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,baseline_property_means={k:old['candidates'][k]['properties']['mean_relative_U_defect']for k in names},GT_accuracy_measured=False,invariants_passed=a['all_model_invariants_passed']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    values=[100*old['candidates'][n]['properties']['mean_relative_U_defect']for n in names];bars=axes[0].bar(labels,values);axes[0].bar_label(bars,fmt='%.2f');axes[0].set_ylabel('Original mean U homogeneity defect (%)')
    for i,n in enumerate(names):
        v=a['candidates'][n]['properties'];axes[1].scatter(i,100*v['worst_amplitude_defect'],label='Amplitude'if i==0 else None,marker='o');axes[1].scatter(i,100*v['worst_mixture_defect'],label='Superposition'if i==0 else None,marker='s')
    axes[1].axhline(1e-4,color='black',linestyle='--',label='Declared1e-6 relative tolerance')
    axes[1].set_yscale('log');axes[1].set_xticks(range(3),labels);axes[1].set_ylabel('Adapter worst internal U defect (%)');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('True reusable API/fullN/coordinate+load gradients; consistency is not GT accuracy')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig);assert(out/'comparison.png').stat().st_size>0
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','metrics.json','per_case.csv','comparison.png'],PNG_visual_review_pending=True,invariants_passed=a['all_model_invariants_passed'],GT_accuracy_measured=False))
    print(dict(status='completed',actual_CPU_seconds=time.perf_counter()-began,invariants_passed=a['all_model_invariants_passed']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
