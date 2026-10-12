"""One scalar-only CPU report after exact terminal540/GT60 validation."""
import argparse
import time
from geobc import available_cad_pilot_data as d


def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_load_linear_knownDEV_v97_plan.json')
    r=d.read('notes/geobc_load_linear_knownDEV_v97_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']and r['source_sha256']==d.sha('tools/export_geobc_load_linear_knownDEV_v97.py')
    assert a['protocol_qualified']and a['prediction_barrier_complete']and a['committed_finite_predictions']==540
    assert a['source_deck_parses']==a['source_deck_parse_attempts']==a['source_deck_hash_reads']==a['source_deck_header_reads']==20
    assert len(a['physics_gate'])==20 and all(v['same_constraints_across3']and v['zero_constraints']and v['full_card_inventory']for v in a['physics_gate'].values())
    assert a['GT_reads']==a['owned_GT_attempts']==60 and a['checkpoint_hash_reads']==a['checkpoint_loads']==3
    assert a['input_tensor_bytes_unchanged']and a['training_updates']==a['scaler_fits']==a['CSV_H5_reads']==0
    out=d.path('results/geobc_load_linear_knownDEV_v97');out.mkdir(exist_ok=False)
    names=['FNO20','FNO_HASH20','TS20'];labels=['FNO','FNO+Hash','Transolver'];rows=[]
    lines=['# 新载荷位移：CAD隔离开发集对照','','**20个已迭代knownDEV CAD×三规范载荷；540份完整原节点预测先全部提交并重验，再共享原physicalGT60一次。新混合载荷位移使用条件线性静力派生参考，未独立重新求解新载荷FEM。**','',
        '| 冻结20遍骨干 | 规范U误差 | 规范S误差 | 原模型混合U误差 | 线性接口混合U误差 |','|---|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        row=a['candidates'][name];q=row['qualification'];pres=row['canonical_preservation'];can=row['canonical_full60'];mix=row['conditional_mixed_U']
        assert row['strict_checkpoint_verified']and row['model_bytes_unchanged']and row['predictions']==180
        assert q['passed']and q['actual_queries']==64 and q['coordinate_gradient_L2']>0 and q['in_span_three_coefficient_gradient_L2']>0
        assert len(row['geometry_profiles'])==20 and pres['passed']and pres['all60_complete']and len(pres['per_case_U_difference'])==60
        assert can['all_cases_complete']and can['fixed_denominator']==60 and len(can['per_case'])==60
        assert all(v['all60_complete']and v['fixed_denominator']==60 and len(v['per_case'])==60 for v in mix.values())
        lines.append(f'| {label} | {100*can["case_mean_u_relative_l2"]:.2f}% | {100*can["case_mean_stress_relative_l2"]:.2f}% | {100*mix["raw"]["mean_U"]:.2f}% | {100*mix["adapter"]["mean_U"]:.2f}% |')
        for key,v in can['per_case'].items():rows.append(dict(candidate=name,role='canonical_actual_GT',key=key,U_relative_L2=v['u_relative_l2'],S_relative_L2=v['stress_relative_l2'],reference='original_aligned_U_mm_S_MPa'))
        for kind,items in mix.items():
            for key,v in items['per_case'].items():rows.append(dict(candidate=name,role='mixed_'+kind,key=key,U_relative_L2=v['u_relative_l2'],S_relative_L2=None,reference=v['reference']))
    assert len(rows)==540
    lines+=['',f'真实完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，训练／scaler拟合0。每模型60个混合场共享同20CAD，固定三组合；不是180个独立CAD或最终盲测。模型容量和精度仍不同，时间包含来源解析、输入／PT／初始化资格、预测保存／屏障、参考读取与GT指标。',
        '', '规范载荷U和原encodedS与旧不可变预测逐案例保真，原GT全节点位移／应力／热点指标分别保留。三种模型的真实最大DEV几何和64个坐标／秩3系数查询梯度通过，模型与输入保持；不证明离节点FEM精度或六个独立载荷DOF。',
        '', '混合U参考是固定系数sum_i c_i原规范GT_U，仅在同几何／MAT1／零支撑／固定K小变形的假设下成立。完整源卡、活动STATICS／实际力矩与输入raster证据已经保存；STATICS本身不充分证明线性。原DEBUG LGSHPSNS／PSOLID数值控制未完整重建，条件性与保真局限仍保留。原signedS不套叠加／缩放／符号规则，不生成混合应力参考。',
        '', '条件mixed-U改善至少3个百分点且规范预测保持的预声明门：'+', '.join(f'{label}: '+('PASS'if a['candidates'][name]['conditional_macro_gate']['passed']else'FAIL')for name,label in zip(names,labels))+ '。这只决定载荷表示是否值得继续，不宣布最终求解器或架构上限；完整60分母／每组合20／配对CAD20／热点／实际成本见JSON。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import csv
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,original_canonical_GT_accuracy_measured=True,independent_newload_FEM_GT_accuracy_measured=False,conditional_derived_mixed_U_reference_used=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=list(range(3))
    raw=[100*a['candidates'][n]['conditional_mixed_U']['raw']['mean_U']for n in names];adapt=[100*a['candidates'][n]['conditional_mixed_U']['adapter']['mean_U']for n in names]
    for offsets,values,label in [(-.18,raw,'Raw model'),(.18,adapt,'Load-linear API')]:
        bars=axes[0].bar([i+offsets for i in x],values,width=.36,label=label);axes[0].bar_label(bars,fmt='%.1f',fontsize=8)
    axes[0].set_ylabel('Conditional derived mixed-load U error (%)');axes[0].legend(fontsize=8)
    for offsets,metric,label in [(-.18,'case_mean_u_relative_l2','U'),(.18,'case_mean_stress_relative_l2','Original signedS')]:
        values=[100*a['candidates'][n]['canonical_full60'][metric]for n in names];bars=axes[1].bar([i+offsets for i in x],values,width=.36,label=label);axes[1].bar_label(bars,fmt='%.1f',fontsize=8)
    axes[1].set_ylabel('Canonical actual GT error (%)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('20 iterated DEV CADs; mixed U uses conditional reference, not new independent FEM')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig);assert(out/'comparison.png').stat().st_size>0
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,new_GT_PT_arrays_GPU=0,CSV_rows=540,PNG_visual_review_pending=True,files=['BRIEF.md','metrics.json','per_case.csv','comparison.png']))
    print(dict(status='completed',actual_CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
