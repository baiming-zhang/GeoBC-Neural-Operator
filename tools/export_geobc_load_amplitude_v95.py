"""CPU scalar report for a real complete target-free response experiment."""
import argparse
import csv
import time
from geobc import available_cad_pilot_data as d


def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,binding=terminal('notes/geobc_load_amplitude_homogeneity_v95_plan.json')
    r=d.read('notes/geobc_load_amplitude_homogeneity_v95_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['SCI']==p['scientific_digest']
    assert r['source_sha256']==d.sha('tools/export_geobc_load_amplitude_v95.py')
    assert a['GT_reads']==a['CSV_H5_reads']==a['physical_encoded_target_reads']==a['training_updates']==a['scaler_fits']==0
    out=d.path('results/geobc_load_amplitude_homogeneity_v95');out.mkdir(exist_ok=False)
    names=['FNO20','FNO_HASH20','TS20'];labels=['FNO','FNO+Hash','Transolver']
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=binding,GT_accuracy_measured=False,limits=p['scientific']['limits']+[p['scientific']['scoped_physics_limit']]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.6,4.3));lines=['# 载荷幅值：位移响应一致性','','**三个冻结20遍模型／三旧TRAIN CAD×三载荷，幅值0.5／1／1.5；没有读取U／S标签，也没有训练或拟合scaler。此处是一致性偏差，不是GT预测误差。**','']
    rows=[]
    if a.get('input_negative'):
        assert not a['protocol_qualified'];lines+=['实际输入条件未通过，保留81预测／54性质的完整预定分母，不给模型排名或均值。',a['error']]
        ax.text(.5,.5,'Input-negative / no eligible model property mean',ha='center',va='center');ax.axis('off')
    else:
        assert a['protocol_qualified']and a['prediction_barrier_complete']and a['committed_finite_predictions']==81 and a['property_rows']==54
        assert a['checkpoint_hash_reads']==a['checkpoint_loads']==3 and a['input_tensor_bytes_unchanged']
        assert a['source_deck_parses']==a['source_deck_header_reads']==3 and len(a['input_witness'])==9
        lines+=['| 模型 | 平均位移响应偏差 | 最坏场偏差 |','|---|---:|---:|']
        for name,label in zip(names,labels):
            c=a['candidates'][name];q=c['properties'];assert c['strict_checkpoint_verified']and c['model_bytes_unchanged']and c['predictions']==27 and q['fixed_denominator']==18 and len(q['per_case'])==18
            mean=q['mean_relative_U_defect'];worst=q['worst_relative_U_defect']
            lines.append('| '+label+' | '+(f'{mean*100:.2f}%'if mean is not None else'未定义')+' | '+(f'{worst*100:.2f}%'if worst is not None else'未定义')+' |')
            for key,v in q['per_case'].items():rows.append(dict(candidate=name,property_key=key,**v))
        for offset,alpha in [(-.18,.5),(.18,1.5)]:
            vals=[]
            for name in names:
                values=[v['relative_U_defect']for v in a['candidates'][name]['properties']['per_case'].values()if v['alpha']==alpha]
                assert len(values)==9
                vals.append(sum(values)/9*100 if all(v is not None for v in values)else float('nan'))
            bars=ax.bar([i+offset for i in range(3)],vals,.36,label=f'Load amplitude {alpha:g}')
            ax.bar_label(bars,fmt='%.1f',fontsize=9)
        ax.axhline(1,color='black',linestyle='--',linewidth=1,label='1% diagnostic tolerance')
        ax.set_xticks(range(3),labels);ax.set_ylabel('Relative displacement homogeneity defect (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        lines+=['','完整81份原节点encoded4／physical4有限预测的Cartesian／哈希／数组屏障通过后，54项性质再比较各模型自己的幅值1位移。指标是 ||U(α)−αU(1)||／(α||U(1)||)，加上mm单位RMS／最大偏差；零范数不填0或加floor。固定1%性质容差不作为准确率／最终solver门。',
            '', '三原deck的真实STATICS／MAT1材料／零SPC、力矩向量与native25材料／力特征已核对。native7:13及grid11:17同步缩放，α1原输入字节一致；所有其他通道保持，真实全25通道重栅格化见证通过，原缓存／模型字节未改。',
            '', 'source控制的DEBUG／PSOLID／数值积分等未完整重建，忽略项在JSON永久保留；这是针对已声明线性静力结构的神经响应诊断，不宣称原FEM配置精确复现或原求解器自身违反此性质。']
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        fields=list(rows[0])if rows else['candidate','property_key','relative_U_defect'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines+=['',f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒；无GT读取、scaler拟合或参数更新。实际输入数组{a["input_array_reads"]}、deck解析{a["source_deck_parses"]}、检查点加载{a["checkpoint_loads"]}，预算／精度／纯推理时间详见JSON；缺失项不按子集均值报告。',
        '', '原signed应力输出仍保存原单位／scaler定义，但没有施加应力齐次定律、制造缩放应力参考或推测符号规则。三旧TRAIN CAD的性质检查不替代新CAD／新载荷真实精度。',
        '', '响应偏差明显时优先设计显式载荷线性的位移表示；保留骨干和原signedS分支，不微调幅值或用一致性结果宣布最终赢家。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig.suptitle('Frozen20-pass solvers; target-free load-amplitude response on TRAIN9')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig);assert (out/'comparison.png').stat().st_size>0
    d.write(out/'manifest.json',dict(status='completed_input_negative'if a.get('input_negative')else'completed',SCI=p['scientific_digest'],terminal_binding=binding,actual_CPU_seconds=time.perf_counter()-began,new_GT_PT_arrays_GPU=0,files=['BRIEF.md','metrics.json','per_case.csv','comparison.png'],PNG_visual_review_pending=True,GT_accuracy_measured=False))
    print(dict(status='exported',actual_CPU_seconds=time.perf_counter()-began,property_rows=len(rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
