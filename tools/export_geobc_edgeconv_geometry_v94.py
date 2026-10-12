"""One scalar-only CPU report after completed exact-exit TRAIN9 experiment."""
import argparse
import csv
import json
import time
from geobc import available_cad_pilot_data as d


def export():
    began=time.perf_counter()
    from geobc.edgeconv_geometry_contract_v94 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal,complete_physical
    p,a,e,binding=terminal(PLAN);s=p['scientific'];z=a['arms'][s['candidate']]
    review=d.read('notes/geobc_edgeconv_geometry_v94_exporter_root_review.json')
    assert review['passed'] and review['conditional_actual_export_released'] and review['SCI']==p['scientific_digest']
    assert review['source_sha256']==d.sha('tools/export_geobc_edgeconv_geometry_v94.py')
    assert a['training_complete'] and a['protocol_qualified'] and a['optimizer_step_attempts']==a['optimizer_step_completions']==900
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['cache_receipt']['physical_TRAIN_target_reads']==a['cache_receipt']['physical_TRAIN_target_read_attempts']==9
    assert a['cache_receipt']['encoded_TRAIN_target_reads']==a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['parent_PT_reads']==0
    assert a['resident_physical_target_reuses']==9 and z['step']==900 and z['case_exposure_counts']=={k:100 for k in s['selected_TRAIN_keys']}
    assert a['geometry_input_tensor_references_versions_unchanged'] and a['matched_body_initial_bytes_verified']
    initial=z['initialization'];ref=initial['reference']
    assert initial['parameter_count']==4562122 and ref['baseline_plain_model_digest']==s['expected_plain_model_initial_digest']
    assert ref['shared_initial_bytes_unchanged'] and initial['shared_body_digests']['operator']==s['expected_operator_initial_digest']
    q=z['qualification'];assert q['passed'] and q['original_nodes']==242936 and q['model_unchanged'] and q['Adam_unchanged'] and q['fourRNG_restored']
    assert q['query_check']['passed'] and all(x['neighbors_per_layer']==[20]*3 and x['padding_excluded'] for x in q['actual_graph_receipt'])
    old=s['baseline_scalar_endpoints']['100']['full_TRAIN9'];new=z['endpoints']['100']['full_TRAIN9']
    keys=s['selected_TRAIN_keys'];complete_physical(old,keys);complete_physical(new,keys)
    out=d.path('results/geobc_edgeconv_geometry_v94');out.mkdir(exist_ok=False);gate=a['TRAIN_gate']
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    lines=['# 动态局部图几何编码：九场短筛选','','同三个CAD×三载荷／900次完整节点更新／100遍。保留原FNO算子和查询头初始权重、原物理损失、Adam和scaler；改为固定k20／三层动态EdgeConv式全局几何编码。原点云／mask／globalBC／native25／grid保持。旧基线仅冻结标量，不重跑。','','| 几何编码 | 位移误差 | signed应力误差 | 热点误差 |','|---|---:|---:|---:|']
    for label,row in [('原PointNet',old),('动态图编码',new)]:lines.append('| '+label+' | '+' | '.join(f'{100*row[f]:.2f}%'for f in fields)+' |')
    lines+=['',f'新减旧：联合{100*gate["primary_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、位移{100*gate["U_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点；固定TRAIN门：{"PASS"if gate["passed"]else"FAIL"}。',
        '',f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒／学习{z["training_epoch_seconds"]:.2f}秒。参数4,562,122对原4,526,858，增加的是几何编码器；算子／查询头初始字节与原基线实际一致。',
        '', '真实最大242936节点的全参数／两leaf梯度、64query坐标链与chunk一致性、model／Adam0／四RNG资格已通过。9标签各读一次，900更新及末端9场常驻复用，CSV／encoded／DEV／FINAL／新预留CAD标签／父PT／scaler拟合0。',
        '', '**九TRAIN场拟合，不是泛化证明或最终选型。** 容量／编码器结构同时不同，不能归因于单一邻接操作；动态图并非FEM体积积分。Batch1／无BatchNorm／bounded32输出是本项目接口适配，不是未经修改的官方分类模型复现。',
        '', '原理参照[EdgeConv论文](https://arxiv.org/abs/1801.07829)。失败关闭固定结构，不k／宽度／seed搜索或用已揭盲场救结果；通过后才另行冻结100CAD同10遍比较，无自动20遍。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=binding,baseline_scalar_endpoints=s['baseline_scalar_endpoints'],TRAIN_gate=gate,generalization_tested=False,final_solver_selected=False))
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        writer=csv.writer(f);writer.writerow(['method','case','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for label,row in [('old_PointNet_scalar',old),('EdgeConv',new)]:
            for key,x in row['per_case'].items():writer.writerow([label,key,x['u_relative_l2'],x['stress_relative_l2'],x['stress_hotspot_relative_l2']])
    loss=[json.loads(x)for x in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss)==900 and [x['step']for x in loss]==list(range(1,901));d.write(out/'loss.json',loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.7,3.7))
    axes[0].plot(range(1,101),[sum(x['physical_relative_squared']for x in loss[i:i+9])/9 for i in range(0,900,9)])
    axes[0].set_yscale('log');axes[0].set_xlabel('Full TRAIN9 passes');axes[0].set_ylabel('Joint physical relative squared loss')
    for offset,row,label in [(-.18,old,'Original PointNet'),(.18,new,'Dynamic EdgeConv')]:
        bars=axes[1].bar([i+offset for i in range(3)],[100*row[f]for f in fields],.36,label=label);axes[1].bar_label(bars,fmt='%.1f',fontsize=8)
    axes[1].set_xticks(range(3),['U','Signed S','Hotspot']);axes[1].set_ylabel('Mean relative L2 (%)');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fixed TRAIN9/900 updates; dynamic geometry encoder, unchanged global inputs')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert (out/'comparison.png').stat().st_size>0
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha('tools/export_geobc_edgeconv_geometry_v94.py'),terminal_binding=binding,files=['BRIEF.md','metrics.json','per_case.csv','loss.json','comparison.png'],actual_CPU_seconds=time.perf_counter()-began,new_GT_PT_arrays_GPU=0,TRAIN_gate=gate,generalization_tested=False,PNG_visual_review_pending=True))
    print(dict(status='completed',actual_CPU_seconds=time.perf_counter()-began,TRAIN_gate=gate))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
