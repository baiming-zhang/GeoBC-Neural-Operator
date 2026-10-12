"""Scalar original-U accuracy report; six fixed cases, no label replay."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physical_U_reserve_accuracy_v112_plan.json')
    r=d.read('notes/geobc_physical_U_reserve_accuracy_v112_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest'] and r['source_sha256']==d.sha('tools/export_geobc_physical_U_reserve_v112.py')
    assert a['prediction_barrier_complete'] and a['completed_fields']==6 and a['CSV_target_parses']==2 and a['original_U_case_reads']==6
    assert a['original_S_values_selected']==a['raw_H5_deck_reads']==a['K_PCG_replays']==a['neural_PT_reads']==a['GPU_execution']==a['training_updates']==0
    o=d.path('results/geobc_physical_U_reserve_accuracy_v112');o.mkdir(exist_ok=False);rows=[]
    lines=['# 两份新物理几何的原规范位移准确率','',
        f'完整2 CAD×3载荷，原U准确率门：{"PASS"if a["original_U_accuracy_gate_passed"]else"FAIL"}。外UTC{e["elapsed_epoch_seconds"]:.3f}秒／mono{e["elapsed_monotonic_seconds"]:.3f}秒，CPU读出，不重跑K／PCG／接口。',
        '', '| CAD | load | 原节点 | U相对L2 % | RMS mm | max mm | 真残差 |','|---|---|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        for load in p['scientific']['selection']['loads']:
            v=a['metrics'][cad+'/'+load];rows.append(v)
            relative=f'{100*v["U_relative_L2"]:.6f}'if v['U_relative_L2']is not None else'null'
            lines.append(f'| {cad} | {load} | {v["all_original_nodes"]} | {relative} | {v["U_RMS_mm"]:.6g} | {v["U_max_mm"]:.6g} | {v["true_relative_residual"]:.4g} |')
    mean=a['complete6_mean_U_relative_L2'];worst=a['complete6_max_U_relative_L2']
    lines+=['',f'六场完整均值{100*mean:.6f}%、最差{100*worst:.6f}%。'if mean is not None and worst is not None else'完整均值／最差为null，不按非零子集求均值。',
        '先对v111已保存的六份规范FP64位移严格hash／body／原节点／独立零初值／真残差1e-8作完整Cartesian屏障，再两原CSV各hash一次、解析一次，仅选择GRID、XYZ和ver／hor／tor的9个原signed U(mm)列。既有H5行到CSV行排列和GRID逐行一致，物理XYZ误差≤0.0002mm；不H5-only、不改原符号／尺度、不读S值。',
        '此处是真正原DeepJEB规范载荷U对照，不是同解恒等，也不是线性派生的新载荷GT。原CSV文本值以FP64读取，物理预测保持FP64；没有新标签数组写出、scaler拟合或训练。两reserve在v93已揭盲，不因本结果调参或换CAD，完整六场各1%门。',
        'v111新geometry源到六基解的实测77.940／63.438秒及独立混合验证、根审等待／清单加载等费用单列继承；本CPU阶段只做准确率闭合，不把物理解算成本隐藏进缓存查询。先前20knownDEV规范U完整均值0.015885%、最差0.026214%为旧冻结结果，非重读或盲测。',
        '**范围仍是近似curvedP2／四Q／weighted-LS RBE3，原DEBUG／PSOLID与精确配置未知。**这六场精度不证明原OptiStruct新load／offnodeGT、任意traction或支持条件、C1、生成shape-family隔离、总体blind结论或神经架构等预算上限。signedS没有解决，完整U／S求解器未选定。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'canonical_U.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,inherited_actual_deployment_cost=p['scientific']['inherited_actual_deployment_cost'],new_signedS=False,new_OptiStruct_load_offnode_GT_accuracy=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,3.5));labels=[v['CAD']+'\n'+v['load']for v in rows]
    if all(v['U_relative_L2']is not None for v in rows):
        bars=ax.bar(labels,[100*v['U_relative_L2']for v in rows]);ax.bar_label(bars,fmt='%.5f%%',fontsize=8)
    ax.set_ylabel('Original canonical U relative L2 (%)');ax.set_title('2 previously revealed reserves / all 6 original-node U fields / '+('PASS'if a['original_U_accuracy_gate_passed']else'FAIL'))
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,original_U_accuracy_gate_passed=a['original_U_accuracy_gate_passed'],CSV_rows=6,new_GT_arrays_PT_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','canonical_U.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,original_U_accuracy_gate=a['original_U_accuracy_gate_passed']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
