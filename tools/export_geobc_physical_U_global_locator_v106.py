"""Scalar-only report of bounded full-CAD location and same-field preservation."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physical_U_global_locator_v106_plan.json')
    r=d.read('notes/geobc_physical_U_global_locator_v106_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest']
    assert r['source_sha256']==d.sha('tools/export_geobc_physical_U_global_locator_v106.py')
    assert a['GT_reads']==a['raw_H5_deck_GRID_reads']==a['K_PCG_replays']==a['neural_PT_reads']==a['training_updates']==0
    if a['locator_gate_passed']:
        assert a['prediction_barrier_complete'] and a['committed_field_queries']==2304 and a['committed_located_IDs']==384
        assert a['committed_outside_decisions']==48 and a['held_numeric_reference_array_reads']==6
        assert len(a['case_records'])==3 and all(v['passed']for v in a['case_records'].values())
    o=d.path('results/geobc_physical_U_global_locator_v106');o.mkdir(exist_ok=False);rows=[]
    lines=['# 全网格连续位移定位接口','',f'实际整阶段{e["elapsed_epoch_seconds"]:.3f}秒，GPU子阶段{a["elapsed_epoch_seconds"]:.3f}秒；固定配置门：{"PASS" if a["locator_gate_passed"] else "FAIL"}。',
        '', '输入物理坐标与完整曲面TET10网格，CUDA FP64保守Bezier包围盒建立64³桶索引，再对候选单元进行有界16步Newton逆映射。定位接口不接收旧单元编号；三CAD共384定位／2304位移查询／48包围盒外拒绝的完整预测先提交，再共享读取旧数值场与编号各一次，GT／K或PCG重算／神经模型／训练均0。',
        '', '| CAD | 全网格单元 | 建索引 | 128点定位＋16外点 | 768位移查询＋分块对照 | 原场相对差 |', '|---|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        row=a['case_records'].get(cad,{})
        record=dict(CAD=cad,full_cells=row.get('index',{}).get('indexed_cells'),cell_bin_references=row.get('index',{}).get('cell_bin_references'),
            index_build_seconds=row.get('index_build_seconds'),index_persistence_seconds=row.get('index_persistence_seconds'),
            location_seconds=row.get('location_seconds'),query_and_chunk_seconds=row.get('six_load_query_and_chunk_seconds'),
            maximum_query_bucket_candidates=row.get('maximum_bucket_candidates'),maximum_inverse_candidates=row.get('maximum_inverse_candidates'),
            exact_original_owner_match=row.get('exact_original_owner_match'),field_relative=row.get('parent_field_parity',{}).get('relative'),passed=row.get('passed',False))
        rows.append(record)
        if row.get('passed'):
            lines.append(f'| {cad} | {record["full_cells"]} | {record["index_build_seconds"]:.3f}s | {record["location_seconds"]:.3f}s | {record["query_and_chunk_seconds"]:.3f}s | {record["field_relative"]:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — |')
    lines+=['','全部原单元117554／167316／150574进入索引，未使用旧32单元子网格。已测内部点均单一严格内部归属，与旧原单元编号完全一致；包围盒外点显式notfound，不能外推为空洞分类器。若候选超限、缺失或存在多个严格内部归属，固定配置返回负结果，不补换点或调容量。边界多归属采用最小原编号并保留ties，本次内部点不涉及边界归属。',
        '保持的是同一已求解近似曲面P2物理场，**不是独立离节点OptiStruct真值准确率**。16步单初值逆映射只证明这批有界测试通过，未证明任意点全局可逆、C1、跨单元或几何梯度。载荷只覆盖实际秩3规范力／矩张成空间；原signed应力契约仍未恢复。',
        '表中分别计索引、定位和实际六载荷API查询＋分块检查；文件持久化、缓存加载、峰值显存与参考检查另列JSON。首次建索引包含首次CUDA相关开销，后续CAD更快，不能把差值当几何效应。旧显式单元查询含不同参考检查，0.206–0.243秒仅历史标量，不作为纯API等成本比。',
        '已有物理基解／K源与求解成本继承，不把缓存接口延迟当新CAD冷部署：三个规范基解普通组76.70／132.39／109.19秒，两个CAD已有固定32粗预条件器构造＋求解48.75／38.11秒；未知完整冷启动组件为null。三CAD为已迭代样本，文件名隔离未证明shape-family隔离；不宣布总体或最终求解器。']
    if not a['locator_gate_passed']:lines+=['',f'固定阶段负结果：{a.get("reason",a.get("scientific_outcome"))}。完整分母384／2304／48保留，未完成条目不按子集排名。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,locator_gate_passed=a['locator_gate_passed'],
        new_offnode_GT_accuracy=False,freshdeployment_fullcold_seconds=None,signed_S_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));names=[row['CAD']for row in rows]
    for key,label in [('index_build_seconds','Index build'),('location_seconds','Location'),('query_and_chunk_seconds','768 U + chunk check')]:
        axes[0].plot(names,[row[key] if row[key] is not None else float('nan')for row in rows],'o-',label=label)
    axes[0].set_ylabel('Measured cached-stage seconds');axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    bars=axes[1].bar(names,[row['maximum_query_bucket_candidates']or 0 for row in rows]);axes[1].bar_label(bars,fmt='%d')
    axes[1].set_ylabel('Max candidate bucket / tested interior point');axes[1].text(.5,.93,'384 owners / 2304 U / 48 outside\nReference U difference = 0' if a['locator_gate_passed'] else 'Fixed complete denominators / candidate FAIL',ha='center',va='top',transform=axes[1].transAxes,fontsize=9)
    axes[1].set_ylim(0,max([row['maximum_query_bucket_candidates']or 1 for row in rows])*1.4)
    fig.suptitle('Full-CAD GPU locator / '+('PASS' if a['locator_gate_passed'] else 'FAIL')+' / independent off-node GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        locator_gate_passed=a['locator_gate_passed'],new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','per_case.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,locator_gate=a['locator_gate_passed']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
