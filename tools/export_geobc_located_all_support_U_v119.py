"""One scalar-only report for the genuine continuous multiple-boundary API."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d
def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_located_all_support_U_response_v119_plan.json');r=d.read('notes/geobc_located_all_support_U_response_v119_exporter_root_review.json')
    assert r['passed']and r['conditional_actual_export_released']and r['actual_conditions_verified_at_utc']and r['actual_export']==0
    assert r['SCI']==p['scientific_digest']and r['source_sha256']==d.sha('tools/export_geobc_located_all_support_U_v119.py')
    assert a['GT_reads']==a['K_PCG_replays']==a['raw_H5_deck_GRID_reads']==a['neural_PT_reads']==a['training_updates']==0
    gate=a['coupled_API_gate_passed']
    if gate:assert a['prediction_barrier_complete']and a['committed_U_queries']==48 and a['committed_positive_owner_points']==16 and a['committed_outside_points']==2 and a['actual_coupled_gradient_points']==4
    o=d.path('results/geobc_located_all_support_U_response_v119');o.mkdir(exist_ok=False);rows=[];cads=p['scientific']['selection']['CADs']
    lines=['# 连续XYZ＋六维力矩＋四支撑平移的物理U接口','',f'固定两CAD／48查询／16owners／2盒外拒绝／4实际耦合梯度点：{"PASS"if gate else"FAIL"}。外UTC {e["elapsed_epoch_seconds"]:.6f}秒／mono {e["elapsed_monotonic_seconds"]:.6f}秒。','',
        '| CAD | 构造／读取秒 | 全CAD索引秒 | 查询＋拒绝＋chunk秒 | 最大同场相对差 | 坐标／力／矩／支撑梯度相对差 |','|---|---:|---:|---:|---:|---|']
    checks=['samefield0','samefield1','samefield2','chunk','owner','XYZ_gradient','force_N_gradient','moment_Nmm_gradient','support12_mm_gradient']
    for cad in cads:
        rec=a['case_records'].get(cad,{})
        if rec.get('samefield_metrics'):
            grads=[rec[k]['relative']for k in ('coupled_XYZ_gradient','coupled_force_N_gradient','coupled_moment_Nmm_gradient','coupled_support12_mm_gradient')]
            lines.append(f'| {cad} | {rec["cache_and_constructor_seconds"]:.4f} | {rec["index_build_seconds"]:.4f} | {rec["actual_query_reject_chunk_seconds"]:.4f} | {max(v["relative"]for v in rec["samefield_metrics"].values()):.3g} | {grads} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — |')
        for name in checks:
            if name.startswith('samefield'):metric=rec.get('samefield_metrics',{}).get(name[-1],{})
            elif name=='chunk':metric=rec.get('chunk_parity',{})
            elif name=='owner':metric=dict(passed=rec.get('exact_fixture_owner_match'),relative=None)
            else:metric=rec.get('coupled_'+name,{})
            rows.append(dict(CAD=cad,check=name,relative=metric.get('relative'),maximum_absolute=metric.get('maximum_absolute'),passed=metric.get('passed')))
    lines+=['','真实接口仅接受physical XYZ(mm)、force6(前三N／后三Nmm)及四支撑平移(mm)，不要求caller单元IDs。全CAD保守Bezier桶定位在no_grad离散分支执行，之后真实curvedP2 Newton逆映射、18物理基解插值和力／支撑系数保留autograd图。四个支撑转角仍为0。',
        '每CAD8个全原单元linspace位置以新bary [.25,.30,.20,.25]生成geometry-only XYZ，fixtureIDs只生成坐标，不传facade／locator。三固定请求共48U、16owners及2真实全Bezier盒外点先finite/hash/body/Cartesian屏障，随后8次同物理API显式单元对照。额外两个8有效＋1无效混合批次整批返回None，未预测子集。',
        '实际4点分别验证XYZ坐标、前三force N、后三moment Nmm、十二support mm梯度，全部非零有限，按各单位分别比较直接接口，不用mm／rad／N的混单位L2假精度。chunk／同场／owner与输入和API基解GPU字节保持核验。不是旧192点梯度重播。',
        '两own NPZ仅读vertices／cells，36物理force＋support基解各严格singlehashload；新索引构造、API、查询、梯度、屏障保存和退出均计入实际双时轴，首次CUDA开销不归因几何。旧源到force6、port0与剩余9基解成本分别继承；加载既有基解不是新几何全冷启动，未知完整工程cold为null。',
        '**同场值／梯度保持是接口数值证据，未测原OptiStruct新载荷、支撑、旋转或独立offnodeGT。** 近似curvedP2四Q／weightedLS RBE3与未知DEBUG／PSOLID限制保留，两CAD已揭盲，不证明C1、shape梯度、任意位置全局逆映射或工程资格；signedS仍未解决。']
    if not gate:lines+=['',f'固定完整分母保留，缺失null；终态：{a.get("reason",a.get("scientific_outcome"))}。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'checks.csv').open('w',encoding='utf-8',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,new_offnode_GT_accuracy=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3.8))
    cache=[a['case_records'].get(cad,{}).get('cache_and_constructor_seconds')for cad in cads];query=[a['case_records'].get(cad,{}).get('actual_query_reject_chunk_seconds')for cad in cads]
    if all(v is not None for v in cache+query):ax.bar(cads,cache,label='Strict basis load + API + index');ax.bar(cads,query,bottom=cache,label='Query + reject + chunk');ax.legend()
    ax.set_ylabel('Actual scoped wall seconds');ax.grid(axis='y',alpha=.2);ax.set_title('Located XYZ / force6 / support12 API '+('PASS'if gate else'FAIL')+'; new offnode GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,complete_declared_gate_passed=gate,CSV_rows=len(rows),PNG_visual_review_pending=True,new_GT_arrays_PT_GPU=0,files=['BRIEF.md','checks.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,gate=gate,CSV_rows=len(rows)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
