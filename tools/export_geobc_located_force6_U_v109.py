"""Scalar-only reusable located six-force U interface report."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_located_force6_U_response_v109_plan.json');r=d.read('notes/geobc_located_force6_U_response_v109_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest'] and r['source_sha256']==d.sha('tools/export_geobc_located_force6_U_v109.py')
    assert a['GT_reads']==a['K_PCG_replays']==a['raw_H5_deck_GRID_reads']==a['neural_PT_reads']==a['training_updates']==0
    if a['coupled_API_gate_passed']:
        assert a['prediction_barrier_complete'] and a['committed_U_queries']==72 and a['committed_positive_owner_points']==24 and a['committed_outside_points']==48
        assert a['actual_coupled_gradient_points']==6 and a['direct_reference_API_calls']==12 and a['mixed_batch_rejection_count']==3 and a['extra_mixed_batch_point_status_count']==27
        assert all(v['passed']for v in a['case_records'].values())
    o=d.path('results/geobc_located_force6_U_response_v109');o.mkdir(exist_ok=False);rows=[]
    lines=['# 物理坐标＋六维参考力矩的统一位移接口','',f'实际整阶段{e["elapsed_epoch_seconds"]:.3f}秒，统一接口门：{"PASS"if a["coupled_API_gate_passed"]else"FAIL"}。',
        '', '调用`LocatedForce6UResponse.query(xyz_mm, force6)`，不提供单元编号；force6前三项N、后三项Nmm，U输出mm。完整曲面TET10的保守Bezier索引定位后，六个独立物理基解通过实际曲面逆映射求值。坐标与力矩保留可微图，离散定位分支固定；失败／多重内部归属的整个批次返回rejected和None，不返回剩余子集场。',
        '', '| CAD | 完整原网格单元 | 查询／拒绝／分块实际时间 | 同场最大相对差 | 坐标／六维力矩梯度相对差 |','|---|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        row=a['case_records'].get(cad,{})
        if row.get('passed'):
            same=max(v['relative']for v in row['samefield_metrics'].values())
            lines.append(f'| {cad} | {row["actual_fullN"]["actual_cells"]} | {row["actual_query_reject_chunk_seconds"]:.3f}s | {same:.3g} | {row["coupled_XYZ_gradient"]["relative"]:.3g}／{row["coupled_force6_gradient"]["relative"]:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — |')
        for name in ('chunk_parity','coupled_XYZ_gradient','coupled_force6_gradient'):
            v=row.get(name,{});rows.append(dict(CAD=cad,check=name,passed=v.get('passed'),relative=v.get('relative'),maximum_absolute=v.get('maximum_absolute')))
        for j in range(3):
            v=row.get('samefield_metrics',{}).get(str(j),{});rows.append(dict(CAD=cad,check='samefield_mix'+str(j),passed=v.get('passed'),relative=v.get('relative'),maximum_absolute=v.get('maximum_absolute')))
    lines+=['','固定新几何点8／CAD、三六维mix：72位移查询、24定位、48包围盒外点的完整有限hash／Cartesian输出先提交，再调用显式单元同物理API数值对照；GT0。另3个“8内部＋1外部”混合批次，共27状态，均明确整批拒绝，无子集输出。6个真实点验证统一接口的坐标与六个物理力矩梯度；未重复旧192点测试。',
        '数值对照证明接口耦合和值保持，**独立新载荷／离节点OptiStruct准确率未测**。原20CAD规范位移平均0.01589%、最差0.02621%只继承旧物理准确率证据；三CAD迭代、未证明生成shape-family隔离。只有固定geometry／material／零支撑／同RBE3参考点六维力矩，未知原DEBUG／PSOLID与近似P2／四Q／weighted-LS RBE3局限保留。',
        '定位为有界单初值逆映射，不保证任意曲面单元全局唯一逆，也未证明C1／跨单元／shape梯度。保守包围盒外拒绝不是孔洞分类器；边界ties保留，严格内部歧义拒绝。原signedS契约仍open，接口不生成应力标量或标签。',
        '每CAD一份own geometry NPZ与六basis字段各一次严格hash/load；全N有限、原字节与秩6资格真实通过。查询表计三mix请求＋两分块请求＋16外点和9点混合拒绝，不称单次纯API速度。缓存加载／构造／index／后置对照／梯度与实际全阶段显存、双时轴见JSON；没有新的K／PCG／标签重播，继承基解与源成本单列，完整新几何冷部署未知null。']
    if not a['coupled_API_gate_passed']:lines+=['',f'固定配置结果：{a.get("reason",a.get("scientific_outcome"))}，完整3CAD／72查询／24定位／48外点分母保留。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'checks.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,coupled_API_gate_passed=a['coupled_API_gate_passed'],new_OptiStruct_GT_accuracy=False,freshdeployment_fullcold_seconds=None,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));cads=p['scientific']['selection']['CADs']
    bars=axes[0].bar(cads,[a['case_records'].get(cad,{}).get('actual_query_reject_chunk_seconds',0)for cad in cads]);axes[0].bar_label(bars,fmt='%.3fs');axes[0].set_ylabel('Actual queries + chunk + batch rejection (s)')
    if a['coupled_API_gate_passed']:
        for name,label in [('coupled_XYZ_gradient','XYZ gradient'),('coupled_force6_gradient','Physical force6 gradient')]:
            values=[a['case_records'][cad][name]['relative']for cad in cads]
            axes[1].plot(cads,values,'o-',label=label)
    axes[1].set_ylabel('Relative difference to direct same-field API');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('physicalXYZ + reference force6 -> U(mm) / '+('PASS'if a['coupled_API_gate_passed']else'FAIL')+' / independent GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,coupled_API_gate_passed=a['coupled_API_gate_passed'],new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','checks.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,coupled_API_gate=a['coupled_API_gate_passed']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
