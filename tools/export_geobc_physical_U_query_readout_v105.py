"""Scalar-only physical U API report after authentic CPU recovery terminal."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physical_U_query_readout_v105_plan.json');r=d.read('notes/geobc_physical_U_query_readout_v105_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest'] and r['source_sha256']==d.sha('tools/export_geobc_physical_U_query_readout_v105.py')
    assert a['API_gate_passed'] and a['protocol_qualified'] and a['prediction_barrier_complete'] and a['committed_interior_queries']==2304
    assert a['current_owned_prediction_array_reads']==9 and a['current_API_constructor_queries_gradients_replays']==a['current_source_basis_input_reads']==a['current_GPU_imports']==a['current_GPU_execution']==a['GT_reads']==0
    assert a['nodal_witness_positions']==960 and a['nodal_canonical_vectors']==2880 and a['gradient_query_points']==192 and len(a['case_records'])==3
    o=d.path('results/geobc_physical_U_query_readout_v105');o.mkdir(exist_ok=False);rows=[]
    lines=['# 连续物理位移查询与坐标／载荷梯度接口','', '三个已通过原节点U物理兼容检查的CAD：120_153／131_345／131_351。复用真实物理三规范基解，FP64曲面TET10；显式单元编号＋物理坐标，通过有界可微Newton反解局部坐标，再查询U(mm)。原力／矩矩阵实际秩3，不声称独立六自由度或全CAD无需单元定位。','',f'原实际GPU阶段{a["original_GPU_seconds"]:.3f}秒，CPU读出恢复{e["elapsed_epoch_seconds"]:.3f}秒；含整数类型适配修复等待的原任务UTC跨度{e["whole_original_campaign_epoch_seconds"]:.3f}秒，原900秒截止未重置。','', '固定2304内部查询、960节点位置／2880规范向量、192内部梯度点完整通过；GT／重新物理求解／神经模型／训练／scaler读取均0。这里验证真实场接口的一致性与可微性，**不是独立离节点FEM准确率**。','', '| CAD | 768查询与对照 | 坐标梯度相对差 | 中心差分相对差 | 秩3载荷梯度相对差 |','|---|---:|---:|---:|---:|']
    for cad,row in a['case_records'].items():
        q=row['qualification'];assert row['passed'] and all(v['passed']for v in q.values()) and row['actual_nodal_positions']==320 and row['actual_nodal_vectors']==960 and row['actual_gradient_points']==64
        lines.append(f'| {cad} | {row["six_load_query_seconds"]:.3f}s | {q["coordinate_autograd_analytic"]["relative"]:.3g} | {q["centered_finite_difference"]["relative"]:.3g} | {q["load_coefficient_gradient"]["relative"]:.3g} |')
        for name,v in q.items():rows.append(dict(CAD=cad,qualification=name,passed=v['passed'],relative=v.get('relative'),RMS_absolute=v.get('RMS_absolute'),maximum_relative=v.get('maximum_relative'),fixed_scalar_components=v.get('fixed_scalar_components')))
    lines+=['','实际字段全N分别188196／267563／239956；构造与有限值资格是真实缓存基解，非toy。每CAD固定32个原单元、4个内部重心点、6载荷组合；坐标64点autograd与原NumPyP2参考在各自坐标约定下转换到物理XYZ后比较，并验证固定步长1e−5单元尺度的中心差分。秩3系数梯度、分块一致性、真实10节点值、齐次性、叠加性、原域外载荷拒绝及输入字节保持均通过；无阈值／查询点救方案。','原场与节点值保持只是对已有数值场的接口证明；之前原规范GT误差均<0.03%属于独立物理终态证据，不移称本次离节点GT精度。需要显式有效TET10编号，未实现全CAD快速定位，未证明C1／跨单元／几何优化梯度。','查询表计入6次128点实际API调用及参考对照，不称纯端到端全冷延迟。加载／构造、全阶段资格、峰值显存及继承的原物理基解求解成本见JSON；120_153普通规范基解组76.70秒，两个新CAD132.39／109.19秒，共享K源成本分别6.97／9.02／6.86秒另列。旧额外混合求解／dtype失败／工程等待保留，完整新几何部署成本存在未知组件，不能把缓存延迟当冷启动。','原GPU已经完成全部三CAD查询／梯度／保存后，最终整数query_cell_rows的torch.int64名称未适配NumPy int64而退出。此实现遗漏保留原失败source／状态／18.82秒成本；CPU恢复仅解释已知dtype名称，9个已保存小查询文件bytes／mtime／SHA／shape／实际float64或int64／有限值严格不变，无API／GPU／基解／输入／GT重播。恢复摘要中的error_type／error为继承的原失败字段，不是本次新异常；实际本次exit0与闭合终态另列。','原型与数学SCI保持，未知DEBUG／PSOLID、近似四点积分／RBE3局限保留。只有三个已迭代CAD，非最终盲测；原signed应力出口未恢复，本接口只输出U，不伪构应力标签或宣布整体求解器完成。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        names=['CAD','qualification','passed','relative','RMS_absolute','maximum_relative','fixed_scalar_components'];w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,API_gate_passed=True,new_offnode_GT_accuracy=False,freshdeployment_fullcold_seconds=None,inherited_original_error_fields_not_new_failure=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8));cads=list(a['case_records'])
    bars=axes[0].bar(cads,[a['case_records'][cad]['six_load_query_seconds']for cad in cads]);axes[0].bar_label(bars,fmt='%.3fs');axes[0].set_ylabel('Cached 768 U queries + reference check (s)')
    for name,label in [('coordinate_autograd_analytic','XYZ autograd / analytic'),('centered_finite_difference','Fixed h XYZ finite difference'),('load_coefficient_gradient','Rank3 load gradient')]:axes[1].semilogy(cads,[a['case_records'][cad]['qualification'][name]['relative']for cad in cads],'o-',label=label)
    axes[1].set_ylabel('Relative gradient discrepancy');axes[1].legend(fontsize=7);axes[1].grid(axis='y',alpha=.2);fig.suptitle('Three real cached P2 fields / API PASS / off-node GT accuracy unmeasured');fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,API_gate_passed=True,new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','per_case.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,API_gate=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
