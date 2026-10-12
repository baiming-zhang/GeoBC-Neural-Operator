"""Scalar-only complete knownDEV physicalU coverage report, no solver replay."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physics_U_knownDEV_coverage_v107_plan.json')
    r=d.read('notes/geobc_physics_U_knownDEV_coverage_v107_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest']
    assert r['source_sha256']==d.sha('tools/export_geobc_physics_U_coverage_v107.py')
    assert a['neural_PT_reads']==a['neural_inference']==a['training_updates']==a['scaler_fits']==a['FINAL_reads']==a['raw_CSV_H5_nodal_target_reads']==0
    complete=a.get('prediction_barrier_complete',False) and a['committed_finite_endpoints']==60 and a['GT_reads']==60
    metrics=a.get('canonical_original_GT_U',{})
    if complete:assert len(metrics)==60 and len(a['case_records'])==20 and a['H5_input_reads']==17 and a['owned_array_reads']==60
    o=d.path('results/geobc_physics_U_knownDEV_coverage_v107');o.mkdir(exist_ok=False);rows=[]
    lines=['# 固定20个开发CAD的物理位移覆盖检查','',f'真实整阶段{e["elapsed_epoch_seconds"]:.2f}秒，完整覆盖门：{"PASS" if a["coverage_gate_passed"] else "FAIL"}。新求解17CAD×三规范载荷，另3CAD×三载荷复用原不可变收敛基解；全分母20CAD／60场。',
        '', '每个新增CAD真实源／几何／K与8个算子资格只构建一次；固定32步masked-Galerkin粗预条件、FP64三并行RHS各自零初值PCG，独立alpha／beta／真残差／停止。全部60场有限字节身份与Cartesian屏障后，共享原physicalGT60一次；标签只用于U指标，不构造signed应力。',
        '', '| CAD | 原三载荷最大U误差 | 最大真残差 | 本轮源＋粗构造＋三载荷 | 状态 |','|---|---:|---:|---:|---|']
    for cad in p['scientific']['selection']['CADs']:
        case=a['case_records'].get(cad,{})
        for load in ('ver','hor','tor'):
            v=metrics.get(cad+'/'+load,{})
            rows.append(dict(CAD=cad,load=load,U_relative_L2=v.get('U_relative_L2'),U_error_percent=None if v.get('U_relative_L2')is None else 100*v['U_relative_L2'],
                U_RMS_mm=v.get('U_RMS_mm'),U_max_mm=v.get('U_max_mm'),true_relative_residual=v.get('true_relative_residual'),
                original_nodes=v.get('original_nodes'),converged=v.get('converged'),U_gate_passed=v.get('U_gate_passed'),
                reused_existing_endpoint=cad in p['scientific']['selection']['reused_CADs'],new_source_coarse_group_seconds=case.get('new_source_coarse_canonical_seconds')))
        fields=[metrics.get(cad+'/'+load,{})for load in ('ver','hor','tor')]
        valid=all(v.get('U_relative_L2')is not None for v in fields)
        if valid:
            cost='旧基解复用'if case.get('reused_existing_endpoints')else f'{case["new_source_coarse_canonical_seconds"]:.2f}s'
            lines.append(f'| {cad} | {100*max(v["U_relative_L2"]for v in fields):.5f}% | {max(v["true_relative_residual"]for v in fields):.3g} | {cost} | {"PASS" if all(v["U_gate_passed"] and v["converged"]for v in fields)else "FAIL"} |')
        else:lines.append(f'| {cad} | 未完整测得 | 未完整测得 | — | 未验收 |')
    lines+=['','固定门要求全60真残差≤1e−8、原规范位移误差每场≤1%，零参考只用固定绝对误差门。任何来源不支持、非有限或缺失均保留完整60分母，不能换CAD、按子集取均值／排名或调粗步数／容差救结果。',
        '这是近似曲面P2／四点积分／weighted-LS RBE3，原DEBUG／PSOLID等控制效果未知；规范原节点U兼容性不是原OptiStruct配置精确复现，不是新的混合载荷或离节点GT。20CAD属于已迭代、可用性条件选择的knownDEV；文件名分离未证明生成shape-family隔离。',
        '当前整阶段包含17个新CAD的源读取、几何K／资格、粗构造、求解、检查点／文件保存、标签与读出；3个旧CAD基解真实历史成本另列JSON，不能把本轮复用成本称20个新几何全冷部署。三并行lane使用实际groupwall，不相加当墙钟；未知完整冷组件null。',
        '此阶段拓展物理U路线覆盖，不重新训练或推理FNO／Hash／Transolver，不宣布联合field最终求解器；原signed应力契约仍open。']
    if not complete:lines+=['',f'阶段完整覆盖未达：{a.get("reason",a.get("scientific_outcome"))}；实际已提交{a["committed_finite_endpoints"]}/60，GT实际{a["GT_reads"]}/60，不能按已有子集验收。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'per_field.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,coverage_gate_passed=a['coverage_gate_passed'],complete60=complete,
        fresh20CAD_fullcold_seconds=None,signed_S_solved=False,final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4));cads=p['scientific']['selection']['CADs'];pos=list(range(20))
    values=[max([100*metrics[cad+'/'+load]['U_relative_L2']for load in ('ver','hor','tor')])if complete and all(metrics[cad+'/'+load]['U_relative_L2']is not None for load in ('ver','hor','tor'))else float('nan')for cad in cads]
    axes[0].bar(pos,values);axes[0].set_ylabel('Max original canonical U error (%)');axes[0].axhline(1,color='red',ls='--',label='Fixed 1% gate');axes[0].legend(fontsize=8)
    costs=[a['case_records'].get(cad,{}).get('new_source_coarse_canonical_seconds',float('nan'))for cad in cads]
    axes[1].bar(pos,costs);axes[1].set_ylabel('New source + coarse + canonical group (s)')
    for ax in axes:ax.set_xticks(pos,cads,rotation=75,fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed knownDEV20 / 60 fields / '+('coverage PASS'if a['coverage_gate_passed']else'fixed coverage FAIL')+' / approximate P2 U only')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        coverage_gate_passed=a['coverage_gate_passed'],new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','per_field.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,coverage_gate=a['coverage_gate_passed']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
