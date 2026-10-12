"""Scalar-only sixload precision and conservative all-work cost report."""
import argparse,time,csv
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_sixload_coarse_preconditioner_v102_plan.json');s=p['scientific'];r=d.read('notes/geobc_sixload_coarse_preconditioner_v102_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest'] and r['source_sha256']==d.sha('tools/export_geobc_sixload_coarse_v102.py')
    assert a['GT_reads']==a['neural_PT_reads']==a['neural_inference']==a['training_updates']==a['scaler_fits']==a['FINAL_reads']==a['raw_H5_deck_GRID_reads']==a['geometry_operator_rebuilds']==a['inherited8fineK_guards_repeated']==0
    whole=max(e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds']);baseline=s['baseline']['exclusive_group_wall_sum_seconds'];limit=s['gate']['required_whole_new_seconds_le']
    full=a['protocol_qualified'] and a['committed_finite_endpoints']==6
    precision=full and a['precision_gate_passed'];cost=precision and whole<=limit
    o=d.path('results/geobc_sixload_coarse_preconditioner_v102');o.mkdir(exist_ok=False);rows=[]
    lines=['# 六载荷固定32步粗空间预条件器','', '固定已迭代开发CAD120_153，三个规范载荷＋三个混合载荷。每组三个载荷同时GPU计算，每个载荷独立零初值、独立PCG系数和真残差；FP64、原曲面P2／四积分点／RBE／约束／刚度矩阵保持。','',f'完整新阶段{whole:.3f}秒，对照两组实际求解{baseline:.3f}秒，减少{100*(1-whole/baseline):.2f}%。成本门需整个新阶段≤{limit:.3f}秒；精度门 '+('PASS'if precision else'FAIL')+'，联合成本门 '+('PASS'if cost else'FAIL')+'。','',f'缓存＋粗构造／资格／保存{a.get("cache_and_coarse_setup_seconds",float("nan")):.3f}秒，两组求解{a["combined_solve_seconds"]:.3f}秒，数值读出{a.get("readout_seconds",float("nan")):.3f}秒。原共享K准备6.970秒单列；本次不重建K、不读原几何或GT、不推理神经模型。']
    if full:
        assert a['prediction_barrier_complete'] and a['numerical_reference_reads']==6 and a['parallel_M_qualification']['passed'] and a['coarse_algebra_guards']['passed']
        assert a['cache_and_coarse_setup_seconds']<=120 and a['combined_solve_seconds']<=600 and a['readout_seconds']<=120
        assert set(a['same_solution_metrics'])=={'ver','hor','tor','mix0','mix1','mix2'}
        lines+=['','| 载荷 | 对同方程旧解的U相对差 | 真残差 | 迭代数 |','|---|---:|---:|---:|']
        for name,v in a['same_solution_metrics'].items():
            rows.append(dict(key=name,**v));rel=v['relative_U'];lines.append(f'| {name} | '+(f'{rel:.3g}'if rel is not None else'undefined')+f' | {v["true_relative_residual"]:.3g} | {v["iterations"]} |')
    else:lines+=['','完整六载荷未成立，保留分母6，不按子集报告精度或成本赢家。原因：'+str(a.get('reason',a.get('scientific_outcome')))]
    lines+=['','成本比较保守：新阶段包含缓存重新加载、粗网格构造、必要资格、PCG、检查点、字段保存与数值对照；旧153.73秒只含两组并行求解，不含共享K构建。这是同CAD共享K的多载荷成本，**不是fresh请求全冷启动**；不累加并行列的耗时。未知完整冷请求组件保持null。','精度对照读取六份已收敛同方程数值解；不是重新读GT。原GT兼容性仅继承v101报告：原三规范U误差均<0.03%；本阶段未新测GT精度。一个已迭代CAD，不证明总体泛化或工程资格。','粗空间为原角点注入／边中点各半／固定行mask，Ac=PtAP，固定32次零初值Jacobi，omega=1/Gamma；没有改steps、lambda、容差、精度或伪逆。全局边映射、受约束中点的非嵌套事实、两个真实向量的能量／对称／Galerkin、真实三列与逐列一致性见JSON。','近似P2／四点积分／RBE3和原DEBUG／PSOLID未知差异保留；原signed应力未构造标签／参考，未解决。未达成本门则关闭本固定32步新CAD六载荷候选，不自动调参挽救。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        names=['key','relative_U','RMS_mm','zero_reference','same_solution_pass','true_relative_residual','converged','iterations'];w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,whole_new_seconds=whole,baseline_group_wall_seconds=baseline,work_reduction_fraction=1-whole/baseline,precision_gate_passed=precision,cost_gate_passed=cost,new_GT_accuracy_measured=False,freshdeployment_cold_seconds=None))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    bars=axes[0].bar(['Block Jacobi\n2 group walls','Fixed32 coarse\nwhole new stage'],[baseline,whole],color=['#777777','#2d8c69']);axes[0].bar_label(bars,fmt='%.1fs');axes[0].axhline(limit,color='#b95757',ls='--',label='25% reduction threshold');axes[0].set_ylabel('Actual seconds');axes[0].legend(fontsize=7)
    if full:
        axes[1].bar([v['key']for v in rows],[v['true_relative_residual']for v in rows]);axes[1].axhline(1e-8,color='#b95757',ls='--');axes[1].set_ylabel('True equilibrium residual');axes[1].set_title('All six precision '+('PASS'if precision else'FAIL'))
    else:axes[1].axis('off');axes[1].text(.5,.5,'Fixed six-load negative\nNo subset precision ranking',ha='center',va='center')
    fig.suptitle('One iterated CAD / same cached K; cost '+('PASS'if cost else'FAIL'));fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed'if full else'completed_scientific_negative',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,precision_gate_passed=precision,cost_gate_passed=cost,new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','metrics.json','per_case.csv','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,whole_new_seconds=whole,precision_gate=precision,cost_gate=cost))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
