"""Scalar report of a genuinely new source-to-response campaign; no body replay."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physical_U_new_geometry_deployment_v111_plan.json')
    review=d.read('notes/geobc_physical_U_new_geometry_deployment_v111_exporter_root_review.json')
    assert review['passed'] and review['whole_source_reviewed'] and review['conditional_actual_export_released']
    assert review['SCI']==p['scientific_digest'] and review['source_sha256']==d.sha('tools/export_geobc_physical_U_new_geometry_v111.py')
    assert a['GT_reads']==a['raw_CSV_H5_nodal_target_reads']==a['neural_PT_reads']==a['training_updates']==a['scaler_fits']==a['FINAL_reads']==0
    if a['deployment_gate_passed']:
        assert a['field_barrier_complete'] and a['numeric_gate_passed'] and a['API_barrier_complete']
        assert a['committed_finite_endpoints']==18 and a['committed_U_queries']==24 and a['committed_owner_points']==8 and a['committed_outside_points']==2
        assert a['loader_calls']==a['loader_geometry_body_reads']==2 and a['loader_basis_body_reads']==12 and a['direct_reference_API_calls']==6
        assert a['H5_input_reads']==a['GRID_map_reads']==a['deck_parses']==a['deck_header_reads']==2
        assert all(v['API_passed']for v in a['case_records'].values())
    o=d.path('results/geobc_physical_U_new_geometry_deployment_v111');o.mkdir(exist_ok=False);(o/'packages').mkdir()
    lines=['# 新几何从源输入到六维位移响应的完整启动','',
        f'固定两CAD，部署门：{"PASS"if a["deployment_gate_passed"]else"FAIL"}。整阶段外UTC{e["elapsed_epoch_seconds"]:.3f}秒／monotonic{e["elapsed_monotonic_seconds"]:.3f}秒；原始时轴差异保留，不缩放。',
        '', '这两CAD没有复用旧K、粗矩阵或物理基解。各读取一次原H5 vertices／cells、deck／header及已有CSV对齐GRID映射，新建一次K／diag／8守卫和固定32步masked Galerkin粗预条件。三个规范载荷、三个新增参考方向、三个六维混合载荷均独立零初值FP64 PCG，三lane各自alpha／beta／真残差停止。',
        '', '| CAD | 原节点／单元 | source＋K秒 | 粗构造秒 | 源到六基解秒 | 独立混合验证秒 | 清单加载秒 | 查询／拒绝／分块秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    numeric=a.get('numeric_same_equation_metrics',{});endpoints=a.get('endpoint_records',{});rows=[]
    for cad in p['scientific']['selection']['CADs']:
        row=a['case_records'].get(cad,{})
        if row.get('API_passed'):
            geom=p['scientific']['cases'][cad]['expected_geometry']
            lines.append(f'| {cad} | {geom["nodes"]}／{geom["cells"]} | {row["source_operator_wall_seconds"]:.3f} | {row["coarse_setup_seconds"]:.3f} | {row["source_to_six_basis_ready_seconds"]:.3f} | {row["groups"]["independent_mixed"]["wall_seconds"]:.3f} | {row["loader_seconds"]:.3f} | {row["query_reject_chunk_seconds"]:.3f} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — | — | — |')
        for role,names in [('canonical_basis',['ver','hor','tor']),('added_basis',['Fy','Mx','My']),('independent_mixed',['mix6_0','mix6_1','mix6_2'])]:
            for name in names:
                key=cad+'/'+role+'/'+name;v=endpoints.get(key,{})
                rows.append(dict(CAD=cad,role=role,load=name,numerical_state=v.get('numerical_state'),true_relative_residual=v.get('true_relative_residual'),
                    iterations=v.get('iterations'),all_original_nodes=v.get('all_original_nodes'),constraint_reaction_norm=v.get('constraint_reaction_norm')))
    same_max=max((v['relative']for v in numeric.values()if v['relative']is not None),default=None)
    lines+=['', f'完整18场先有限hash／equilibrium／Cartesian提交；六基解组合对三个独立混合解的同方程最大相对差：{same_max if same_max is not None else "未完整测得"}。每个数值终态均保留，预算未收敛／不支持输入不改样本或容差，完整两CAD／18分母。',
        '每份新清单在真实字段完成后由独立根核验其source／单位／REFC／原节点／字段来源与digest，运行程序不自批。原绝对UTC／mono截止在根审等待期间继续计时；加载使用根审给出的expected digest，既有110工厂／108物理API／106全CAD定位／109无ID接口保持。',
        f'本轮根审等待实测：{a.get("independent_root_wait_seconds", "未执行")}秒。两次清单加载各一份新own geometry NPZ＋六份新基解body；18场屏障读取与加载器12份基解读取单独计数、单独收费，未把全阶段科学body误称只读取一次。',
        '固定新几何点4／CAD、三六维mix：24位移查询、8owner、2外点明确整批拒绝先全部有限hash／body提交，再做6次显式单元同物理API对照。夹具ID仅生成XYZ，不传入loader／locator／响应接口；分块与原输入／基解字节保持，无重复旧耦合梯度测试。',
        '**没有读取GT或原CSV标签，未验证这两CAD相对原OptiStruct规范GT的位移准确率。**三维规范GT的旧20CAD准确率只继承历史，不移称本次新几何GT证据；独立混合解是相同近似P2方程的数值参考，不是原OptiStruct新load／offnodeGT。',
        '表中“源到六基解”含每CADsource／K／粗资格／两组基解求解和保存；程序启动import等共享成本仍在整阶段，额外独立混合验证、根审等待、清单重新加载、API验证分别列出。真实cache-miss输入到可用响应的成本与验证额外成本分清，不相加并行lane时间；未知完整工程冷部署成本null，不把缓存秒数等同新几何冷解。',
        '固定源geometry／material／zeroSPC／RBE参考点六维力矩，近似curvedP2／四Q／weighted-LS RBE3与未知DEBUG／PSOLID控制保留。两CAD在v93已揭盲，不是blind／无偏／生成shape-family隔离证明；定位有界，不证明任意曲面全局唯一逆／C1／shape-gradient。signedS仍未解决，完整U／S求解器未选定，也非与点云NN等预算排名。']
    if not a['deployment_gate_passed']:lines+=['',f'固定配置终态：{a.get("reason",a.get("scientific_outcome"))}。完整2CAD／18数值场／24查询／8owner／2拒绝分母保留，缺失量为null，不按成功子集排名。']
    if a['deployment_gate_passed']:
        proposal=d.read(d.path(p['output'])/'package_proposal.json')
        for cad,pkg in proposal['packages'].items():d.write(o/'packages'/f'{cad}.json',pkg)
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'endpoints.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,deployment_gate_passed=a['deployment_gate_passed'],
        fresh_engineering_fullcold_seconds=None,new_OptiStruct_GT_accuracy=False,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));cads=p['scientific']['selection']['CADs']
    if a['deployment_gate_passed']:
        six=[a['case_records'][cad]['source_to_six_basis_ready_seconds']for cad in cads]
        extra=[a['case_records'][cad]['groups']['independent_mixed']['wall_seconds']for cad in cads]
        axes[0].bar(cads,six,label='Source -> six basis (incl. K/coarse)');axes[0].bar(cads,extra,bottom=six,label='Extra independent mixed validation')
        axes[0].set_ylabel('Actual per-case seconds, parallel group wall');axes[0].legend(fontsize=7)
        values=[v['true_relative_residual']for v in endpoints.values()]
        axes[1].plot(range(1,len(values)+1),values,'o',markersize=4);axes[1].axhline(1e-8,color='r',linestyle='--',label='True residual 1e-8')
        axes[1].set_yscale('log');axes[1].set_xlabel('All 18 independent zero-start fields');axes[1].set_ylabel('True relative residual');axes[1].legend(fontsize=7)
    else:
        for ax in axes:ax.text(.5,.5,'Fixed campaign did not pass\nFull denominators retained',ha='center',va='center',transform=ax.transAxes)
    fig.suptitle('New geometry source -> physical U response / '+('PASS'if a['deployment_gate_passed']else'FAIL')+' / original GT unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        deployment_gate_passed=a['deployment_gate_passed'],CSV_rows=len(rows),new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,
        files=['BRIEF.md','endpoints.csv','metrics.json','comparison.png'],metadata_packages_present=a['deployment_gate_passed']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,deployment_gate=a['deployment_gate_passed'],CSV_rows=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
