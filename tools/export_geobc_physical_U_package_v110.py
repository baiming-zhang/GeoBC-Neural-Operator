"""Scalar-only package loading evidence and metadata descriptor delivery."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physical_U_response_package_v110_plan.json')
    r=d.read('notes/geobc_physical_U_response_package_v110_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest']
    assert r['source_sha256']==d.sha('tools/export_geobc_physical_U_package_v110.py')
    assert a['GT_reads']==a['K_PCG_replays']==a['raw_H5_deck_GRID_reads']==a['neural_PT_reads']==a['training_updates']==0
    if a['package_gate_passed']:
        assert a['prediction_barrier_complete'] and a['committed_U_queries']==36 and a['committed_positive_owner_points']==12 and a['committed_outside_points']==3
        assert a['metadata_tamper_rejections']==9 and a['loader_calls']==3 and a['input_cache_reads']==3 and a['basis_array_reads']==18
        assert a['actual_coupled_gradient_points']==0 and a['direct_reference_API_calls']==9
        assert all(v['passed']for v in a['case_records'].values())
    o=d.path('results/geobc_physical_U_response_package_v110');o.mkdir(exist_ok=False);(o/'packages').mkdir();rows=[]
    obj=d.read(p['scientific']['implementation']['descriptor_identity']['path'])
    lines=['# 冻结物理位移响应包的单次加载','',f'整阶段实际{e["elapsed_epoch_seconds"]:.3f}秒，清单加载门：{"PASS"if a["package_gate_passed"]else"FAIL"}。',
        '', '`load_response(d, np, torch, descriptor, reviewed_digest, guard)`单次加载完整几何与六个收敛基解，返回可直接调用的`query(xyz_mm, force6)`对象。descriptor身份必须来自独立冻结的计划／根审，不依赖调用者清单的自称digest。',
        '', '| CAD | 完整节点／单元 | 清单加载总秒 | API构造／索引构造秒 | 查询／拒绝／分块秒 | 同场最大相对差 |',
        '|---|---:|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        row=a['case_records'].get(cad,{})
        d.write(o/'packages'/f'{cad}.json',obj['packages'][cad])
        if row.get('passed'):
            same=max(v['relative']for v in row['samefield_metrics'].values())
            lines.append(f'| {cad} | {row["actual_fullN"]["actual_allN"]}／{row["actual_fullN"]["actual_cells"]} | {row["loader_total_seconds"]:.3f} | {row["API_constructor_seconds"]:.3f}／{row["index_constructor_seconds"]:.3f} | {row["actual_query_reject_chunk_seconds"]:.3f} | {same:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — | — |')
        for name in ('chunk_parity',):
            v=row.get(name,{});rows.append(dict(CAD=cad,check=name,passed=v.get('passed'),relative=v.get('relative'),maximum_absolute=v.get('maximum_absolute')))
        for j in range(3):
            v=row.get('samefield_metrics',{}).get(str(j),{});rows.append(dict(CAD=cad,check='samefield_mix'+str(j),passed=v.get('passed'),relative=v.get('relative'),maximum_absolute=v.get('maximum_absolute')))
        for name in ('physical_U_unit','field_identity','basis_order'):
            vals=[v for v in a['metadata_tamper_fixtures']if v['CAD']==cad and v['fixture']==name]
            rows.append(dict(CAD=cad,check='tamper_'+name,passed=vals[0]['rejected']if len(vals)==1 else None,relative=None,maximum_absolute=None))
    lines+=['', '每CAD一次geometry NPZ（仅vertices／cells）与六份FP64物理位移字段各严格hash／load一次；三CAD总3＋18次科学body读取。九份仅元数据的unit／field identity／basis order篡改清单，即使重新生成自称digest，仍在科学body／构造前被原reviewed digest拒绝。原科学数组不修改、不复制到报告或私人仓库；packages目录只含带原本地路径的元数据。',
        '固定新4几何点／CAD、三六维mix，共36位移值、12owner和3明确整批外点拒绝，先完成有限hash／body／Cartesian屏障，之后9次显式单元同物理API对照。完整同场值／分块／定位保持；只继承v109真实耦合坐标／力矩导数证据，本阶段不重复梯度测试，GT0。',
        'API／index计时通过scoped profiler观察原构造函数并同步CUDA，不改数学实现。加载总时间还包含metadata校验、hash、IO和GPU staging；剩余项不是纯磁盘IO。所有import／构造／查询／保存／屏障／对照／退出计入全阶段，双时轴和显存实测见JSON。',
        '**这是已有基解的缓存加载，不是新几何全冷物理解算，也没有新的OptiStruct离节点／新载荷GT精度证据。**继承源、K与六基解成本独立列出，未知完整冷部署成本保留null；原20CAD规范U精度只继承旧证据。',
        '固定geometry／material／zeroSPC／RBE参考点六维力矩；近似curvedP2／四Q／weighted-LS RBE3与未知DEBUG／PSOLID局限保留。定位为有界单初值逆，不保证任意曲面全局唯一逆／C1／shape-gradient，文件名隔离未证明shape-family；signedS仍未提供，完整U／S求解器未选定。']
    if not a['package_gate_passed']:lines+=['',f'固定配置终态：{a.get("reason",a.get("scientific_outcome"))}，完整3CAD／36查询／12owner／3拒绝／9篡改分母保留。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'checks.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,package_gate_passed=a['package_gate_passed'],new_OptiStruct_GT_accuracy=False,freshdeployment_fullcold_seconds=None,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));cads=p['scientific']['selection']['CADs']
    vals=[a['case_records'].get(cad,{}).get('loader_total_seconds')for cad in cads]
    query=[a['case_records'].get(cad,{}).get('actual_query_reject_chunk_seconds')for cad in cads]
    if all(v is not None for v in vals):
        bars=axes[0].bar(cads,vals);axes[0].bar_label(bars,fmt='%.3fs')
    if all(v is not None for v in query):
        bars=axes[1].bar(cads,query);axes[1].bar_label(bars,fmt='%.3fs')
    axes[0].set_ylabel('Reviewed cache load + API + index (s)');axes[1].set_ylabel('Queries + chunk + outside rejection (s)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Immutable response package / '+('PASS'if a['package_gate_passed']else'FAIL')+' / new geometry cold solve unmeasured')
    fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig)
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        package_gate_passed=a['package_gate_passed'],new_GT_PT_arrays_GPU=0,CSV_rows=len(rows),PNG_visual_review_pending=True,
        files=['BRIEF.md','checks.csv','metrics.json','comparison.png','packages/120_153.json','packages/131_345.json','packages/131_351.json']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,package_gate=a['package_gate_passed'],CSV_rows=len(rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
