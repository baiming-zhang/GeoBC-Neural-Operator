"""Scalar-only fullrank6 physical reference response report after real exit."""
import argparse,csv,time
from geobc import available_cad_pilot_data as d

def export():
    began=time.perf_counter()
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal('notes/geobc_physics_U_fullrank6_v108_plan.json');r=d.read('notes/geobc_physics_U_fullrank6_v108_exporter_root_review.json')
    assert r['passed'] and r['conditional_actual_export_released'] and r['SCI']==p['scientific_digest'] and r['source_sha256']==d.sha('tools/export_geobc_physics_U_fullrank6_v108.py')
    assert a['GT_reads']==a['raw_H5_deck_GRID_reads']==a['geometry_K_coarse_rebuilds']==a['neural_PT_reads']==a['training_updates']==0
    if a['fullrank6_gate_passed']:
        assert a['prediction_barrier_complete'] and a['API_prediction_barrier_complete'] and a['committed_finite_endpoints']==18
        assert a['old_canonical_reference_reads']==9 and a['actual_mixed_query_count']==1152 and a['actual_force6_gradient_points']==192 and all(v['passed']for v in a['case_records'].values())
    o=d.path('results/geobc_physics_U_fullrank6_v108');o.mkdir(exist_ok=False);rows=[]
    lines=['# 六维参考点力／矩的连续物理位移响应','',f'固定三个CAD；实际整阶段{e["elapsed_epoch_seconds"]:.3f}秒，六维扩展门：{"PASS" if a["fullrank6_gate_passed"]else"FAIL"}。',
        '', '原三规范方向Fz／−Fx／Mz之外，新增Fy／Mx／My三个独立方向，各通过原真实RBE3成员位置与Hbar的力、矩、虚功传递。再独立求三个固定六维混合载荷；3CAD×6新RHS＝18个FP64原节点字段均从零PCG，未用基解相加冒充独立新解。全18有限hash／equilibrium／Cartesian屏障之后才读取旧九规范U基解，GT0。',
        '', '| CAD | 新6RHS两组墙钟 | 6维矩阵条件数 | 独立混合解最大相对差 | 物理力／矩梯度相对差 |','|---|---:|---:|---:|---:|']
    for cad in p['scientific']['selection']['CADs']:
        row=a['case_records'].get(cad,{});wall=sum(v['wall_seconds']for v in row.get('groups',{}).values())
        if row.get('passed'):
            same=max(v['relative']for v in row['numeric_same_equation'].values());grad=row['physical_force6_autograd']['relative']
            lines.append(f'| {cad} | {wall:.3f}s | {row["physical_load_condition"]:.3f} | {same:.3g} | {grad:.3g} |')
        else:lines.append(f'| {cad} | 未完整验收 | — | — | — |')
        for name in p['scientific']['selection']['new_basis_names']+p['scientific']['selection']['validation_mix_names']:
            rec=a.get('endpoint_records',{}).get(cad+'/'+name,{})
            rows.append(dict(CAD=cad,RHS=name,numerical_state=rec.get('numerical_state'),true_relative_residual=rec.get('true_relative_residual'),iterations=rec.get('iterations'),all_original_nodes=rec.get('all_original_nodes'),initial_guess=rec.get('initial_guess')))
    lines+=['','新的可复用FullRank6PhysicalUQuery直接接收N／Nmm六维力矩；六基解、完整6×6载荷矩阵的真实秩与条件数实际检查。继承原FP64曲面TET10有界可微逆映射；1152个连续混合查询对独立求解混合P2场做数值对照，192个实际内部点的六个物理力／矩方向autograd与解析响应核验，分块及基解字节保持另列。未重复旧坐标梯度资格，未证明跨单元C1／shape梯度。',
        '作用点仍为同一个源RBE3参考点，geometry／material／零支撑固定；这不是任意节点载荷位置、任意约束或材料参数的已学习响应。原REFC123456／正常方程秩6为真实源证据，不把原三规范列误称为六维张成。',
        '混合对照来自同一个独立近似曲面P2／四Q／weighted-LS RBE3方程，**不是原OptiStruct六维新载荷或离节点GT**。之前20CAD原规范U精度兼容性只继承标量证据；本轮GT0，不移称新载荷准确率。原signedS契约仍open，本接口只提供物理U(mm)。',
        '缓存输入／三个算子数组／完整粗CSR与对角各一次严格hash-load，原K／diag／coarse不重建，旧8fineguards不重播。旧基解、源／K与失效历史成本完整继承；表计实际三lane groupwall，未相加lane时间，缓存六维响应延迟不当新几何全冷部署。峰值显存／全部IO／初始化／查询／保存／退出与双时轴见JSON；未知完整冷组件null。',
        '只有三个已经迭代CAD，未证明生成shape-family隔离或总体盲测，不宣称完整signed U/S求解器。固定18分母，失败缺失为null，不按子集排名、补换方向或调整预算／步数／容差。']
    if not a['fullrank6_gate_passed']:lines+=['',f'阶段结果：{a.get("reason",a.get("scientific_outcome"))}，已提交{a["committed_finite_endpoints"]}/18。']
    (o/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with(o/'per_RHS.csv').open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d.write(o/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,fullrank6_gate_passed=a['fullrank6_gate_passed'],new_OptiStruct_GT_accuracy=False,freshdeployment_fullcold_seconds=None,signedS_solved=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));cads=p['scientific']['selection']['CADs']
    bars=axes[0].bar(cads,[sum(v['wall_seconds']for v in a['case_records'].get(cad,{}).get('groups',{}).values())for cad in cads]);axes[0].bar_label(bars,fmt='%.2fs');axes[0].set_ylabel('Two independent 3-lane solve groups (s)')
    if a['fullrank6_gate_passed']:
        axes[1].semilogy(cads,[max(v['relative']for v in a['case_records'][cad]['numeric_same_equation'].values())for cad in cads],'o-',label='Independent mixed / six-basis sum')
        axes[1].semilogy(cads,[a['case_records'][cad]['physical_force6_autograd']['relative']for cad in cads],'s-',label='Force6 autograd / analytic')
    axes[1].set_ylabel('Relative numerical discrepancy');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Fixed reference-point force6 response / '+('PASS'if a['fullrank6_gate_passed']else'FAIL')+' / new OptiStruct GT unmeasured');fig.tight_layout();fig.savefig(o/'comparison.png',dpi=170);plt.close(fig);assert(o/'comparison.png').stat().st_size>0
    d.write(o/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,fullrank6_gate_passed=a['fullrank6_gate_passed'],new_GT_PT_arrays_GPU=0,PNG_visual_review_pending=True,files=['BRIEF.md','per_RHS.csv','metrics.json','comparison.png']))
    print(dict(status='CPU_export_completed',actual_CPU_seconds=time.perf_counter()-began,fullrank6_gate=a['fullrank6_gate_passed']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
