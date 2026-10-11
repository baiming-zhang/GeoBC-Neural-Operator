"""Scalar-only one-page short-screen result; no arrays, checkpoints or inference."""
import argparse
import time
from geobc import available_cad_pilot_data as d

OWN='tools/export_geobc_three_short_screens.py'
PLAN='notes/geobc_macro_screen_readout_v2_plan.json'
REVIEW='notes/geobc_three_short_screens_exporter_review.json'


def run():
    entry=time.monotonic();marker=d.read(REVIEW);plan=d.read(PLAN)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    assert marker['scientific_digest']==plan['scientific_digest']
    folder=d.path(plan['output']);summary=d.read(folder/'summary.json');outer=d.read(folder/'external_terminal.json')
    assert summary['status']=='completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert outer['within_external_budget'] and outer['exit_code']==0 and outer['closed'] and not outer['active']
    assert all(d.exact_process_exited(outer[k]) for k in ('guard_process','child_process'))
    assert summary['execution']['scientific_digest']==outer['scientific_digest']==plan['scientific_digest']
    assert summary['committed_finite_predictions']==180 and summary['owned_GT_array_reads']==60
    assert summary['raw_GT_reads']==summary['FINAL_numeric_reads']==summary['training_updates']==summary['scaler_fits']==0
    assert all(d.sha(p)==sha for p,sha in plan['source_pins'].items())
    names=[c['name'] for c in plan['scientific']['candidates']];rows=summary['candidates']
    assert set(rows)==set(names) and len(names)==3
    labels={'FNO9_10pass_v2':'FNO','PP256_NATIVE_10pass_v2':'Native Transolver++','HASH_PN_10pass_v2':'Geometry-conditioned Hash'}
    points=[]
    for name in names:
        row=rows[name];metric=row['full60'];assert metric['all_cases_complete'] and metric['fixed_denominator']==60
        assert row['step']==1200 and row['case_exposure_count']==10
        binding=row['upstream_terminal_bindings']
        source=d.read(binding['summary']['path']); external=d.read(binding['external']['path'])
        assert d.identity(binding['summary']['path'],hashed=True)==binding['summary']
        assert source['status']=='completed' and source['training_complete'] and source['closed'] and not source['active']
        assert external['within_external_budget'] and external['exit_code']==0
        assert not source['possible_partial_optimizer_update'] and not source['optimizer_step_in_progress']
        assert source['step']==source['optimizer_step_completions']==1200
        assert all(d.exact_process_exited(external[k]) for k in ('guard_process','child_process'))
        points.append(dict(name=name,label=labels[name],U_percent=100*metric['case_mean_u_relative_l2'],
            S_percent=100*metric['case_mean_stress_relative_l2'],hot_percent=100*metric['case_mean_stress_hotspot_relative_l2'],
            joint_score=metric['selection_score'],training_stage_UTC_seconds=source['elapsed_epoch_seconds'],
            training_stage_monotonic_seconds=source['elapsed_monotonic_seconds'],training_updates_UTC_seconds=source['training_epoch_seconds'],
            inference_UTC_seconds=row['inference_epoch_seconds'],inference_monotonic_seconds=row['inference_seconds'],model_profile=row['model_profile']))
    points.sort(key=lambda p:p['joint_score']);minimum=points[0]['joint_score'];leaders=[p['name'] for p in points if p['joint_score']==minimum]
    output=d.path('results/geobc_three_short_screens');output.mkdir(parents=True,exist_ok=True)
    assert not (output/'manifest.json').exists(),'One actual export, no silent replay'
    text=['# 三方法10遍短筛','',
        '同TRAIN40 CAD×三载荷、seed2026、1200更新、仅TRAIN40拟合scaler；单方法训练上限30分钟。完整开发20 CAD×三载荷、全部原节点、signed位移／应力。已知开发池与单seed不能替代最终测试。','',
        '| 方法 | 位移误差 | 应力误差 | 训练阶段分钟 | 60例推理秒 |','|---|---:|---:|---:|---:|']
    for p in points:text.append(f"| {p['label']} | {p['U_percent']:.2f}% | {p['S_percent']:.2f}% | {p['training_stage_UTC_seconds']/60:.2f} | {p['inference_UTC_seconds']:.2f} |")
    text+=['',f"短筛联合评分最低：{', '.join(labels[n] for n in leaders)}。以.5(meanU+meanS)预设指标排序；这决定下一轮投入优先级，不是最终求解器选型。",
        '', '训练阶段时间含输入常驻、首全节点梯度验证、保存及TRAIN读出；主循环训练时间、原monotonic轴、全60／每载荷20／最差／热点／CAD配对差保留在JSON。',
        '', 'FNO和原生++约4.53M参数；Hash采用不同容量与TCNN混合精度，不称单因子或容量匹配消融。原生++使用全原节点上下文；Hash全局编码几何并保留局部BC／材料特征，尚不等于验证整体BC全局表示。',
        '', '所有180份全节点有限预测先保存并重新验证，再共享已知GT60一次；未读FINAL，未增加训练或scalerfit。连续查询接口一致性不等于独立离节点FEM精度。']
    (output/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(output/'metrics.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    for j,p in enumerate(points):ax.bar([i+(j-1)*.24 for i in range(2)],[p['U_percent'],p['S_percent']],width=.24,label=p['label'])
    ax.set_xticks([0,1],['Displacement','Signed stress']);ax.set_ylabel('Case mean relative L2 (%)')
    ax.set_title('Known DEV20 x 3 loads: 10 TRAIN passes, 1,200 updates');ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    d.write(output/'manifest.json',dict(status='completed',scientific_digest=plan['scientific_digest'],
        source_sha256=d.sha(OWN),summary_identity=d.identity(folder/'summary.json',hashed=True),
        external_identity=d.identity(folder/'external_terminal.json',hashed=True),full_prediction_denominator=180,
        full_case_denominator=60,points=points,eligible_best=leaders,final_solver_selected=False,
        actual_export_seconds=time.monotonic()-entry,files=['BRIEF.md','metrics.json','comparison.png']))
    print({'status':'completed','eligible_best':leaders,'actual_CPU_export_seconds':time.monotonic()-entry})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
