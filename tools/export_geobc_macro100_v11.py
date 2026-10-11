"""One terminal scalar-only report for the fresh100CAD five-pass pair."""
import argparse
import time

OWN='tools/export_geobc_macro100_v11.py'
PLAN='notes/geobc_macro100_readout_v11_plan.json'
REVIEW='notes/geobc_macro100_v11_exporter_root_review.json'


def run():
    started=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.macro100_readout_contract_v11 import approved_contract,complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    plan,_,_,_,upstreams=approved_contract(PLAN);science=plan['scientific']
    assert marker['scientific_digest']==plan['scientific_digest']
    folder=d.path(plan['output']);summary=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json')
    assert summary['status']=='completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['exit_code']==0 and external['within_external_budget'] and external['closed'] and not external['active']
    assert external['scientific_digest']==summary['execution']['scientific_digest']==plan['scientific_digest']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    assert summary['committed_finite_predictions']==120 and summary['owned_GT_array_reads']==summary['owned_GT_read_attempts']==60
    assert summary['raw_GT_reads']==summary['FINAL_numeric_reads']==summary['training_updates']==summary['scaler_fits']==0
    points=[]
    for candidate in science['candidates']:
        name=candidate['name'];row=summary['candidates'][name];metric=row['full60'];complete_physical(metric,science['case_keys'])
        upstream=upstreams[name]
        points.append(dict(name=name,U_percent=100*metric['case_mean_u_relative_l2'],S_percent=100*metric['case_mean_stress_relative_l2'],hot_percent=100*metric['case_mean_stress_hotspot_relative_l2'],
            parameter_count=row['model_profile']['parameter_count'],training_seconds=upstream['summary']['training_epoch_seconds'],whole_training_stage_seconds=upstream['external']['elapsed_epoch_seconds'],inference_seconds=row['inference_epoch_seconds']))
    out=d.path('results/geobc_macro100_short5_v11')
    assert not out.exists(),'No silent report replay'
    out.mkdir(parents=True)
    lines=['# 100CAD／5遍：Transolver++与FNO短筛','',
        '**同100个训练CAD、1500次完整病例更新，开发60例全原节点物理误差；只用于初筛，不是新FINAL或架构上限。**','',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 训练分钟 |','|---|---:|---:|---:|---:|']
    for x in points:lines.append(f"| {x['name']} | {x['U_percent']:.2f}% | {x['S_percent']:.2f}% | {x['hot_percent']:.2f}% | {x['training_seconds']/60:.2f} |")
    lines+=['',f"120份完整finite预测先提交／重验，再共享已知开发标签60次，完整评估{external['elapsed_epoch_seconds']:.2f}秒。训练分钟含训练期检查点；整阶段另含缓存／最大节点梯度检查／全TRAIN读出，详细成本保留在JSON。",
        '', '相同TRAIN100×三载荷、seed2026、病例顺序、oldTRAIN40-only scaler、完整节点encoded4病例均值MSE和Adam配方，均fresh初始化。native++与FNO R64表示／算子容量不同，不是纯架构因果对照。',
        '', '历史FNO100／40遍在同knownDEV的U/S为23.17%/50.85%，仅作成熟端点背景，不能用于本次5遍同曝光排名。各载荷20例、20CAD配对差、热点／最差和参数／成本均在metrics.json。',
        '', 'signed U(mm)/S(MPa)按原CSV GRID→H5 vertices对齐，不造符号、不引用撤回H5-only结果。knownDEV来自可用官方TRAIN池，单seed；已揭盲FINAL未读取或用于本次选型。连续接口资格不等于独立离节点FEM精度。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,candidates=summary['candidates'],paired_CAD20=summary['paired_CAD20'],eligible_best=summary['eligible_best'],stage_seconds=external['elapsed_epoch_seconds']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(7.2,3.8))
    for j,x in enumerate(points):axis.bar([i+(j-.5)*.3 for i in range(3)],[x['U_percent'],x['S_percent'],x['hot_percent']],width=.3,label=x['name'])
    axis.set_xticks([0,1,2],['Displacement','Signed stress','Stress hotspot']);axis.set_ylabel('Case mean relative L2 (%)')
    axis.set_title('100 TRAIN CAD / 5 passes / known DEV20 x 3');axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/name).is_file() for name in ['BRIEF.md','metrics.json','comparison.png'])
    d.write(out/'manifest.json',dict(status='completed',scientific_digest=plan['scientific_digest'],exporter_source_sha256=d.sha(OWN),points=points,
        summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True),
        actual_CPU_seconds=time.monotonic()-started,rawCSV_reads=0,numerical_array_reads=0,model_inference=0,
        FINAL_reads=0,final_solver_selected=False,files=['BRIEF.md','metrics.json','comparison.png']))
    print(dict(status='completed',points=points,CPU_seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
