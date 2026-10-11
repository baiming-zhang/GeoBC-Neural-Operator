"""Small scalar-only GINO decision report, no arrays/checkpoint/GT replay."""
import argparse
import math
import time

PLAN='notes/geobc_gino_knownDEV_v53_plan.json'
REVIEW='notes/geobc_gino_knownDEV_v53_exporter_source_review.json'


def export():
    from geobc import available_cad_pilot_data as d
    from geobc.gino_readout_contract_v53 import approved_contract,METHOD,complete_physical
    started=time.monotonic()
    er=d.read(REVIEW)
    assert er['passed'] and er['whole_source_reviewed'] and er['actual_export_released']
    assert all(d.sha(k)==h for k,h in er['source_pins'].items())
    p,root,m,truths,up=approved_contract(PLAN)
    out=d.path(p['output']);a=d.read(out/'summary.json');e=d.read(out/'external_terminal.json')
    assert a==d.read(out/'status.json') and a['closed'] and not a['active']
    assert e['closed'] and not e['active'] and e['actual_child_exited']
    assert a['execution']['run_process']==e['child_process']
    assert a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest']
    assert e['source_pins']==p['source_pins']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    output=d.path('results/geobc_gino_knownDEV_v53')
    assert not (output/'manifest.json').exists(), 'One export; preserve old report on repair'
    good=(a['status']=='completed' and a['protocol_qualified'] and e['exit_code']==0
        and e['within_external_budget'] and not e['external_timeout']
        and a['owned_GT_array_reads']==a['owned_GT_read_attempts']==60
        and a['committed_finite_predictions']==60 and a['prediction_barrier_complete'])
    heading='GINO 新CAD短筛选'
    rows=[];scope=a.get('comparison_scope','完整验证尚未完成；保留60例分母，不按子集排名。')
    if good:
        for name,row in a['candidates'].items():complete_physical(row['full60'],p['scientific']['case_keys'])
        g=a['candidates'][METHOD];f=a['candidates']['R64_FNO_W16_1500'];ts=a['candidates']['R16_TRANSOLVER_1500']
        for name,title,row in [(METHOD,'GINO',g),('R64_FNO_W16_1500','FNO width16／5遍',f),('R16_TRANSOLVER_1500','原版TS R16／5遍',ts)]:
            metric=row['full60']
            rows.append(dict(name=name,label=title,u_percent=100*metric['case_mean_u_relative_l2'],
                stress_percent=100*metric['case_mean_stress_relative_l2'],hotspot_percent=100*metric['case_mean_stress_hotspot_relative_l2']))
        # Metric summaries differ in some historical versions; require actual
        # finite recorded scalars and never invent a missing readout value.
        assert all(math.isfinite(row[k]) for row in rows for k in ('u_percent','stress_percent','hotspot_percent'))
        phrase=('完成固定5遍。' if a['formal5_eligible'] else
            f"训练预算到点，读取{p['scientific']['training_endpoint']['step']}步的原子检查点；属于未满5遍诊断。")
        best=min(a['candidates'],key=lambda n:a['candidates'][n]['full60']['selection_score'])
        headline=phrase+' 验证覆盖20个新CAD×三载荷、全部原节点；最终求解器尚未确定。'
        table='| 方法 | 位移误差 | 应力误差 | 热点误差 |\n|---|---:|---:|---:|\n'
        table+='\n'.join(f"| {r['label']} | {r['u_percent']:.2f}% | {r['stress_percent']:.2f}% | {r['hotspot_percent']:.2f}% |" for r in rows)
        train=up['summary']['training_epoch_seconds']/60
        text=(f'# {heading}\n\n{headline}\n\n{table}\n\n'
            f'GINO 本次训练{train:.2f}分钟，验证{e["elapsed_epoch_seconds"]:.2f}秒；两次训练前失败另计118.23秒。'
            '未满5遍结果不参与正式等曝光排名。另一个FNO width9对照、60例／每载荷20例及配对差保存在 metrics.json。\n\n'
            '原版DeepJEB，signed U(mm)/S(MPa)，CSV GRID对齐H5原节点；knownDEV已用于迭代、单seed、架构容量与后端不同。没有新FINAL或独立离节点FEM精度。\n')
    else:
        headline='GINO验证未完整通过；不提供部分案例的平均误差或胜者。'
        best=None
        text=(f'# {heading}\n\n{headline}\n\n'
            f'已提交预测{a["committed_finite_predictions"]}/60，已读验证场{a["owned_GT_array_reads"]}/60。'
            f'状态：{a["status"]}；错误：{a.get("error_type")} / {a.get("error")}。'
            f'实际验证耗时{e["elapsed_epoch_seconds"]:.2f}秒。保留已发生成本与检查点，需先处理直接技术原因。\n')
    output.mkdir(parents=True,exist_ok=True)
    (output/'BRIEF.md').write_text(text,encoding='utf-8')
    d.write(output/'metrics.json',dict(status='completed' if good else 'negative',
        formal5_eligible=a['formal5_eligible'],diagnostic_only=a['diagnostic_only'],
        candidates=a['candidates'],paired_CAD20=a.get('paired_CAD20'),scope=scope,
        formal_eligible_best=a.get('eligible_best') if good else None,
        numeric_diagnostic_lowest_score_method=best,
        full_readout_identity=d.identity(out/'summary.json',hashed=True),
        external_identity=d.identity(out/'external_terminal.json',hashed=True),
        final_solver_selected=False))
    manifest=dict(status='completed' if good else 'negative',source_plan=d.identity(PLAN,hashed=True),
        summary_identity=d.identity(out/'summary.json',hashed=True),
        external_identity=d.identity(out/'external_terminal.json',hashed=True),
        files={file:d.identity(output/file,hashed=True) for file in ('BRIEF.md','metrics.json')},
        CPU_export_seconds=time.monotonic()-started,scientific_arrays_read=0,GT_array_reads=0,
        PT_body_reads=0,GPU_execution=0,inference_replayed=False)
    d.write(output/'manifest.json',manifest)
    return dict(status=manifest['status'],report=d.relative(output/'BRIEF.md'),seconds=manifest['CPU_export_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:
        import json
        print(json.dumps(export()))
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
