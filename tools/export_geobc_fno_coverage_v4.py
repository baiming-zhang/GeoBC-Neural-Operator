"""CPU scalar-only coverage report; no checkpoint, GT or prediction arrays."""
import argparse
import time

OWN = 'tools/export_geobc_fno_coverage_v4.py'
PLAN = 'notes/geobc_fno_coverage_readout_v4_plan.json'
REVIEW = 'notes/geobc_fno_coverage_v4_exporter_review.json'


def run():
    entry = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.fno_coverage_readout_contract_v4 import approved_contract
    from tools.export_geobc_exposure_comparison import paired
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN)
    plan, _, manifest, _, qualified = approved_contract(PLAN)  # metadata/JSON only
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output'])
    summary = d.read(folder / 'summary.json')
    outer = d.read(folder / 'external_terminal.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified']
    assert summary['closed'] and not summary['active']
    assert outer['within_external_budget'] and outer['exit_code'] == 0
    assert outer['closed'] and not outer['active']
    assert all(d.exact_process_exited(outer[k]) for k in ('guard_process', 'child_process'))
    assert summary['execution']['scientific_digest'] == outer['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == 120 and summary['owned_GT_array_reads'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    keys = plan['scientific']['case_keys']
    reference_plan = d.read('notes/geobc_macro_warm_readout_v3_plan.json')
    assert reference_plan['scientific']['case_keys'] == keys
    assert reference_plan['scientific']['scaler'] == plan['scientific']['scaler'] == manifest['scaler']
    assert reference_plan['scientific']['GT_manifest_identity'] == plan['scientific']['GT_manifest_identity']
    reference = d.read(d.path(reference_plan['output']) / 'summary.json')
    assert reference['status'] == 'completed' and reference['protocol_qualified']
    reference_name = 'FNO9_40pass_v3'
    metrics = {reference_name: reference['candidates'][reference_name]['full60']}
    points = []
    for name, cad_count, step, passes in [(reference_name, 40, 4800, 40),
                                         ('FNO100_step4800', 100, 4800, 16),
                                         ('FNO100_step12000', 100, 12000, 40)]:
        if name != reference_name:
            row = summary['candidates'][name]
            assert row['step'] == step and row['train_CAD_count'] == cad_count and row['case_exposure_count'] == passes
            metrics[name] = row['full60']
        metric = metrics[name]
        assert metric['all_cases_complete'] and metric['fixed_denominator'] == 60
        assert set(metric['per_case']) == set(keys)
        points.append(dict(name=name, TRAIN_CAD=cad_count, updates=step, passes=passes,
                           U_percent=100 * metric['case_mean_u_relative_l2'],
                           S_percent=100 * metric['case_mean_stress_relative_l2'],
                           hot_percent=100 * metric['case_mean_stress_hotspot_relative_l2'],
                           joint_score=metric['selection_score']))
    paired_results = {name: paired(metrics[reference_name], metrics[name], keys)
                      for name in ('FNO100_step4800', 'FNO100_step12000')}
    train = qualified['FNO100_step12000']['summary']
    assert train['status'] == 'completed' and train['step'] == 12000
    cost = dict(whole_training_campaign_UTC_seconds=train['elapsed_epoch_seconds'],
                whole_training_campaign_monotonic_seconds=train['elapsed_monotonic_seconds'],
                training_UTC_seconds=train['training_epoch_seconds'],
                intermediate_readout_and_checkpoint_cost_included=True,
                lower_endpoint_isolated_whole_cost_exact=False,
                lower_endpoint_isolated_whole_cost_seconds=None,
                DEV_readout_UTC_seconds=summary['elapsed_epoch_seconds'])
    output = d.path('results/geobc_fno_coverage_v4')
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'manifest.json').exists(), 'No silent export replay'
    text = ['# FNO：40→100个训练CAD是否有效', '',
            '同seed2026、FNO9/R64/native25和old40-only scaler。完整knownDEV20 CAD×三载荷、全部原节点；不是FINAL。', '',
            '| TRAIN CAD | 更新／遍数 | 位移误差 | 应力误差 |',
            '|---|---:|---:|---:|']
    for point in points:
        text.append(f"| {point['TRAIN_CAD']} | {point['updates']}／{point['passes']} | {point['U_percent']:.2f}% | {point['S_percent']:.2f}% |")
    text += ['', f"100CAD整轮训练阶段耗时{cost['whole_training_campaign_UTC_seconds']/60:.2f}分钟，含缓存、最大全节点梯度验证、检查点及TRAIN读出；DEV读出{cost['DEV_readout_UTC_seconds']:.2f}秒。4800是同一12000整轮的中间端点，其独立完整成本未精确记录，不零填或冒用最终成本。",
             '', '4800对比固定更新数但每病例曝光减少，40遍对比固定曝光但总更新／计算增加；不能将二者都称纯数据单因素。40CAD端点来自自身1200继续，100CAD从fresh初始化；输入、架构、seed、损失及optimizer配方相同，浮点路径未证明逐位一致。',
             '', '全120份有限预测先提交／重验，再共享已知GT60一次；未读FINAL、原CSV或新拟合scaler。全60、每载荷20、热点、最差及20CAD配对差保留在JSON。此结果只决定下一阶段，不宣布最终求解器。']
    (output / 'BRIEF.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    d.write(output / 'metrics.json', dict(points=points, cost=cost, paired_CAD20=paired_results,
              reference=reference['candidates'][reference_name], candidates=summary['candidates']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for index, point in enumerate(points):
        ax.bar([j + (index-1)*.24 for j in range(2)], [point['U_percent'], point['S_percent']], width=.24,
               label=f"{point['TRAIN_CAD']} CAD / {point['updates']} updates")
    ax.set_xticks([0, 1], ['Displacement', 'Signed stress'])
    ax.set_ylabel('Case mean relative L2 (%)')
    ax.set_title('Known DEV20 x 3 loads: FNO training CAD coverage')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / 'comparison.png', dpi=160); plt.close(fig)
    d.write(output / 'manifest.json', dict(status='completed', scientific_digest=plan['scientific_digest'],
             exporter_source_sha256=d.sha(OWN), points=points, cost=cost,
             summary_identity=d.identity(folder / 'summary.json', hashed=True),
             external_identity=d.identity(folder / 'external_terminal.json', hashed=True),
             reference_summary_identity=d.identity(d.path(reference_plan['output']) / 'summary.json', hashed=True),
             actual_CPU_export_seconds=time.monotonic()-entry,
             final_solver_selected=False, files=['BRIEF.md', 'metrics.json', 'comparison.png']))
    print(dict(status='completed', points=points, actual_CPU_export_seconds=time.monotonic()-entry))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        run()
