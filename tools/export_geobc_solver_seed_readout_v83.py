"""Scalar-only paired3seed knownDEV report after genuine terminal."""
import argparse
import csv
import time

OWN = 'tools/export_geobc_solver_seed_readout_v83.py'
REVIEW = 'notes/geobc_solver_seed_readout_v83_exporter_root_review.json'
PACKAGES = ('FNO9_plain_query_MLP', 'FNO9_TCNN_query')
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.solver_seed_readout_contract_v83 import PLAN, SOURCE_REVIEW
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical
    p, a, e, binding = terminal(PLAN); s = p['scientific']; keys = s['case_keys']; marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    assert marker['source_review_sha256'] == d.sha(SOURCE_REVIEW)
    assert a['protocol_qualified'] and a['prediction_barrier_complete'] and a['committed_finite_predictions'] == 240
    assert a['owned_GT_read_attempts'] == a['owned_GT_array_reads'] == 60
    assert a['checkpoint_read_attempts'] == a['checkpoint_reads'] == a['checkpoint_body_hash_reads'] == 4
    assert a['raw_GT_reads'] == a['FINAL_numeric_reads'] == a['training_updates'] == a['scaler_fits'] == 0
    names = [z['name'] for z in s['candidates']]
    assert set(a['strict_loaded_checkpoint_tensor_state_verified']) == set(names)
    assert all(a['strict_loaded_checkpoint_tensor_state_verified'].values())
    assert set(a['candidates']) == set(names+s['reference_names'])
    for row in a['candidates'].values():
        complete_physical(row['full60'], keys)
    t = a['three_seed_comparison']
    assert t['paired_seed_denominator'] == 3 and t['distinct_CAD_denominator'] == 20 and t['fixed_seed_case_denominator'] == 60
    assert set(t['seed_metrics']) == {'2026', '2027', '2028'} and t['gate']['thresholds'] == s['gate']
    assert not t['final_solver_selected'] and not t['statistical_significance_claimed'] and not t['FINAL_blind_test']
    out = d.path('results/geobc_solver_seed_readout_v83'); assert not (out/'manifest.json').exists(), 'One immutable actual export'
    out.mkdir(parents=True, exist_ok=True)
    label = {PACKAGES[0]: 'FNO', PACKAGES[1]: 'FNO + 条件Hash'}
    costs = {}
    for package in PACKAGES:
        rows = [a['candidates'][z['name']] for z in s['candidates'] if z['package'] == package]
        assert len(rows) == 2
        costs[package] = dict(new_seed_mean_training_minutes=sum(z['training_epoch_seconds'] for z in rows)/120,
            new_seed_mean_inference60_seconds=sum(z['inference_epoch_seconds'] for z in rows)/2,
            parameters=rows[0]['model_profile']['parameter_count'])
    passed = t['gate']['passed']
    lead = '条件Hash达到预声明的跨seed实际收益门。' if passed else '条件Hash未达到预声明的跨seed实际收益门，默认保留较小的普通FNO。'
    lines = ['# 三组种子的求解器比较', '', '**'+lead+'** 这是已迭代开发集上的路线选择，尚不是最终盲测或架构上限。', '',
        '| 三seed平均（最小—最大） | 位移误差 | 应力误差 | 热点误差 | 新seed学习分钟／60场推理秒 |',
        '|---|---:|---:|---:|---:|']
    for package in PACKAGES:
        vals = [f'{100*t["package_means"][package][f]:.2f}%（{100*t["package_ranges"][package][f][0]:.2f}—{100*t["package_ranges"][package][f][1]:.2f}）' for f in FIELDS]
        c = costs[package]
        lines.append('| '+label[package]+' | '+' | '.join(vals)+f' | {c["new_seed_mean_training_minutes"]:.2f}／{c["new_seed_mean_inference60_seconds"]:.2f} |')
    lines += ['', f'Hash减FNO：位移{100*t["mean_U_difference"]:+.2f}、应力{100*t["mean_S_difference"]:+.2f}、热点{100*t["mean_hotspot_difference"]:+.2f}个百分点；联合误差{100*t["mean_primary_difference"]:+.2f}个百分点，三seed中{t["paired_primary_wins"]}组联合误差领先。门要求平均联合／应力至少改善3个百分点、字段／热点劣化不超2个百分点，至少两组联合领先。', '',
        f'FNO参数{costs[PACKAGES[0]]["parameters"]:,}，Hash包{costs[PACKAGES[1]]["parameters"]:,}；头容量与精度不同，不能单独归因于Hash编码。训练／推理时间仅两个新增seed的每模型均值，2026实际续训、原失败和公共缓存等完整成本保留在JSON，表中数字不当全生命周期成本。', '',
        f'本次验证实际{e["elapsed_epoch_seconds"]:.2f}秒。四新端点240份全原节点预测全部保存并重验后，已有signed物理GT60共享读一次；2026 FNO／Hash与TS20只复用冻结标量。未读取原CSV、FINAL或重新拟合scaler。', '',
        '三组seed共用同20CAD×三载荷，不能当180个独立病例或60个独立CAD；开发集已迭代且来源受可用性影响。所有seed保留，不挑最好seed，不自动加遍数或继续Hash微调。每载荷／CAD配对差和完整案例见metrics.json与per_case.csv。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        package_costs=costs, comparison=t, final_solver_selected=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file); writer.writerow(['candidate', 'CAD', 'load', 'U_relative_L2', 'signed_S_relative_L2', 'hotspot_relative_L2'])
        for name, row in a['candidates'].items():
            for key in keys:
                z = row['full60']['per_case'][key]
                writer.writerow([name, *key.split('/'), z['u_relative_l2'], z['stress_relative_l2'], z['stress_hotspot_relative_l2']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for ax, field, title in zip(axes, FIELDS, ('Displacement U', 'Signed stress S', 'Stress hotspot')):
        for package, color in zip(PACKAGES, ('#356fbd', '#cd7435')):
            vals = [100*t['seed_metrics'][str(seed)][package][field] for seed in (2026, 2027, 2028)]
            ax.plot((2026, 2027, 2028), vals, 'o-', color=color, label='FNO' if package == PACKAGES[0] else 'FNO + TCNN')
        ax.set_xticks((2026, 2027, 2028)); ax.set_title(title); ax.set_ylabel('Mean relative L2 (%)'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Paired seeds, same knownDEV20 CAD x 3 loads; not a blind final test')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    files = ['BRIEF.md', 'metrics.json', 'per_case.csv', 'comparison.png']; assert all((out/f).is_file() for f in files)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN),
        terminal_binding=binding, files=files, actual_CPU_seconds=time.perf_counter()-began, GT_reads=0, PT_body_reads=0, GPU_calls=0))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, practical_Hash_gate=passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","GT_PT_arrays_GPU":0}')
