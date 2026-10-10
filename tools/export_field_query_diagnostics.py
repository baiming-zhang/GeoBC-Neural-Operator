"""CPU scalar-only report; never reload models, predictions or ground truth."""
import argparse
import time
from geobc import available_cad_pilot_data as d


def run():
    started = time.monotonic()
    paths = ['runs/geobc_exploration/continuous_query_qualification_v1',
             'runs/geobc_exploration/FP32_query_ablation_v1']
    summaries = []
    identities = []
    for folder in paths:
        external = d.read(folder + '/external_terminal.json')
        summary = d.read(folder + '/summary.json')
        assert external['closed'] and not external['active'] and external['exit_code'] == 0
        assert external['within_external_budget'] and summary['status'] == 'completed'
        assert summary['closed'] and not summary['active']
        assert all(d.exact_process_exited(external[key]) for key in ('child_process', 'guard_process'))
        assert all(summary['receipt'][key] == 0 for key in
                   ('raw_CSV_GT_reads', 'owned_GT_array_reads', 'training_updates', 'scaler_fits'))
        summaries.append(summary)
        identities.append({'summary': d.identity(folder + '/summary.json'),
                           'external': d.identity(folder + '/external_terminal.json'),
                           'exact_child_and_guard_exited': True})
    native, fp32 = summaries
    assert len(native['cases']) == 27 and len(fp32['cases']) == 9
    groups = {}
    for key, row in native['cases'].items():
        groups.setdefault(key.split(':')[0], []).append(row)
    groups['FNO_grid25_FP32_head'] = list(fp32['cases'].values())
    rows = []
    for name, values in groups.items():
        row = {'candidate': name, 'cases': len(values),
               'max_native_replay_relative_L2': max(v['native_encoded_replay_relative_L2'] for v in values),
               'fullN_query_seconds_sum': sum(v['native_query_seconds'] for v in values)}
        if 'chunk_max_encoded_difference' in values[0]:
            row.update(chunk_max_encoded_difference=max(v['chunk_max_encoded_difference'] for v in values),
                       permutation_max_encoded_difference=max(v['permutation_max_encoded_difference'] for v in values),
                       physical_gradient_FD_max_relative_error=max(max(v['physical_gradient_centered_FD_relative_errors_by_channel']) for v in values),
                       physical_gradient_FD_max_by_channel=[max(v['physical_gradient_centered_FD_relative_errors_by_channel'][c] for v in values) for c in range(4)])
        else:
            row.update(other_seed_max_encoded_relative_L2=max(max(v['other_seed_relative_L2']) for v in values),
                       query_subset_max_encoded_relative_L2=max(v['native_vs_query_subset_relative_L2'] for v in values))
        rows.append(row)
    output = 'results/geobc_field_queries'
    d.path(output).mkdir(parents=True, exist_ok=True)
    report = {'status': 'completed', 'rows': rows, 'source_identities': identities,
              'native_external_seconds': d.read(paths[0]+'/external_terminal.json')['elapsed_epoch_seconds'],
              'FP32_external_seconds': d.read(paths[1]+'/external_terminal.json')['elapsed_epoch_seconds'],
              'GT_reads': 0, 'optimizer_updates': 0,
              'limitations': ['3 CADs x 3 loads, 8 interior TET10 points/case; no whole-mesh locator qualification',
                              'No independent off-node FEM truth or measured derivative accuracy against FEM',
                              'FP32 head changes predictions; original 60-case accuracy remains the original TF32 evaluation',
                              '++ query subset changes the context; it is a sensitivity test, not equivalent arbitrary querying']}
    fno = [r for r in rows if 'physical_gradient_FD_max_relative_error' in r]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    labels = ['New FNO / TF32', 'Old FNO', 'New FNO / FP32']
    assert len(fno) == 3
    for ax, field, title in zip(axes, ['chunk_max_encoded_difference', 'physical_gradient_FD_max_relative_error'],
                               ['Chunk sensitivity (encoded max abs)', 'AD vs finite difference (relative)']):
        vals = [r[field] for r in fno]
        ax.bar(labels, [max(x, 1e-9) for x in vals], color=['#dc6f51', '#3578ad', '#389969'])
        ax.set_yscale('log'); ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        for i, v in enumerate(vals):
            ax.annotate(f'{v:.3g}', (i, max(v,1e-9)), xytext=(0,4), textcoords='offset points', ha='center', fontsize=9)
        ax.set_ylim(1e-9, max(vals)*10)
    fig.savefig(d.path(output+'/query_precision.png'), dpi=160); plt.close(fig)
    text = '''# 连续查询：当前结论

**FNO的连续查询路线可用；新查询头需要FP32，Transolver++仍需设计固定几何上下文的查询接口。**

固定3个CAD×3载荷、原4800检查点，不训练、不读取标签。原三模型诊断25.73秒，独立FP32头对照9.60秒。

| 检查 | 旧FNO | 新FNO原TF32头 | 新FNO改FP32头 |
|---|---:|---:|---:|
| 查询分块最大编码差 | 0 | 6.56e-4 | 4.77e-7 |
| 物理梯度AD/有限差分最大相对差 | 0.98% | 471% | 0.66% |

相同权重、输入和缓存上下文下，只改查询头精度即可消除明显不一致，支持TF32查询数学是直接原因。FP32与原节点TF32输出的相对L2差最多0.0286%；它不是对原20CAD应力准确性的重新评估。

++同推理种子能精确重放，换种子后编码场相差6.8%–9.0%。只保留64查询节点还会改变全局上下文，不能当作独立连续位置查询。

这里验证的是函数重放、稳定性和数值微分一致性。每病例只有8个TET10内部点，没有独立离节点FEM真值或全几何单元定位验收。下一步统一表示、容量和查询头比较FNO与++，分清算子和原配置的贡献。
'''
    d.path(output+'/BRIEF.md').write_text(text, encoding='utf-8')
    report['CPU_export_seconds'] = time.monotonic()-started
    d.write(output+'/summary.json', report)
    print({'status': 'completed', 'CPU_seconds': report['CPU_export_seconds'], 'rows': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: run()
