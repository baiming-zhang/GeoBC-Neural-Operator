"""Scalar/loss-only TRAIN report after genuine four-arm recovery terminal."""
import argparse
import csv
import json
import time

OWN = 'tools/export_geobc_solver_seeds_training_v82.py'
REVIEW = 'notes/geobc_solver_seeds_training_v82_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2',
          'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.solver_seed_recovery_contract_v82 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical, check_identity
    p, a, e, binding = terminal(PLAN); s = p['scientific']; keys = s['TRAIN_keys']
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN) and marker['SCI'] == p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and a['matched_body_initial_bytes_verified']
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 24000
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts'] == a['cache_receipt']['physical_TRAIN_target_reads'] == 300
    assert a['total_across_attempts_physical_TRAIN_reads'] == 600 and a['resident_physical_target_reuses'] == 1200
    assert a['cache_receipt']['encoded_TRAIN_target_reads'] == a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == a['parent_PT_reads'] == 0
    assert max(e['original_campaign_epoch_seconds'], e['original_campaign_monotonic_seconds']) <= s['whole_training_campaign_budget_seconds']
    failure = d.read(check_identity(p['recovery']['failure_identity']))
    assert failure['optimizer_updates'] == 0 and failure['model_Adam_RNG_checkpoint_not_written']
    assert set(a['arms']) == {spec['name'] for spec in s['arms']}
    rows = {}
    for spec in s['arms']:
        row = a['arms'][spec['name']]
        assert row['status'] == 'completed' and row['step'] == 6000
        assert row['package'] == spec['package'] and row['seed'] == spec['seed']
        assert row['case_exposure_counts'] == {k: 20 for k in keys}
        assert row['qualification']['passed'] and row['qualification']['original_nodes'] == 325387
        assert row['qualification']['model_unchanged'] and row['qualification']['Adam_unchanged'] and row['qualification']['fourRNG_restored']
        rows[spec['name']] = row['endpoints']['20']['full_TRAIN']; complete_physical(rows[spec['name']], keys)
    for seed in s['fresh_seeds']:
        pair = [row for row in a['arms'].values() if row['seed'] == seed]
        assert len(pair) == 2 and pair[0]['initialization']['shared_body_digests'] == pair[1]['initialization']['shared_body_digests']
    out = d.path('results/geobc_solver_seeds_training_v82')
    assert not (out/'manifest.json').exists(), 'One actual immutable export'
    out.mkdir(parents=True, exist_ok=True)
    packages = ('FNO9_plain_query_MLP', 'FNO9_TCNN_query'); summaries = {}
    lines = ['# 配对种子确认：训练完成，验证待进行', '',
        '**新seed2027／2028，每方案固定100CAD×三载荷／20遍。公共骨干在每seed的两方案间初始字节一致，头的容量与精度不同；完整原节点physical U/S目标、原Adam与scaler。**', '',
        '| 方案：两个新seed平均 | 位移误差 | 应力误差 | 热点误差 | 每模型学习分钟 |',
        '|---|---:|---:|---:|---:|---:|']
    for package, label in zip(packages, ('FNO', 'FNO + 条件Hash')):
        names = [spec['name'] for spec in s['arms'] if spec['package'] == package]
        means = {f: sum(rows[n][f] for n in names)/2 for f in FIELDS}
        cost = sum(a['arms'][n]['training_epoch_seconds'] for n in names)/120
        summaries[package] = dict(mean=means, ranges={f: [min(rows[n][f] for n in names), max(rows[n][f] for n in names)] for f in FIELDS}, new_seed_mean_training_minutes=cost)
        lines.append(f'| {label} | {100*means[FIELDS[0]]:.2f}% | {100*means[FIELDS[1]]:.2f}% | {100*means[FIELDS[2]]:.2f}% | {cost:.2f} |')
    lines += ['', '**这些是TRAIN拟合，不能据此选择新CAD求解器。** 所有四个端点均进入预声明独立开发集预测先提交的比较，随后连同2026冻结标量比较三seed；不筛掉训练差的seed、不选最好seed、不自动加遍数。', '',
        f'恢复执行阶段{e["elapsed_epoch_seconds"]/60:.2f}分钟；从原任务启动到实际结束{e["original_campaign_epoch_seconds"]/60:.2f}分钟，包含原失败、定位／恢复等待、二次缓存、全部模型与读出。epoch与monotonic分别保留，不缩放修饰。', '',
        f'原元数据字段错误在0更新时退出，完整{failure["elapsed_epoch_seconds"]:.2f}秒与300份标签读取保留。没有原模型／Adam／RNG检查点，按同构造与seed重建fresh0，未谎称恢复不存在的状态。全局原绝对截止不变；首恢复arm还保守扣除整次失败耗时。恢复再读300标签，跨两attempt共600，原CSV／DEV／FINAL／编码标签／父PT与scaler拟合均0。', '',
        '每模型6000更新，四模型合计24000更新。最大的真实325387节点全参数／两leaf梯度和64query链通过，资格保留model／Adam／四RNG；各模型的6000端点、完整案例、梯度、初值hash、成本、失效历史在JSON／CSV，检查点体积文件留本地。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        original_failure=failure, package_TRAIN_summaries=summaries, knownDEV_completed=False, final_solver_selected=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file); writer.writerow(['arm', 'package', 'seed', 'case', 'U_relative_L2', 'signed_S_relative_L2', 'hotspot_relative_L2'])
        for spec in s['arms']:
            for key in keys:
                z = rows[spec['name']]['per_case'][key]
                writer.writerow([spec['name'], spec['package'], spec['seed'], key, z['u_relative_l2'], z['stress_relative_l2'], z['stress_hotspot_relative_l2']])
    loss = [json.loads(line) for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss) == 24000
    for spec in s['arms']:
        part = [z for z in loss if z['arm'] == spec['name']]
        assert [z['step'] for z in part] == list(range(1, 6001)) and all(z['seed'] == spec['seed'] for z in part)
    d.write(out/'loss.json', loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.7))
    for spec in s['arms']:
        part = [z for z in loss if z['arm'] == spec['name']]
        values = [sum(z['physical_relative_squared'] for z in part[i:i+300])/300 for i in range(0, 6000, 300)]
        ax.plot(range(1, 21), values, label=spec['name'])
    ax.set_yscale('log'); ax.set_xlabel('Full TRAIN300 exposures'); ax.set_ylabel('Mean joint squared relative loss')
    ax.grid(alpha=.2); ax.legend(); ax.set_title('Paired independent seeds / TRAIN only, knownDEV readout required')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    files = ['BRIEF.md', 'metrics.json', 'per_case.csv', 'loss.json', 'comparison.png']
    assert all((out/f).is_file() for f in files)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN),
        terminal_binding=binding, files=files, actual_CPU_seconds=time.perf_counter()-began, GT_reads=0, PT_body_reads=0, GPU_calls=0))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, knownDEV_completed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","scientific_bodies":0,"GPU":0}')
