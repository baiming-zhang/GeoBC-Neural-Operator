"""One terminal CPU scalar/loss brief, no array/checkpoint bodies or inference."""
import argparse
import math
import time

OWN = 'tools/export_geobc_coarse_three_v21.py'
PLAN = 'notes/geobc_coarse_three_readout_v21_plan.json'
REVIEW = 'notes/geobc_coarse_three_v21_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.coarse_transolver_readout_contract_v21 import approved_contract, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    plan, _, _, _, qualified = approved_contract(PLAN); science = plan['scientific']
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output']); summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['exit_code'] == 0 and external['within_external_budget'] and external['closed'] and not external['active']
    assert not external['external_timeout'] and all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == 180 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    labels = dict(R16_TRANSOLVER_1500='Transolver R16', R16_TRANSOLVER_PP_1500='Transolver++ R16', R16_TS3_1500='Transolver-3 R16', FNO100_PHYSICAL_1500='FNO R64')
    baseline_plan = d.read('notes/geobc_macro100_FNO9_physical5_v12_plan.json')
    baseline = d.read(d.path(baseline_plan['output'])/'summary.json')
    points = []; curves = {}; train = next(iter(qualified.values()))['summary']
    for name in ['FNO100_PHYSICAL_1500'] + [c['name'] for c in science['candidates']]:
        dev = summary['candidates'][name]['full60']; complete_physical(dev, science['case_keys'])
        if name == 'FNO100_PHYSICAL_1500':
            upstream = baseline
        else:
            upstream = qualified[name]['arm']
        metric = upstream['endpoints']['1500']['full_TRAIN']
        complete_physical(metric, next(iter(qualified.values()))['plan']['scientific']['TRAIN_keys'])
        point = dict(name=name, label=labels[name], TRAIN={f: metric[f] for f in fields}, DEV={f: dev[f] for f in fields},
            selection_score=dev['selection_score'], training_epoch_seconds=upstream['training_epoch_seconds'],
            training_monotonic_seconds=upstream['training_monotonic_seconds'], model_profile=upstream['model_profile'],
            inference_epoch_seconds=summary['candidates'][name]['inference_epoch_seconds'],
            inference_monotonic_seconds=summary['candidates'][name]['inference_seconds'])
        points.append(point)
        if name != 'FNO100_PHYSICAL_1500':
            method = upstream['method']; path = d.path(next(iter(qualified.values()))['plan']['output'])/method/'loss.jsonl'
            import json
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 1500 and [row['step'] for row in rows] == list(range(1, 1501))
            assert all(math.isfinite(row['physical_relative_squared']) and row['physical_relative_squared'] >= 0 for row in rows)
            curves[name] = [sum(row['physical_relative_squared'] for row in rows[p*300:(p+1)*300])/300 for p in range(5)]
    best_score = min(point['selection_score'] for point in points)
    best = [point for point in points if point['selection_score'] == best_score]
    best_ts_score = min(point['selection_score'] for point in points[1:])
    best_ts = [point for point in points[1:] if point['selection_score'] == best_ts_score]
    assert sorted(point['name'] for point in best) == summary['eligible_best']
    base = points[0]; selected = best_ts[0]
    du = selected['DEV'][fields[0]]-base['DEV'][fields[0]]; ds = selected['DEV'][fields[1]]-base['DEV'][fields[1]]
    out = d.path('results/geobc_coarse_three_physical5_v21'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# 三版Transolver与FNO：同100CAD／5遍短筛', '',
        f"**本轮联合U/S最佳：{'、'.join(p['label'] for p in best)}；TS系列最佳：{'、'.join(p['label'] for p in best_ts)}。**", '',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |', '|---|---:|---:|---:|---:|']
    for point in points:
        m = point['DEV']
        lines.append(f"| {point['label']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    lines += ['', f"最佳TS减FNO：位移{du*100:+.2f}、应力{ds*100:+.2f}个百分点，联合差{.5*(du+ds)*100:+.2f}个百分点；正值表示误差更大。比较为完整knownDEV20CAD×三载荷、全部原节点。", '',
        f"三TS共用一次输入／原物理标签GPU缓存，准备{train['cache_setup_epoch_seconds']:.2f}秒；三方法训练、初始化验证、TRAIN300读出及清理整轮{train['elapsed_epoch_seconds']/60:.2f}分钟。新180份预测与knownGT60共享评估整轮{external['elapsed_epoch_seconds']:.2f}秒；基线只复用已有完整60例标量。表中纯训练含每遍检查点保存，不含共享缓存和最终TRAIN读出。", '',
        '三TS使用原版／++／3官方实现，4096个R16全局token来自原target-free R64 occupied-voxel池化，全部原节点仍由相同native25／PointNet／连续MLP头监督；没有独立token分块注意力。FNO使用R64。池化可能丢薄壁或稀疏边界信息，容量约1.88M／4.52M／7.68M，与FNO4.53M不同，因此不是单因素架构消融。', '',
        '原signed U(mm)/S(MPa)、同物理相对损失、seed2026、oldTRAIN40-only scaler、Adam .001fused、300病例同顺序／1500更新／5遍；最高FP32、TF32关闭。++沿用官方Gumbel随机评估，每病例固定seed；只测一个seed。', '',
        '完整180有限encoded4／physical4预测保存重验后才共享读取knownGT60一次；rawCSV、FINAL、scalerfit均0。已揭盲FINAL不参与本轮；knownDEV是可用官方TRAIN来源且已用于研究迭代，不能当新盲测。连续查询接口不证明独立离节点FEM精度。', '',
        '完整300／60、每载荷20、配对20CAD、最坏误差、容量、显存、四RNG／最大N资格验证和逐遍损失见metrics.json。预设推进门槛：联合改善至少3个百分点、任何单场恶化不超过2个百分点；通过也需另行设计5→10，未自动延长到40遍。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(points=points, full_DEV_candidates=summary['candidates'], full_TRAIN_methods=train['methods'],
        paired_CAD20=summary['paired_CAD20'], DEV_gates=summary['DEV_gates'], pass_mean_physical_relative_squared=curves,
        shared_cache_cost=dict(epoch_seconds=train['cache_setup_epoch_seconds'], monotonic_seconds=train['cache_setup_monotonic_seconds']),
        campaign_clocks=dict(epoch_seconds=train['elapsed_epoch_seconds'], monotonic_seconds=train['elapsed_monotonic_seconds']),
        peak_memory={k: train[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')},
        cache_receipt=train['cache_receipt'], physical_TRAIN_readout_reuses=train['cached_physical_TRAIN_readout_reuses'],
        best=[p['name'] for p in best], best_TS=[p['name'] for p in best_ts]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = ['#245d9c', '#37a28c', '#c88320', '#9b55a5']
    for i, point in enumerate(points):
        axes[0].bar([x+(i-1.5)*.2 for x in range(3)], [100*point['DEV'][f] for f in fields], width=.19, color=colors[i], label=point['label'])
    axes[0].set_xticks([0, 1, 2], ['Displacement', 'Signed stress', 'Hotspot']); axes[0].set_ylabel('Case mean relative L2 (%)')
    axes[0].set_title('Known DEV20 CAD x 3 loads / whole nodes'); axes[0].legend(fontsize=8)
    for i, point in enumerate(points[1:], 1):
        axes[1].plot(range(1, 6), curves[point['name']], marker='o', color=colors[i], label=point['label'])
    axes[1].set_xticks(range(1, 6)); axes[1].set_xlabel('Full passes (300 cases each)')
    axes[1].set_ylabel('Pass mean training physical relative squared loss'); axes[1].set_title('Learning curves / not DEV error'); axes[1].legend(fontsize=8)
    fig.suptitle('Same physical objective / 100 CAD / 5 passes / representation and capacity differ')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md', 'metrics.json', 'comparison.png'))
    d.write(out/'manifest.json', dict(status='completed', SCI=plan['scientific_digest'], source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json', hashed=True), external_identity=d.identity(folder/'external_terminal.json', hashed=True),
        actual_CPU_seconds=time.monotonic()-began, numerical_array_reads=0, PT_body_reads=0, GPU_calls=0,
        actual_visual_accepted=False, final_solver_selected=False, files=['BRIEF.md', 'metrics.json', 'comparison.png']))
    print(dict(status='completed', best=[p['label'] for p in best], best_TS=[p['label'] for p in best_ts], CPU_seconds=time.monotonic()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: run()
