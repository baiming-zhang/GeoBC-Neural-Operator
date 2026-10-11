"""Terminal scalar-only occupied-token control report. No PT/GT/array/GPU I/O."""
import argparse
import time

OWN = 'tools/export_geobc_occupied_ts_v27.py'
PLAN = 'notes/geobc_occupied_ts_readout_v27_plan.json'
REVIEW = 'notes/geobc_occupied_ts_v27_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.occupied_ts_readout_contract_v27 import approved_contract, check_identity, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    plan, _, _, _, qualified = approved_contract(PLAN)
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output']); summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['closed'] and not external['active'] and external['exit_code'] == 0 and external['within_external_budget']
    assert not external['external_timeout'] and all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    for row in summary['candidates'].values():
        complete_physical(row['full60'], plan['scientific']['case_keys'])
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    names = ('R64_FNO_W16_1500', 'R16_TRANSOLVER_1500', 'R16_OCCUPIED_TRANSOLVER_1500')
    labels = ('FNO R64 width16', 'Transolver dense R16', 'Transolver occupied R16')
    points = []
    for name, label in zip(names, labels):
        row = summary['candidates'][name]
        upstream = d.read(check_identity(row['upstream_terminal_bindings']['summary']))
        arm = upstream['methods'][row['model_profile']['method']]
        tm = arm['endpoints']['1500']['full_TRAIN']
        complete_physical(tm, qualified[names[2]]['plan']['scientific']['TRAIN_keys'])
        points.append(dict(name=name, label=label, TRAIN={f: tm[f] for f in fields}, DEV={f: row['full60'][f] for f in fields},
            training_epoch_seconds=arm['training_epoch_seconds'], training_monotonic_seconds=arm['training_monotonic_seconds'],
            model_profile=arm['model_profile'], inference_epoch_seconds=row['inference_epoch_seconds'], inference_monotonic_seconds=row['inference_seconds']))
    base, dense, control = points
    delta = {f: control['DEV'][f]-dense['DEV'][f] for f in fields}
    train_delta = {f: control['TRAIN'][f]-dense['TRAIN'][f] for f in fields}
    score = min(row['full60']['selection_score'] for row in summary['candidates'].values())
    best = sorted(name for name, row in summary['candidates'].items() if row['full60']['selection_score'] == score)
    assert best == summary['eligible_best']
    upstream = qualified[names[2]]; train = upstream['summary']; arm = upstream['arm']
    assert arm['actual_initial_full_state_equal_dense_originalTS']
    counts = train['TRAIN_occupied_token_counts']; assert len(counts) == 300 and all(0 < n <= 4096 for n in counts.values())
    fraction = sum(counts.values())/(300*4096)
    assert fraction == train['occupied_token_fraction_mean']
    out = d.path('results/geobc_occupied_ts_v27'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# Transolver：空网格token控制', '',
        '**原版DeepJEB，同100训练CAD×三载荷／5遍，knownDEV20个新CAD×三载荷，全部原节点。原版TS完整初始权重、参数、头、损失和病例顺序一致。**', '',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |', '|---|---:|---:|---:|---:|']
    for point in points:
        m = point['DEV']; lines.append(f"| {point['label']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    lines += ['', f"排空token减原denseTS：DEV位移{delta[fields[0]]*100:+.2f}、应力{delta[fields[1]]*100:+.2f}、热点{delta[fields[2]]*100:+.2f}个百分点；TRAIN位移{train_delta[fields[0]]*100:+.2f}、应力{train_delta[fields[1]]*100:+.2f}个百分点。完整七候选联合最佳：{'、'.join(best)}。", '',
        f"TRAIN300输入中的R16占据token比例均值{fraction*100:.2f}%（最少{min(counts.values())}、最多{max(counts.values())}个／4096），其余为空。排除空token与其latent设零是耦合的路由改动，不能将结果当作纯attention掩码因果证明，也不能仅凭空token很多宣称它是误差原因。", '',
        f"控制训练整轮{upstream['external']['elapsed_epoch_seconds']/60:.2f}分钟，其中纯训练{arm['training_epoch_seconds']:.2f}秒；DEV整轮{external['elapsed_epoch_seconds']:.2f}秒。UTC与monotonic原值分列，不缩放、不抹成本。", '',
        '固定原版TS1,882,172参数，全部初始state SHA与denseTS相等；所有占据token一同进入原官方算子，不抽样，空latent精确零，沿用旧block-center trilinear query与native25头。真实全300GPU缓存、最大325387节点encoder/operator/head及联合leaf有限正梯度、四RNG恢复先于Adam。', '',
        '同seed2026／1500均衡更新／5遍／Adam .001fused／最高FP32-TF32关闭／旧TRAIN40-only scaler／30.5GB SI。signed U(mm)/S(MPa)与CSV GRID对齐H5 vertices保留；物理原节点损失与物理指标一致。', '',
        '新60份finite全N预测保存重验后才共享已知DEV60份GT一次；六个旧候选仅复用完整60例标量，无baseline推理或数组重播。rawCSV／FINAL／scalerfit0。knownDEV已用于迭代、单seed、可用官方TRAIN来源，不是新盲测，也不能宣称架构上限。', '',
        '完整七行、每载荷20、配对20CAD、最坏／热点、TRAIN300、容量／显存／token数／资格与成本见metrics.json。若无明显改善停止此路线，下一阶段需解决宏观训练或条件表示问题，不默认40遍或24000步。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(points=points, full_DEV_candidates=summary['candidates'], full_TRAIN_control=train['methods'],
        paired_CAD20=summary['paired_CAD20'], DEV_gates=summary['DEV_gates'], control_minus_denseTS=delta,
        TRAIN_control_minus_denseTS=train_delta, occupied_counts=counts, occupied_fraction=fraction,
        best=best, cache_receipt=train['cache_receipt'], training_campaign=upstream['external'], readout_campaign=external,
        peak_memory={k: train[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for axis, split in zip(axes, ('TRAIN', 'DEV')):
        for i, point in enumerate(points):
            axis.bar([x+(i-1)*.23 for x in range(3)], [100*point[split][f] for f in fields], width=.22,
                     color=('#245d9c', '#c88320', '#37a28c')[i], label=point['label'])
        axis.set_xticks([0, 1, 2], ['Displacement', 'Signed stress', 'Hotspot'])
        axis.set_title('TRAIN100 x 3' if split == 'TRAIN' else 'Known DEV20 x 3'); axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)'); fig.suptitle('Original Transolver / same initial state and 5 passes / dense vs occupied tokens')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md', 'metrics.json', 'comparison.png'))
    d.write(out/'manifest.json', dict(status='completed', SCI=plan['scientific_digest'], source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json', hashed=True), external_identity=d.identity(folder/'external_terminal.json', hashed=True),
        actual_CPU_seconds=time.monotonic()-began, numerical_array_reads=0, PT_body_reads=0, GPU_calls=0,
        actual_visual_accepted=False, final_solver_selected=False, files=['BRIEF.md', 'metrics.json', 'comparison.png']))
    print(dict(status='completed', best=best, CPU_seconds=time.monotonic()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: run()
