"""Terminal scalar-only occupied-token control report. No PT/GT/array/GPU I/O."""
import argparse
import time

OWN = 'tools/export_geobc_encoded_pair_v29.py'
PLAN = 'notes/geobc_encoded_pair_readout_v29_plan.json'
REVIEW = 'notes/geobc_encoded_pair_v29_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.encoded_pair_readout_contract_v29 import approved_contract, check_identity, complete_physical
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
    assert summary['committed_finite_predictions'] == 120 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    for row in summary['candidates'].values():
        complete_physical(row['full60'], plan['scientific']['case_keys'])
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    names = ('R64_FNO_W16_1500', 'R16_TRANSOLVER_1500', 'R16_TRANSOLVER_ENCODED_1500', 'R16_FNO16_CONTROL_1500', 'R16_FNO16_ENCODED_1500')
    labels = ('FNO R64 width16', 'Transolver physical loss', 'Transolver encoded MSE', 'FNO16 physical loss', 'FNO16 encoded MSE')
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
    base, dense, control, fno_physical, fno_encoded = points
    delta = {f: control['DEV'][f]-dense['DEV'][f] for f in fields}
    train_delta = {f: control['TRAIN'][f]-dense['TRAIN'][f] for f in fields}
    score = min(row['full60']['selection_score'] for row in summary['candidates'].values())
    best = sorted(name for name, row in summary['candidates'].items() if row['full60']['selection_score'] == score)
    assert best == summary['eligible_best']
    upstream = qualified[names[2]]; train = upstream['summary']
    assert all(v['actual_initial_state_equal_own_physical_control'] for v in train['methods'].values())
    fno_delta = {f: fno_encoded['DEV'][f]-fno_physical['DEV'][f] for f in fields}
    out = d.path('results/geobc_encoded_pair_v29'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# 同R16双模型：训练目标控制', '',
        '**原版DeepJEB，同100训练CAD×三载荷／5遍，knownDEV20个新CAD×三载荷，全部原节点。每种模型的完整初始权重与自己的物理损失控制一致。**', '',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |', '|---|---:|---:|---:|---:|']
    for point in points[:3]:
        m = point['DEV']; lines.append(f"| {point['label']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    lines += ['', f"TS MSE减其物理控制：DEV位移{delta[fields[0]]*100:+.2f}、应力{delta[fields[1]]*100:+.2f}、热点{delta[fields[2]]*100:+.2f}个百分点；TRAIN位移{train_delta[fields[0]]*100:+.2f}、应力{train_delta[fields[1]]*100:+.2f}个百分点。", '',
        f"FNO16物理控制U/S{fno_physical['DEV'][fields[0]]*100:.2f}%／{fno_physical['DEV'][fields[1]]*100:.2f}%，MSE控制{fno_encoded['DEV'][fields[0]]*100:.2f}%／{fno_encoded['DEV'][fields[1]]*100:.2f}%；差{fno_delta[fields[0]]*100:+.2f}／{fno_delta[fields[1]]*100:+.2f}个百分点。完整九候选联合最佳：{'、'.join(best)}。", '',
        f"新双模型训练整轮{upstream['external']['elapsed_epoch_seconds']/60:.2f}分钟，其中TS／FNO纯训练{train['methods']['transolver_encoded']['training_epoch_seconds']:.2f}／{train['methods']['FNO16_encoded']['training_epoch_seconds']:.2f}秒；DEV整轮{external['elapsed_epoch_seconds']:.2f}秒。UTC与monotonic原值分列，不缩放、不抹成本。", '',
        '只改训练目标：冻结旧TRAIN40 scaler的三位移标准化通道和一signed-asinh应力通道，完整原节点sum/(4N)，等病例权重。该目标改变条件化及U/S相对权重（三U通道、一S），不是只改一个scheduler或纯应力变换的因果消融。模型输出仍按原scaler解码，最终全部物理U(mm)/S(MPa)与热点指标；不把MSE变小等同物理误差改善。', '',
        '原版denseTS1,882,172和FNO16控制4,244,244参数，原R16输入／旧query／native25头保持；与各自物理控制全初始state SHA一致。每arm真实全300GPU缓存和最大325387节点encoder/operator/head及jointleaf正有限梯度、四RNG恢复先于Adam。CPU33337节点含569尾块的逐参数梯度对完整一次反传差约1e-9，CPU检查不是GPU资格替代。', '',
        '同seed2026／1500均衡更新／5遍／Adam .001fused／最高FP32-TF32关闭／30.5GB SI。只加载TRAIN物理目标300一次，两个物理TRAIN读出共600缓存复用；signed标签与CSV GRID对齐H5 vertices保留。', '',
        '两组共120份finite全N预测保存重验后才共享已知DEV60份GT一次；七个旧候选只复用完整60例标量，无baseline推理或数组重播。rawCSV／FINAL／scalerfit0。knownDEV已用于迭代、单seed、可用官方TRAIN来源，不是新盲测或架构上限。', '',
        '完整九行、每载荷20、配对20CAD、最坏／热点、TRAIN300、容量／显存／实际成本见metrics.json。若无明显改善停止此目标路线，不追加40遍或24000步；下一阶段针对宏观表示或物理尺度问题。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(points=points, full_DEV_candidates=summary['candidates'], full_TRAIN_control=train['methods'],
        paired_CAD20=summary['paired_CAD20'], DEV_gates=summary['DEV_gates'], control_minus_denseTS=delta,
        TRAIN_control_minus_denseTS=train_delta, FNO16_encoded_minus_physical=fno_delta,
        best=best, cache_receipt=train['cache_receipt'], training_campaign=upstream['external'], readout_campaign=external,
        peak_memory={k: train[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for axis, split in zip(axes, ('TRAIN', 'DEV')):
        for i, point in enumerate(points[1:]):
            axis.bar([x+(i-1.5)*.19 for x in range(3)], [100*point[split][f] for f in fields], width=.18,
                     color=('#c88320', '#bc3a3a', '#245d9c', '#37a28c')[i], label=point['label'])
        axis.set_xticks([0, 1, 2], ['Displacement', 'Signed stress', 'Hotspot'])
        axis.set_title('TRAIN100 x 3' if split == 'TRAIN' else 'Known DEV20 x 3'); axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)'); fig.suptitle('Same R16 and five passes / physical relative loss vs standardized MSE4')
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
