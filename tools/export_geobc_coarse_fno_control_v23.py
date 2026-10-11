"""One actual terminal scalar-only report, no numerical arrays/PT/inference."""
import argparse
import time

OWN = 'tools/export_geobc_coarse_fno_control_v23.py'
PLAN = 'notes/geobc_coarse_fno_readout_v23_plan.json'
REVIEW = 'notes/geobc_coarse_fno_v23_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.coarse_fno_readout_contract_v23 import approved_contract, check_identity, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    plan, _, _, _, qualified = approved_contract(PLAN); science = plan['scientific']
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output']); summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['closed'] and not summary['active']
    assert external['closed'] and not external['active'] and external['within_external_budget'] and external['exit_code'] == 0
    assert not external['external_timeout'] and all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == 60 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    control_name = 'R16_FNO16_CONTROL_1500'; base_name = 'FNO100_PHYSICAL_1500'
    ts_names = ('R16_TRANSOLVER_1500', 'R16_TRANSOLVER_PP_1500', 'R16_TS3_1500')
    names = [base_name, control_name] + list(ts_names)
    labels = dict(FNO100_PHYSICAL_1500='FNO R64', R16_FNO16_CONTROL_1500='FNO R16 control',
                  R16_TRANSOLVER_1500='Transolver R16', R16_TRANSOLVER_PP_1500='Transolver++ R16', R16_TS3_1500='Transolver-3 R16')
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    upstream = qualified[control_name]; train = upstream['summary']; points = []
    base_plan = d.read('notes/geobc_macro100_FNO9_physical5_v12_plan.json')
    baseline = d.read(d.path(base_plan['output'])/'summary.json')
    for name in names:
        row = summary['candidates'][name]; complete_physical(row['full60'], science['case_keys'])
        if name == base_name: arm = baseline
        elif name == control_name: arm = upstream['arm']
        else:
            old = d.read(check_identity(row['upstream_terminal_bindings']['summary']))
            arm = old['methods'][row['model_profile']['method']]
        tm = arm['endpoints']['1500']['full_TRAIN']; complete_physical(tm, upstream['plan']['scientific']['TRAIN_keys'])
        points.append(dict(name=name, label=labels[name], TRAIN={f: tm[f] for f in fields}, DEV={f: row['full60'][f] for f in fields},
            selection_score=row['full60']['selection_score'], training_epoch_seconds=arm['training_epoch_seconds'],
            training_monotonic_seconds=arm['training_monotonic_seconds'], model_profile=arm['model_profile'],
            inference_epoch_seconds=row['inference_epoch_seconds'], inference_monotonic_seconds=row['inference_seconds']))
    base, control = points[:2]; ts_score = min(p['selection_score'] for p in points[2:])
    best_ts = [p for p in points[2:] if p['selection_score'] == ts_score]
    best = [p for p in points if p['selection_score'] == min(p['selection_score'] for p in points)]
    assert sorted(p['name'] for p in best) == summary['eligible_best']
    roles = [base, control] + best_ts
    delta = {f: control['DEV'][f]-best_ts[0]['DEV'][f] for f in fields}
    versus64 = {f: control['DEV'][f]-base['DEV'][f] for f in fields}
    out = d.path('results/geobc_coarse_fno_control_v23'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# 紧凑R16表示：FNO与Transolver控制', '',
        '**同100训练CAD／5遍、knownDEV20个新CAD×三载荷、全部原节点。FNO控制与三TS共享R16输入／连续头，容量未精确匹配。**', '',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |', '|---|---:|---:|---:|---:|']
    for point in roles:
        m = point['DEV']
        lines.append(f"| {point['label']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    lines += ['', f"同R16输入下，FNO控制减最佳TS：位移{delta[fields[0]]*100:+.2f}、应力{delta[fields[1]]*100:+.2f}个百分点；减FNO64：位移{versus64[fields[0]]*100:+.2f}、应力{versus64[fields[1]]*100:+.2f}个百分点。", '',
        f"完整五方法联合U/S最佳：{'、'.join(p['label'] for p in best)}。控制TRAIN位移／应力{100*control['TRAIN'][fields[0]]:.2f}%／{100*control['TRAIN'][fields[1]]:.2f}%，TS系列TRAIN应力约94%；输入压缩本身不足以解释全部拟合差距，但算子、容量、优化与表示的相互作用尚未分开。", '',
        f"控制训练整轮{train['elapsed_epoch_seconds']/60:.2f}分钟（缓存{train['cache_setup_epoch_seconds']:.2f}秒），纯训练{control['training_epoch_seconds']:.2f}秒；新开发读出{external['elapsed_epoch_seconds']:.2f}秒。epoch与monotonic原值分别留存，未缩放对齐。", '',
        'FNO16固定width16/modes8/4blocks、4.244M参数，比紧凑++4.523M少6.17%；同XYZ-first61 pooledR16输入、PointNet1024、native25、76→128→128→4头、真实block-center trilinear查询与全节点物理损失。FNO64 width9/modes12、4.527M，分辨率／频带／宽度不同，所以不是纯分辨率或精确容量匹配消融。', '',
        '同seed2026、病例顺序／1500更新／5遍、Adam .001fused、oldTRAIN40-only scaler、最高FP32／TF32关闭；原signed U(mm)/S(MPa)与GRID对齐标签保留。未做Hash微调或自动延长40遍。', '',
        '控制60份finite encoded4／physical4保存重验后才共享knownGT60一次；四个旧候选仅复用完整60例标量，无基线推理或数组重播。rawCSV／FINAL／scalerfit均0。knownDEV已用于迭代、单seed、可用官方TRAIN来源，不能称新盲测或架构上限。', '',
        '五行完整结果／300TRAIN与60DEV／每载荷20／配对20CAD／最坏及热点／容量与显存／实际梯度资格与成本见metrics.json。连续接口一致不等于独立离节点FEM精度。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(points=points, full_DEV_candidates=summary['candidates'], full_TRAIN_control=train['methods'],
        paired_CAD20=summary['paired_CAD20'], DEV_gates=summary['DEV_gates'], control_minus_bestTS=delta, control_minus_FNO64=versus64,
        campaign_clocks=dict(epoch=train['elapsed_epoch_seconds'], monotonic=train['elapsed_monotonic_seconds']),
        cache_receipt=train['cache_receipt'], peak_memory={k: train[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')},
        best=[p['name'] for p in best], best_TS=[p['name'] for p in best_ts]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    colors = ['#245d9c', '#37a28c', '#c88320']
    for axis, split in zip(axes, ('TRAIN', 'DEV')):
        for i, point in enumerate(roles):
            axis.bar([x+(i-1)*.23 for x in range(3)], [100*point[split][f] for f in fields], width=.22, color=colors[i % len(colors)], label=point['label'])
        axis.set_xticks([0, 1, 2], ['Displacement', 'Signed stress', 'Hotspot']); axis.set_title('TRAIN100 x 3' if split == 'TRAIN' else 'Known DEV20 x 3')
        axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)'); fig.suptitle('Same R16 input/head control / 5 passes / FNO64 recipe and capacities differ')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md', 'metrics.json', 'comparison.png'))
    d.write(out/'manifest.json', dict(status='completed', SCI=plan['scientific_digest'], source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json', hashed=True), external_identity=d.identity(folder/'external_terminal.json', hashed=True),
        actual_CPU_seconds=time.monotonic()-began, numerical_array_reads=0, PT_body_reads=0, GPU_calls=0,
        actual_visual_accepted=False, final_solver_selected=False, files=['BRIEF.md', 'metrics.json', 'comparison.png']))
    print(dict(status='completed', best=[p['label'] for p in best], CPU_seconds=time.monotonic()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: run()
