"""One scalar-only report of the same physical FNO trajectory at 5/10 passes."""
import argparse
import time

OWN = 'tools/export_geobc_physical10_v16.py'
PLAN = 'notes/geobc_physical10_readout_v16_plan.json'
REVIEW = 'notes/geobc_physical10_v16_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.physical10_readout_contract_v16 import approved_contract, complete_physical, check_identity
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    plan, _, _, _, qualified = approved_contract(PLAN)
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output']); science = plan['scientific']
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json')
    assert summary['status'] == 'completed' and summary['protocol_qualified'] and summary['prediction_barrier_complete']
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= science['whole_budget_seconds']
    training = qualified['FNO100_PHYSICAL_3000']; variant = training['summary']; ts = training['plan']['scientific']
    baseline = d.read(check_identity(ts['parent_summary_identity']))
    base_external = d.read(check_identity(ts['parent_external_identity']))
    assert variant['restored_model_Adam_fourRNG_cursor_exposure'] and variant['additional_optimizer_step_completions'] == 1500
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    points = []
    for step, upstream, clock in ((1500, baseline, base_external), (3000, variant, training['external'])):
        name = 'FNO100_PHYSICAL_'+str(step)
        train = upstream['endpoints'][str(step)]['full_TRAIN']; dev = summary['candidates'][name]['full60']
        complete_physical(train, ts['TRAIN_keys']); complete_physical(dev, science['case_keys'])
        cumulative = step == 3000
        points.append(dict(name=name, passes=step//300, TRAIN={f: train[f] for f in fields}, DEV={f: dev[f] for f in fields},
            trajectory_training_epoch_seconds=upstream['trajectory_training_epoch_seconds'] if cumulative else upstream['training_epoch_seconds'],
            trajectory_training_monotonic_seconds=upstream['trajectory_training_monotonic_seconds'] if cumulative else upstream['training_monotonic_seconds'],
            segment_training_epoch_seconds=upstream['training_epoch_seconds'], segment_training_monotonic_seconds=upstream['training_monotonic_seconds'],
            segment_stage_epoch_seconds=clock['elapsed_epoch_seconds'], segment_stage_monotonic_seconds=clock['elapsed_monotonic_seconds'],
            trajectory_stage_epoch_seconds=clock['elapsed_epoch_seconds']+(base_external['elapsed_epoch_seconds'] if cumulative else 0),
            trajectory_stage_monotonic_seconds=clock['elapsed_monotonic_seconds']+(base_external['elapsed_monotonic_seconds'] if cumulative else 0)))
    old, new = points
    du = new['DEV'][fields[0]]-old['DEV'][fields[0]]; ds = new['DEV'][fields[1]]-old['DEV'][fields[1]]
    dh = new['DEV'][fields[2]]-old['DEV'][fields[2]]
    gate = science['DEV_gate']; passed = .5*(du+ds) <= -gate['primary_absolute_improvement_at_least'] and max(du, ds) <= gate['maximum_either_field_absolute_deterioration']
    assert passed == summary['DEV_gate']['passed']
    out = d.path('results/geobc_physical10_v16'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# FNO100：物理损失5→10遍', '',
        '**同一模型训练轨迹；20个已知开发CAD×三载荷、全原节点物理误差。增加曝光量，不是不同求解器的同训练量比较。**', '',
        '| 总遍数 | 位移误差 | 应力误差 | 热点误差 | 累计纯训练分钟 |', '|---|---:|---:|---:|---:|']
    for point in points:
        m = point['DEV']
        lines.append(f"| {point['passes']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {point['trajectory_training_epoch_seconds']/60:.2f} |")
    lines += ['', f"10遍减5遍：位移{100*du:+.2f}、应力{100*ds:+.2f}、热点{100*dh:+.2f}个百分点；联合U/S改善{-50*(du+ds):.2f}个百分点，预设门槛{'通过' if passed else '未通过'}。应力仍高，不能称收敛或最终求解器。", '',
        f"追加5遍：纯训练{new['segment_training_epoch_seconds']:.2f}秒，完整阶段{new['segment_stage_epoch_seconds']/60:.2f}分钟，开发读出{external['elapsed_epoch_seconds']:.2f}秒。两段完整训练阶段累计{new['trajectory_stage_epoch_seconds']/60:.2f}分钟，包含缓存、真实最大全节点反传、检查点和TRAIN读出，排除阶段间分析等待；epoch与monotonic原始计时分别存档，不缩放。", '',
        f"TRAIN300位移／应力：5遍{100*old['TRAIN'][fields[0]]:.2f}%／{100*old['TRAIN'][fields[1]]:.2f}%，10遍{100*new['TRAIN'][fields[0]]:.2f}%／{100*new['TRAIN'][fields[1]]:.2f}%。", '',
        '恢复自身5遍的model、Adam、四类RNG、cursor与曝光记录，只追加1500更新到3000；每个TRAIN病例累计10遍。原损失、seed2026、架构、FP32、scaler及病例顺序不变；30.5GB预算。', '',
        '60份新全节点有限预测先保存并重验，再共享既有knownDEV物理GT60一次；5遍基线仅复用不可变标量，未重跑其推理或数组。原CSV、FINAL与scaler拟合均0。', '',
        '开发CAD与TRAIN隔离，但该开发池已用于方案迭代、不是新盲测；单seed及可用官方TRAIN来源，不是官方无偏测试。全60例、每载荷20例、配对20CAD、最差误差及完整成本见metrics.json。未测独立离节点FEM精度，不自动释放20/40遍。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(points=points, TRAIN_baseline=baseline['endpoints']['1500']['full_TRAIN'],
        TRAIN_variant=variant['endpoints']['3000']['full_TRAIN'], DEV_candidates=summary['candidates'], paired_CAD20=summary['paired_CAD20'],
        TRAIN_gate=variant['TRAIN_gate'], DEV_gate=summary['DEV_gate'], parent_cost=variant['parent_cost'],
        parent_state_restored=True, analysis_wait_included=False, peak_memory={k: variant[k] for k in ('peak_allocated_bytes', 'peak_reserved_bytes', 'peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.0), sharey=True)
    for axis, split, title in zip(axes, ('TRAIN', 'DEV'), ('TRAIN100 x 3', 'Known DEV20 x 3')):
        for j, point in enumerate(points):
            axis.bar([i+(j-.5)*.32 for i in range(3)], [100*point[split][f] for f in fields], width=.32, label=str(point['passes'])+' passes')
        axis.set_xticks([0, 1, 2], ['Displacement', 'Signed stress', 'Hotspot']); axis.set_title(title); axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)'); fig.suptitle('Same physical FNO trajectory / 5 to 10 passes')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md', 'metrics.json', 'comparison.png'))
    d.write(out/'manifest.json', dict(status='completed', SCI=plan['scientific_digest'], source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json', hashed=True), external_identity=d.identity(folder/'external_terminal.json', hashed=True),
        actual_CPU_seconds=time.monotonic()-began, numerical_array_reads=0, PT_reads=0, raw_GT_reads=0, GPU_calls=0,
        DEV_gate_passed=passed, final_solver_selected=False, files=['BRIEF.md', 'metrics.json', 'comparison.png']))
    print(dict(status='completed', DEV_gate_passed=passed, CPU_seconds=time.monotonic()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        run()
