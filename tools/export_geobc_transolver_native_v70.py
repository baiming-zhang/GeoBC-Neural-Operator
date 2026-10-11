"""CPU scalar report after exact terminal native-TS recovery, no arrays/GT/PT."""
import argparse
import csv
import shutil
import time

OWN = 'tools/export_geobc_transolver_native_v70.py'
PLAN = 'notes/geobc_transolver_native_recovery_v70_plan.json'
REVIEW = 'notes/geobc_transolver_native_v70_exporter_root_review.json'
FAMILY = 'TS_ORIGINAL_BF16_NATIVE_HASH'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2',
          'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_native_recovery_contract_v70 import approved_contract
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical, check_identity
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN)
    approved_contract(PLAN)
    p, a, e, binding = terminal(PLAN)
    s, keys = p['scientific'], p['scientific']['selected_TRAIN_keys']
    assert marker['SCI'] == p['scientific_digest'] and a['training_complete']
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 3000
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['physical_TRAIN_attempts'] == a['physical_TRAIN_reads'] == 3
    assert a['resident_physical_target_reuses'] == 9 and a['input_array_reads'] == a['input_array_attempts'] == 12
    assert all(a[k] == 0 for k in ('encoded_TRAIN_reads','raw_GT_reads','DEV_FINAL_reads',
        'scaler_fits','parent_checkpoint_hash_reads','parent_checkpoint_loads'))
    assert max(a['learning_interval_epoch_seconds'], a['learning_interval_monotonic_seconds']) <= 900
    assert max(e['elapsed_epoch_seconds'], e['elapsed_monotonic_seconds']) <= 1200
    assert a['absolute_original_deadline_preserved'] and a['prior_metadata_failure_cost']['physical_TRAIN_reads'] == 3
    state = a['models'][FAMILY]
    assert state['case_exposure_counts'] == {k:1000 for k in keys}
    assert set(state['gradient_checks']) == set(keys) and all(x['passed'] for x in state['gradient_checks'].values())
    assert a['model_initialization'][FAMILY]['model_digest'] == s['expected_FP32_TS_initialization']['model_digest']
    _, old, olde, oldb = terminal('notes/geobc_transolver_bf16_v65_plan.json')
    hp, hasha, hashe, hashb = terminal('notes/geobc_shared_decoder_v61_plan.json')
    assert hp['scientific']['selected_TRAIN_keys'] == keys
    endpoints = {'Native TS + same decoder':a['aggregate_endpoints'][FAMILY],
                 'R64 TS + same decoder':old['aggregate_endpoints']['TS_ORIGINAL_BF16_GLOBAL_HASH'],
                 'FNO + same decoder':hasha['aggregate_endpoints']['FNO_GLOBAL_HASH']}
    for values in endpoints.values():
        assert set(values) == {'0','100','1000'}
        for z in values.values():
            complete_physical(z, keys)
    costs = {'Native TS + same decoder':a['learning_interval_epoch_seconds'],
             'R64 TS + same decoder':old['learning_interval_epoch_seconds'],
             'FNO + same decoder':hasha['models']['FNO_GLOBAL_HASH']['segment_epoch_seconds']}
    z = endpoints['Native TS + same decoder']['1000']
    gate = s['ability_gate']
    ability = (z[FIELDS[0]] <= gate['maximum_mean_u_relative_l2'] and
               z[FIELDS[1]] <= gate['maximum_mean_stress_relative_l2'] and
               z[FIELDS[2]] <= gate['maximum_mean_stress_hotspot_relative_l2'])
    plot = a['representative_field_plot']
    image = check_identity(plot['identity'])
    assert plot['metrics_all_original_nodes'] and plot['original_stress_sign_unchanged']
    out = d.path('results/geobc_transolver_native_shared3_v70')
    assert not (out/'manifest.json').exists(), 'No repeat export'
    out.mkdir(parents=True, exist_ok=True)
    text = ['# 原网格TS：与R64 TS相同连续解码器的三场拟合', '',
        '**相同三个困难TRAIN场、共享一个fresh模型、每场1000全节点曝光；同PointNet/官方原版TS/TCNN初值、精度、scaler与物理U/S联合损失。只改为原节点上下文和native latent。**', '',
        '| 方案 | 位移误差 | 应力误差 | 热点误差 | 学习区间 |',
        '|---|---:|---:|---:|---:|']
    for label, values in endpoints.items():
        row = values['1000']
        text.append(f'| {label} | {100*row[FIELDS[0]]:.2f}% | {100*row[FIELDS[1]]:.2f}% | {100*row[FIELDS[2]]:.2f}% | {costs[label]:.1f}秒 |')
    text += ['', f'预声明三场能力门通过：{ability}。这是已知场拟合，不自动释放100CAD、不选最终solver。当前native child完整{a["elapsed_epoch_seconds"]:.2f}秒；从原尝试启动计入失败与分析等待的整体{e["elapsed_epoch_seconds"]:.2f}秒，原绝对1200秒截止不重置。学习区间包含初始化、资格、读出和保存，不是纯训练。', '',
        '原69在创建模型与optimizer前因历史元数据shared_external缺失退出，0更新、3份物理标签读取和成本保留；恢复重新读取3份，两个尝试共6份读取，恢复常驻端点复用9次。原68前向/全节点梯度资格核心通过，但最后计数断言失败，原exit1与9.13秒成本保留；没有把技术失败改写为正常completed或重新跑资格。', '',
        '原版TS两种表示均9,601,208参数、同BF16算子/FP32参数Adam与返回latent、TCNN FP16；FNO条件Hash12,245,894且FP32骨干，容量/精度不等。Native attention按节点密度加权，不是体积积分；更换上下文与latent延拓，不是单独空体素因果消融。旧occupiedTS和Native++负结果仍保留。', '',
        '全原节点signed U(mm)/S(MPa)，不造符号、不裁剪指标。Native任意坐标需要显式有效TET10单元；离节点接口尚未实际资格验证，无C1、重网格或独立离节点FEM精度声明。云图仅显示样本P99范围，全部节点指标不截断。参考仅标量复用，不重训或读GT/PT。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        reference_bindings={'R64_TS':oldb,'FNO_Hash':hashb}, endpoints=endpoints,
        learning_interval_cost_seconds=costs, ability_gate_passed=ability,
        total_failed_and_recovery_physical_TRAIN_reads=6, final_solver_selected=False,
        knownDEV_accuracy=None, offnode_runtime_qualified=False))
    with (out/'per_field.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['method','field','passes','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for label, values in endpoints.items():
            for step in sorted(values, key=int):
                for key in keys:
                    row = values[step]['per_case'][key]
                    writer.writerow([label,key,step,row['u_relative_l2'],row['stress_relative_l2'],row['stress_hotspot_relative_l2']])
    shutil.copyfile(image, out/'representative_stress.png')
    assert d.sha(out/'representative_stress.png') == plot['identity']['sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, values in endpoints.items():
        steps = sorted(map(int, values))
        axes[0].plot(steps, [100*values[str(t)][FIELDS[1]] for t in steps], '-o', label=label)
        axes[1].plot(range(3), [100*values['1000']['per_case'][k]['stress_relative_l2'] for k in keys], '-o', label=label)
    axes[0].set_xlabel('Full-node exposures per field')
    axes[0].set_ylabel('Physical stress relative L2 (%)')
    axes[1].set_xticks(range(3), keys, rotation=15)
    axes[1].set_title('Shared3 fitting / no new CAD accuracy')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)
    files = ['BRIEF.md','metrics.json','per_field.csv','comparison.png','representative_stress.png']
    assert all((out/f).is_file() for f in files)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'],
        source_sha256=d.sha(OWN), terminal_binding=binding, files=files,
        actual_CPU_seconds=time.perf_counter()-began, GT_reads=0, PT_body_reads=0,
        GPU_calls=0, ability_gate_passed=ability))
    print(dict(status='completed', ability_gate_passed=ability, CPU_seconds=time.perf_counter()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","GPU_calls":0,"GT_reads":0}')
