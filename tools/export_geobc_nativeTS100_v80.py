"""One scalar/loss CPU report after genuine nativeTS100 terminal; no arrays/PT."""
import argparse
import csv
import json
import time

OWN = 'tools/export_geobc_nativeTS100_v80.py'
REVIEW = 'notes/geobc_nativeTS100_matched10_v80_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2',
          'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.nativeTS100_contract_v80 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical
    p, a, e, binding = terminal(PLAN); s = p['scientific']; keys = s['TRAIN_keys']
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN) and marker['SCI'] == p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and a['qualification']['passed']
    assert a['qualification']['nodes'] == 325387 and a['qualification']['exact_dense_initial_bytes']
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == a['step'] == a['cursor'] == 3000
    assert a['case_exposure_counts'] == {k: 10 for k in keys}
    assert a['physical_TRAIN_attempts'] == a['physical_TRAIN_reads'] == 300
    assert a['cache_receipt']['physical_TRAIN_target_reads'] == 300
    assert a['cache_receipt']['encoded_TRAIN_target_reads'] == a['cache_receipt']['R64_input_array_reads'] == 0
    assert a['resident_physical_target_readout_reuses'] == 600
    assert a['parent_checkpoint_hash_reads'] == a['parent_checkpoint_loads'] == a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == 0
    assert not a['automatic_continuation'] and not a['final_solver_selected']
    native = {step: a['endpoints'][str(step*300)]['full_TRAIN'] for step in (5, 10)}
    dense = {step: s['reference_TRAIN'+str(step)] for step in (5, 10)}
    for z in list(native.values())+list(dense.values()):
        complete_physical(z, keys)
    diff = {f: 100*(native[10][f]-dense[10][f]) for f in FIELDS}
    gate = a['TRAIN_gate']
    out = d.path('results/geobc_nativeTS100_matched10_v80')
    assert not (out/'manifest.json').exists(), 'One immutable actual export'
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# 原网格Transolver：100CAD短比较', '',
        '**固定100 TRAIN CAD×三载荷、全原节点物理U/S目标。同原版TS参数、PointNet／TCNN初值、精度、Adam和曝光顺序；改为原节点上下文及native latent。**', '',
        '| 同10遍方案 | 位移误差 | 应力误差 | 热点误差 |',
        '|---|---:|---:|---:|']
    for label, z in [('原网格TS10', native[10]), ('已有R64 TS10', dense[10])]:
        lines.append(f'| {label} | {100*z[FIELDS[0]]:.2f}% | {100*z[FIELDS[1]]:.2f}% | {100*z[FIELDS[2]]:.2f}% |')
    lines += ['', f'原网格相对R64：位移{diff[FIELDS[0]]:+.2f}、应力{diff[FIELDS[1]]:+.2f}、热点{diff[FIELDS[2]]:+.2f}个百分点；预声明TRAIN推进门通过：{gate["passed"]}。', '',
        '下一步独立冻结同曝光开发集比较，仍不自动20遍。' if gate['passed'] else '关闭这条表示分支，不扩大训练，不用开发集小调救结果。', '',
        f'完整阶段{e["elapsed_epoch_seconds"]/60:.2f}分钟；学习区间{a["training_epoch_seconds"]/60:.2f}分钟，包含初始化／资格、中间5遍读出及保存，10遍终点读出另计。原阶段／不同方案成本不是等预算；epoch与monotonic原样保留在JSON。', '',
        '最大真实325387节点全参数组及context／native latent梯度通过，资格未改变模型、空Adam或四RNG。物理TRAIN300只读一次，两个全TRAIN端点常驻标签读出复用600次；编码目标、R64输入、原CSV、DEV、FINAL、scaler拟合和父PT读取均0。', '',
        '**这里只检验共享拟合，不能宣布新CAD赢家。** 原节点注意力按节点密度，更换上下文与latent延拓属于表示包，不是单一空体素因果。原网格离节点查询需显式有效TET10单元，没有独立离节点FEM精度声明。旧occupied-R16和Native++负结果仍保留，未重放。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        dense_TRAIN_references=dense, differences_pp=diff, gate=gate, new_knownDEV_completed=False,
        final_solver_selected=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file); writer.writerow(['representation', 'passes', 'case', 'U_relative_L2', 'signed_S_relative_L2', 'hotspot_relative_L2'])
        for label, data in [('native', native), ('dense_R64_reference', dense)]:
            for passes, metric in data.items():
                for key in keys:
                    z = metric['per_case'][key]
                    writer.writerow([label, passes, key, z['u_relative_l2'], z['stress_relative_l2'], z['stress_hotspot_relative_l2']])
    loss = [json.loads(line) for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss) == 3000 and [z['step'] for z in loss] == list(range(1, 3001))
    d.write(out/'loss.json', loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    values = [sum(z['physical_relative_squared'] for z in loss[i:i+300])/300 for i in range(0, 3000, 300)]
    axes[0].plot(range(1, 11), values, '-o'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Full TRAIN300 passes'); axes[0].set_ylabel('Mean physical squared relative loss')
    for shift, z, label in [(-.15, native[10], 'Native TS10'), (.15, dense[10], 'Dense R64 TS10')]:
        axes[1].bar([i+shift for i in range(3)], [100*z[f] for f in FIELDS], width=.29, label=label)
    axes[1].set_xticks(range(3), ['U', 'Signed S', 'Hotspot']); axes[1].set_ylabel('Case mean relative L2 (%)'); axes[1].legend()
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Original Transolver / matched10 TRAIN exposure, not new CAD evaluation')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    files = ['BRIEF.md', 'metrics.json', 'per_case.csv', 'loss.json', 'comparison.png']
    assert all((out/name).is_file() for name in files)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN),
        terminal_binding=binding, files=files, actual_CPU_seconds=time.perf_counter()-began,
        PT_body_reads=0, new_GT_reads=0, GPU_calls=0))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, TRAIN_gate_passed=gate['passed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","scientific_bodies":0,"GPU":0}')
