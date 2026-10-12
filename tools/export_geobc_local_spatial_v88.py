"""Scalar-only report of the real fixed TRAIN9 local spatial operator attempt."""
import argparse
import csv
import json
import time

OWN = 'tools/export_geobc_local_spatial_v88.py'
REVIEW = 'notes/geobc_local_spatial_operator_v88_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.local_spatial_operator_contract_v88 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical
    p, a, e, binding = terminal(PLAN); s = p['scientific']; r = d.read(REVIEW); arm = s['candidate']; z = a['arms'][arm]
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256'] == d.sha(OWN) and r['SCI'] == p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 900
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts'] == a['cache_receipt']['physical_TRAIN_target_reads'] == 9
    assert a['cache_receipt']['encoded_TRAIN_target_reads'] == a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == a['parent_PT_reads'] == 0
    assert a['resident_physical_target_reuses'] == 27 and z['step'] == 900 and z['case_exposure_counts'] == {k: 100 for k in s['selected_TRAIN_keys']}
    assert z['qualification']['passed'] and z['qualification']['original_nodes'] == 242936
    assert z['qualification']['model_unchanged'] and z['qualification']['Adam_unchanged'] and z['qualification']['fourRNG_restored']
    assert z['initialization']['original_model_digest'] == s['expected_original_model_digest']
    assert z['initialization']['shared_body_digests'] == s['expected_original_shared_body_digests'] and z['initialization']['parameter_count'] == 4535642
    assert z['initialization']['local_parameters_initially_zero'] and z['initialization']['local_branch_parameters'] == 8784
    assert z['qualification']['initial_function_parity']['passed'] and z['qualification']['initial_function_parity']['whole_R64_latent_exact']
    assert all(v['finite'] and not v['missing'] and v['L2'] > 0 for v in z['qualification']['local_branch_gradients'])
    for point in z['endpoints'].values(): complete_physical(point['full_TRAIN9'], s['selected_TRAIN_keys'])
    out = d.path('results/geobc_local_spatial_operator_v88'); assert not (out/'manifest.json').exists()
    out.mkdir(parents=True, exist_ok=True); gate = a['TRAIN_gate']
    old = s['baseline_scalar_endpoints']['100']['full_TRAIN9']; new = z['endpoints']['100']['full_TRAIN9']
    lines = ['# FNO内部局部空间传播：短对照', '', '**同三CAD×三载荷／100遍／900次全节点更新；原基线已有权重初值、输入、解码器、物理损失、Adam和scaler保持。新四个零初始卷积分支位于频域块内部；旧基线仅冻结标量，不重跑。**', '',
        '| 场算子 | 位移误差 | signed应力误差 | 热点误差 |', '|---|---:|---:|---:|']
    for label, row in (('原FNO', old), ('FNO + 局部3D空间传播', new)):
        lines.append('| '+label+' | '+' | '.join(f'{100*row[f]:.2f}%' for f in FIELDS)+' |')
    lines += ['', f'新算子减旧基线：位移{100*gate["U_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点，联合{100*gate["primary_difference"]:+.2f}个百分点。预声明TRAIN门：{"PASS" if gate["passed"] else "FAIL"}。', '',
        f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒／学习{z["training_epoch_seconds"]:.2f}秒，新参数4,535,642对旧4,526,858，只增加8,784（四个9→9/kernel3/padding1分支）。原width9/modes12/layers4/R64输入61/latent16/native25查询4输出保持。', '',
        '新分支全部从0开始；原参数子集digest与旧基线完全一致。真实最大242936节点资格时，开启零分支与禁用分支的整个R64 latent及64个查询输出精确一致。各新增分支与原encoder/operator/head/context/latent梯度有限非零，model／Adam／四RNG保留，实际64坐标链与分块检查通过。', '',
        '**仍是九TRAIN场拟合，不是新CAD泛化。** 这是项目局部空间增强FNO变体，不冒充作者UNO/U-FNO复现；增加空间传播与容量，不能单独归因于高频。不是冻结teacher后的节点残差head，也不加入canonicalcloud、符号幅值BCE或真值参考张量。', '',
        '原9物理标签各读一次，三个端点常驻复用27；编码／CSV／DEV／FINAL／父PT／scaler拟合0。所有案例与端点保留，不按子集或早停选结果。', '',
        '门失败关闭固定分支，不扫kernel/width/modes/seed/loss或读DEV救结果；门通过仅支持另冻结的短100TRAIN宏观比较，不能据训练结果宣布solver winner或自动扩大。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        baseline_scalar_endpoints=s['baseline_scalar_endpoints'], TRAIN_gate=gate, generalization_tested=False, final_solver_selected=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file); writer.writerow(['method', 'passes', 'case', 'U_relative_L2', 'signed_S_relative_L2', 'hotspot_relative_L2'])
        for label, endpoints in (('old_plain_scalar', s['baseline_scalar_endpoints']), (arm, z['endpoints'])):
            for passes in (0, 10, 100):
                for key, row in endpoints[str(passes)]['full_TRAIN9']['per_case'].items():
                    writer.writerow([label, passes, key, row['u_relative_l2'], row['stress_relative_l2'], row['stress_hotspot_relative_l2']])
    loss = [json.loads(line) for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss) == 900 and [v['step'] for v in loss] == list(range(1, 901))
    d.write(out/'loss.json', loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.6))
    for field, label in (('physical_relative_squared', 'Local spatial FNO physical loss'),):
        vals = [sum(v[field] for v in loss[i:i+9])/9 for i in range(0, 900, 9)]
        axes[0].plot(range(1, 101), vals, label=label)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Full TRAIN9 passes'); axes[0].legend()
    for offset, row, label in ((-.18, old, 'Old plain FNO'), (.18, new, 'Local spatial FNO')):
        axes[1].bar([i+offset for i in range(3)], [100*row[f] for f in FIELDS], .36, label=label)
    axes[1].set_xticks(range(3), ('U', 'Signed S', 'Hotspot')); axes[1].set_ylabel('Mean relative L2 (%)'); axes[1].set_ylim(0,max(row[f] for row in (old,new) for f in FIELDS)*100+10); axes[1].legend(loc='upper center',ncol=2,fontsize=8)
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle('TRAIN9 fixed900updates; local spatial operator; original inputs and decoder unchanged')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN), terminal_binding=binding,
        files=['BRIEF.md', 'metrics.json', 'per_case.csv', 'loss.json', 'comparison.png'], actual_CPU_seconds=time.perf_counter()-began,
        new_GT_PT_arrays_GPU=0, TRAIN_gate=gate, generalization_tested=False))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, TRAIN_gate=gate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: export()
    else: print('{"status":"source_only","new_GT_PT_arrays_GPU":0}')
