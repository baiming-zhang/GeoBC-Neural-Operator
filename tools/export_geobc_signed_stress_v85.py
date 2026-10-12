"""Scalar-only report of the real fixed TRAIN9 sign/magnitude attempt."""
import argparse
import csv
import json
import time

OWN = 'tools/export_geobc_signed_stress_v85.py'
REVIEW = 'notes/geobc_signed_stress_factorization_v85_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.signed_stress_factorization_contract_v85 import PLAN
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
    assert all(v > 0 for v in z['qualification']['stress_output_row_gradients'].values())
    assert z['initialization']['shared_body_digests'] == s['expected_shared_body_digests']
    assert z['initialization']['canonical_copy_proof']['canonical_model_digest'] == s['expected_canonical_initial_model_digest']
    assert z['initialization']['parameter_count'] == 4526987
    for point in z['endpoints'].values(): complete_physical(point['full_TRAIN9'], s['selected_TRAIN_keys'])
    out = d.path('results/geobc_signed_stress_factorization_v85'); assert not (out/'manifest.json').exists()
    out.mkdir(parents=True, exist_ok=True); gate = a['TRAIN_gate']
    old = s['baseline_scalar_endpoints']['100']['full_TRAIN9']; new = z['endpoints']['100']['full_TRAIN9']
    lines = ['# 符号／幅值联合建模短验证', '', '**固定三CAD×三载荷／100遍／900次全节点更新。旧普通FNO仅复用冻结标量，没有重跑基线或读取旧PT。**', '',
        '| 方法 | 位移误差 | signed应力误差 | 热点误差 |', '|---|---:|---:|---:|']
    for label, row in (('普通FNO旧基线', old), ('符号／幅值＋固定辅助监督', new)):
        lines.append('| '+label+' | '+' | '.join(f'{100*row[f]:.2f}%' for f in FIELDS)+' |')
    lines += ['', f'新方案减旧基线：位移{100*gate["U_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点，联合误差{100*gate["primary_difference"]:+.2f}个百分点。预声明S至少改善5个百分点／联合至少3个百分点／任意字段与热点劣化不超2个百分点：{"PASS" if gate["passed"] else "FAIL"}。', '',
        f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，学习{z["training_epoch_seconds"]:.2f}秒；全部初始化、资格、缓存、三端点读出和保存成本分别保留。参数增加129至4,526,987；原骨干和U头初始字节保留，初始应力函数改变。表示与辅助目标一起改变，不是单因果实验。', '',
        '实际signed标签不变。正幅值softplus与连续软符号tanh组合，再经原scaler返回encoded4；固定0.1辅助BCE按原物理应力平方能量加权，目标取物理S>0，不能取去均值编码的符号。零GT辅助权重0，无真值符号oracle进入预测。', '',
        '**这是TRAIN拟合，不是新CAD泛化。** 未通过门则关闭这一固定配置，不调lambda／头尺寸或读DEV救结果；通过也只推进下一独立设计，不自动扩大。物理损失、未乘系数辅助BCE、总目标分别记录，训练曲线总目标不能冒充纯物理损失。', '',
        '真实242936节点的全部参数组／两leaf／幅值与符号输出行梯度及64query链通过，资格保持model／Adam／四RNG；原9标签各读一次，三端点常驻复用27。原CSV／编码标签／DEV／FINAL／父PT／scaler拟合0；own检查点及全部成本保留本地。']
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
    assert all(abs(v['total_objective']-v['physical_relative_squared']-.1*v['energy_weighted_sign_BCE']) < 1e-8 for v in loss)
    d.write(out/'loss.json', loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.6))
    for field, label in (('physical_relative_squared', 'Physical joint loss'), ('energy_weighted_sign_BCE', 'Sign BCE (unweighted)'), ('total_objective', 'Physical + 0.1 BCE')):
        vals = [sum(v[field] for v in loss[i:i+9])/9 for i in range(0, 900, 9)]
        axes[0].plot(range(1, 101), vals, label=label)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Full TRAIN9 passes'); axes[0].legend()
    for offset, row, label in ((-.18, old, 'Old plain FNO'), (.18, new, 'Sign/magnitude recipe')):
        axes[1].bar([i+offset for i in range(3)], [100*row[f] for f in FIELDS], .36, label=label)
    axes[1].set_xticks(range(3), ('U', 'Signed S', 'Hotspot')); axes[1].set_ylabel('Mean relative L2 (%)'); axes[1].legend()
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle('TRAIN9 fixed900updates; physical loss and auxiliary objective distinguished')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN), terminal_binding=binding,
        files=['BRIEF.md', 'metrics.json', 'per_case.csv', 'loss.json', 'comparison.png'], actual_CPU_seconds=time.perf_counter()-began,
        new_GT_PT_arrays_GPU=0, TRAIN_gate=gate, generalization_tested=False))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, TRAIN_gate=gate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: export()
    else: print('{"status":"source_only","new_GT_PT_arrays_GPU":0}')
