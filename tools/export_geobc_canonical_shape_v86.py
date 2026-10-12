"""Scalar-only report of the real fixed TRAIN9 canonical geometry attempt."""
import argparse
import csv
import json
import time

OWN = 'tools/export_geobc_canonical_shape_v86.py'
REVIEW = 'notes/geobc_canonical_shape_encoding_v86_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.canonical_shape_contract_v86 import PLAN
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
    assert z['initialization']['model_digest'] == s['expected_model_initial_digest']
    assert z['initialization']['shared_body_digests'] == s['expected_shared_body_digests'] and z['initialization']['parameter_count'] == 4526858
    geometry = a['geometry_canonicalization']
    assert geometry['CAD_transform_computations'] == 3 and geometry['global_input_tensor_references_and_versions_unchanged']
    assert geometry['labels_not_accessed_or_transformed'] and geometry['no_learned_context_cached']
    assert all(v['proper_orthogonal_frame'] and v['max_roundtrip_normalized_error'] < 2e-6 for v in geometry['frames'].values())
    for point in z['endpoints'].values(): complete_physical(point['full_TRAIN9'], s['selected_TRAIN_keys'])
    out = d.path('results/geobc_canonical_shape_encoding_v86'); assert not (out/'manifest.json').exists()
    out.mkdir(parents=True, exist_ok=True); gate = a['TRAIN_gate']
    old = s['baseline_scalar_endpoints']['100']['full_TRAIN9']; new = z['endpoints']['100']['full_TRAIN9']
    lines = ['# 规范化形状点云：短对照', '', '**同三CAD×三载荷／100遍／900次全节点更新，原普通FNO全模型初始权重、物理损失、Adam和scaler保持。仅改变PointNet输入形状点云；旧基线仅冻结标量，不重跑。**', '',
        '| 几何编码输入 | 位移误差 | signed应力误差 | 热点误差 |', '|---|---:|---:|---:|']
    for label, row in (('原点云', old), ('中心／尺度／PCA形状点云', new)):
        lines.append('| '+label+' | '+' | '.join(f'{100*row[f]:.2f}%' for f in FIELDS)+' |')
    lines += ['', f'新输入减旧基线：位移{100*gate["U_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点，联合{100*gate["primary_difference"]:+.2f}个百分点。预声明TRAIN门：{"PASS" if gate["passed"] else "FAIL"}。', '',
        f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒／学习{z["training_epoch_seconds"]:.2f}秒，参数4,526,858。三个CAD各一次无标签／载荷依赖的中心、RMS尺度和右手PCA变换，三种载荷共享。原global xyz／native25／grid／stencil／mask／材料力矩和U／signedS标签保持，不旋转载荷或标签。', '',
        '**仍是九TRAIN场拟合，不是新CAD泛化。** 规范化同时改变中心、尺度和姿态，不能归因于单一旋转或宣称全模型旋转等变；近重根用单位阵回退，网格采样密度及形状帧连续性局限保留。原查询和全局输入仍带姿态与物理尺度。', '',
        '真实三CAD变换的正交性、右手系、mask、往返误差／同CAD跨载荷复用和未改变全局张量引用与版本已核对；实际242936节点全参数／两leaf、64query链和model／Adam／四RNG资格通过。', '',
        '原9标签各读一次，三端点常驻复用27；编码／CSV／DEV／FINAL／父PT／scaler拟合0，无v85符号幅值结构或辅助BCE。所有案例、三端点、帧及成本保留，未通过门则关闭固定输入包，不轴／尺度搜索或DEV救结果，不自动扩大。']
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
    for field, label in (('physical_relative_squared', 'Canonical-cloud physical loss'),):
        vals = [sum(v[field] for v in loss[i:i+9])/9 for i in range(0, 900, 9)]
        axes[0].plot(range(1, 101), vals, label=label)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Full TRAIN9 passes'); axes[0].legend()
    for offset, row, label in ((-.18, old, 'Old plain FNO'), (.18, new, 'Canonical shape cloud')):
        axes[1].bar([i+offset for i in range(3)], [100*row[f] for f in FIELDS], .36, label=label)
    axes[1].set_xticks(range(3), ('U', 'Signed S', 'Hotspot')); axes[1].set_ylabel('Mean relative L2 (%)'); axes[1].set_ylim(0,max(row[f] for row in (old,new) for f in FIELDS)*100+10); axes[1].legend(loc='upper center',ncol=2,fontsize=8)
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle('TRAIN9 fixed900updates; geometry cloud package only, global frame unchanged')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN), terminal_binding=binding,
        files=['BRIEF.md', 'metrics.json', 'per_case.csv', 'loss.json', 'comparison.png'], actual_CPU_seconds=time.perf_counter()-began,
        new_GT_PT_arrays_GPU=0, TRAIN_gate=gate, generalization_tested=False))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, TRAIN_gate=gate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: export()
    else: print('{"status":"source_only","new_GT_PT_arrays_GPU":0}')
