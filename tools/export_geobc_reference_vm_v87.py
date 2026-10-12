"""Scalar-only accepted reference-U closure report; no field reloads."""
import argparse
import csv
import time

OWN = 'tools/export_geobc_reference_vm_v87.py'
REVIEW = 'notes/geobc_reference_vm_closure_v87_exporter_root_review.json'


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_decoder_contract_v64 import terminal
    from geobc.reference_vm_contract_v87 import PLAN, CASES
    p, a, e, binding = terminal(PLAN); r = d.read(REVIEW)
    assert r['passed'] and r['whole_source_reviewed'] and r['conditional_actual_export_released']
    assert r['source_sha256'] == d.sha(OWN) and r['SCI'] == p['scientific_digest']
    assert a['budget_true'] and a['GT_U_archives_attempted'] == a['GT_U_archives_completed'] == 3
    assert a['cached_S_attempted'] == a['cached_S_completed'] == 3 and a['all_q_committed_before_S']
    assert a['DEV_FINAL_reads'] == a['training_updates'] == a['raw_CSV_H5_reads'] == a['PT_reads'] == a['new_physical_solves'] == 0
    assert not a['sign_rule_search'] and not a['fit_or_calibration']
    assert set(a['case_metrics']) == set(a['reference_q_arrays']) == set(CASES)
    assert all(v['completed'] for v in a['read_receipt']) and len(a['read_receipt']) == 12
    assert all(a['reference_q_arrays'][k]['full_node_coverage'] and a['case_metrics'][k]['all_finite']
               and a['case_metrics'][k]['original_nodes'] == p['scientific']['records'][k]['nodes'] for k in CASES)
    rows = a['case_metrics']; out = d.path('results/geobc_reference_vm_closure_v87')
    assert not (out/'manifest.json').exists(); out.mkdir(parents=True, exist_ok=True)
    old = sum(v['old_physical_q_relativeL2'] for v in rows.values())/3
    old_hot = sum(v['old_physical_q_hotspot_relativeL2'] for v in rows.values())/3
    lines = ['# 位移真值代入应力恢复：幅值闭合诊断', '',
        '**换成原位移真值，当前恢复公式的幅值差异仍约10.78%，几乎未变。不能把这个公式当成原CSV应力的精确定义。**', '',
        '| 同三TRAIN CAD／ver／全部原节点 | 幅值误差 | 热点误差 |', '|---|---:|---:|',
        f'| 旧有限元位移→固定恢复公式（冻结标量） | {100*old:.2f}% | {100*old_hot:.2f}% |',
        f'| 原位移真值→相同公式（新诊断） | {100*a["mean_relativeL2"]:.2f}% | {100*a["mean_hotspot_relativeL2"]:.2f}% |', '',
        f'幅值差仅{100*(a["mean_relativeL2"]-old):+.4f}个百分点，热点{100*(a["mean_hotspot_relativeL2"]-old_hot):+.4f}个百分点。预声明每例幅值≤5%、热点≤10%的闭合门：{"PASS" if a["GT_U_removes_magnitude_mismatch_gate"] else "FAIL"}。完整阶段{e["elapsed_epoch_seconds"]:.2f}秒。', '',
        '使用原GRID对齐U(mm)、真实MAT1与已接受全TET10四点物理梯度；体积加权单元平均张量，再对原节点做正权平均，最后计算非负VM。与旧恢复数学保持一致，唯一改变是位移来源。未枚举节点／平均／符号规则，未拟合或校准材料。', '',
        '所有三份参考q先保存并核验文件哈希，再各读取一次已有原signedS，仅比较abs(S)，不改原标签。原应力CSV缺少完整张量及H3D导出契约，此结果不能确认哪个后处理环节导致差异。', '',
        '**这是使用真值位移的参考诊断，不是可部署求解器，也不是49%模型误差的下限。** 结论仅限这三个旧TRAIN CAD／ver和固定恢复proxy；不据此排除物理信息。位移求解正确与signed应力出口完全一致是两个问题，不能直接将此proxy加入精确标量物理损失。', '',
        '本次U档案3、已有S缓存3、输入缓存3（四成员）、梯度3、权重3；新DEV／FINAL、原CSV/H5、PT、优化更新和物理解算均0。旧物理成本与指标保留，未重播旧求解／tensor／head负结果。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        source_semantics_not_identified=True, signed_rule_inferred=False, final_solver_selected=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['case', 'nodes', 'old_physical_U_q_relativeL2', 'reference_GT_U_q_relativeL2', 'old_hotspot_relativeL2', 'reference_hotspot_relativeL2'])
        for k in CASES:
            v = rows[k]; writer.writerow([k, v['original_nodes'], v['old_physical_q_relativeL2'], v['relativeL2'], v['old_physical_q_hotspot_relativeL2'], v['hotspot_relativeL2']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for offset, vals, name in ((-.18, [old, old_hot], 'Existing physical U (scalar reuse)'),
                               (.18, [a['mean_relativeL2'], a['mean_hotspot_relativeL2']], 'Original GT U reference')):
        bars = ax.bar([offset, 1+offset], [100*v for v in vals], .36, label=name)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=2)
    ax.set_xticks([0, 1], ['Magnitude', 'Top 1% hotspot']); ax.set_ylabel('Mean relative L2 (%)')
    ax.set_ylim(0, 25); ax.legend(loc='upper center', fontsize=8)
    ax.set_title('TRAIN3/ver reference closure: magnitude only; not a deployed solver')
    ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN), terminal_binding=binding,
        files=['BRIEF.md', 'metrics.json', 'per_case.csv', 'comparison.png'], actual_CPU_seconds=time.perf_counter()-began,
        new_GT_PT_arrays_GPU=0, signed_prediction=False, reference_U_gate=a['GT_U_removes_magnitude_mismatch_gate']))
    print(dict(status='completed', CPU_seconds=time.perf_counter()-began, mean_reference_magnitude=a['mean_relativeL2']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: export()
    else: print('{"status":"source_only","new_GT_PT_arrays_GPU":0}')
