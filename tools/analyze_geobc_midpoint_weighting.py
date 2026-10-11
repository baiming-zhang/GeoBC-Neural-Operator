"""Explain saved TRAIN40->100exposure errors with case/node weighting.

Only already computed physical metric scalars are read. This is an exploratory
TRAIN diagnostic, not a new loss experiment or a change of selection metric.
"""
import argparse
import math
import statistics
import time


def analyze():
    started = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from tools.train_geobc_single_case_diagnostic import terminal_binding
    from tools.export_geobc_exposure_comparison import complete
    plan = d.read('notes/geobc_exposure_extension_plan.json'); science = plan['scientific']
    assert plan['scientific_digest'] == d.digest(science)
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    review = d.read(plan['root_review']); baseline = terminal_binding(review['parent_binding'])
    status = d.read(d.path(plan['output'])/'status.json')
    assert baseline['both4800'] and baseline['execution']['scientific_digest'] == science['parent_digest']
    assert status['execution']['scientific_digest'] == plan['scientific_digest']
    keys = science['TRAIN_keys']; assert len(keys) == 120
    cads = sorted({k.split('/')[0] for k in keys}); assert len(cads) == 40
    fields = {'U': ('u_relative_l2', 'u_vector_error_RMS_mm'), 'S': ('stress_relative_l2', 'stress_RMSE_MPa')}
    report = dict(status='completed_actual_TRAIN_midpoint_weighting_diagnostic',
        scientific_digest=plan['scientific_digest'], source_sha256=d.sha('tools/analyze_geobc_midpoint_weighting.py'),
        baseline_binding=review['parent_binding'], arms={}, case_denominator=120, CAD_denominator=40,
        fixed_primary_metric='.5(case-mean U relativeL2+case-mean signed S relativeL2); unchanged',
        reference_inference='Fixed per-case physical truth RMS = saved errorRMS/relativeL2; require old/new agreement, no numerical target body access',
        node_pooled_formula='sqrt(sum(N*errorRMS^2)/sum(N*referenceRMS^2)); supplemental, not primary selection',
        real_input_GT_PT_array_reads=0, GPU_new_inference_or_training=0, loss_or_training_changes=0,
        causal_loss_or_architecture_effect_proven=False, development_or_FINAL_result=False)
    for arm in ('FNO9', 'PP256'):
        old_metric = baseline['arms'][arm]['TRAIN120']; point = status['arms'][arm]['endpoints']['12000']
        assert point['case_exposure_count'] == 100 and 'checkpoint_identity' in point
        new_metric = point['TRAIN120']; complete(old_metric, keys); complete(new_metric, keys)
        old = old_metric['per_case']; new = new_metric['per_case']
        row = dict(old_metric_digest=d.digest(old_metric), midpoint_metric_digest=d.digest(new_metric),
            input_scalar_metrics={'40': old_metric, '100': new_metric}, checkpoint_identity=point['checkpoint_identity'], metrics={})
        for name, (rel, rms) in fields.items():
            reference = {}
            for key in keys:
                a = old[key]; b = new[key]
                assert a['nodes'] == b['nodes'] and a[rel] > 0 and b[rel] > 0
                reference[key] = a[rms]/a[rel]
                assert reference[key] > 0 and math.isclose(reference[key], b[rms]/b[rel], rel_tol=1e-10, abs_tol=1e-12)
            denominator = sum(old[k]['nodes']*reference[k]**2 for k in keys)
            deltas = {k: new[k][rel]-old[k][rel] for k in keys}
            cad_deltas = {c: sum(deltas[c+'/'+load] for load in ('ver', 'hor', 'tor'))/3 for c in cads}
            sorted_keys = sorted(keys, key=lambda k: (reference[k], k))
            groups = {}
            for index, label in enumerate(('low_reference_RMS40', 'middle_reference_RMS40', 'high_reference_RMS40')):
                group = sorted_keys[index*40:(index+1)*40]
                groups[label] = dict(cases=40, reference_RMS_range=[reference[group[0]], reference[group[-1]]],
                    mean_relative_error40=sum(old[k][rel] for k in group)/40,
                    mean_relative_error100=sum(new[k][rel] for k in group)/40,
                    worsened_cases=sum(deltas[k] > 0 for k in group))
            row['metrics'][name] = dict(units='mm' if name == 'U' else 'MPa',
                worsened_relative_and_RMS_cases=sum(deltas[k] > 0 for k in keys),
                improved_cases=sum(deltas[k] < 0 for k in keys),
                worsened_CAD_mean_count=sum(v > 0 for v in cad_deltas.values()),
                per_CAD40_relative_error_change=cad_deltas,
                median_relative_change_percentage_points=100*statistics.median(deltas.values()),
                mean_relative_error={phase: sum(rows[k][rel] for k in keys)/120 for phase, rows in [('40', old), ('100', new)]},
                case_mean_physical_error_RMS={phase: sum(rows[k][rms] for k in keys)/120 for phase, rows in [('40', old), ('100', new)]},
                node_pooled_relative_error={phase: math.sqrt(sum(rows[k]['nodes']*rows[k][rms]**2 for k in keys)/denominator)
                                           for phase, rows in [('40', old), ('100', new)]},
                per_load40={load: dict(worsened_cases=sum(deltas[k] > 0 for k in keys if k.endswith('/'+load)),
                    mean_relative_change_percentage_points=100*sum(deltas[k] for k in keys if k.endswith('/'+load))/40)
                    for load in ('ver', 'hor', 'tor')}, reference_RMS_strata=groups)
        report['arms'][arm] = row
    report['elapsed_CPU_seconds'] = time.monotonic()-started
    d.write('notes/geobc_exposure_midpoint_weighting_diagnostic.json', report)
    pp = report['arms']['PP256']['metrics']['U']; fno = report['arms']['FNO9']['metrics']['U']
    text = '# ++中间位移误差：权重与病例分布\n\n'
    text += (f'从40次至100次曝光，++位移误差在 **{pp["worsened_relative_and_RMS_cases"]}/120病例、'
             f'{pp["worsened_CAD_mean_count"]}/40 CAD** 上变大。相同病例的相对误差与实际RMS变化方向一致；这不只是一两个病例的现象。\n\n')
    a = pp['mean_relative_error']; b = pp['node_pooled_relative_error']; c = pp['case_mean_physical_error_RMS']
    text += (f'病例均值relative-L2从{100*a["40"]:.1f}%恶化到{100*a["100"]:.1f}%，'
             f'但按节点及参考位移能量合并的relative-L2从{100*b["40"]:.1f}%降至{100*b["100"]:.1f}%。'
             f'病例均值实际位移误差RMS几乎不变：{c["40"]:.5f}→{c["100"]:.5f}mm。大位移／大网格病例的改善能补偿其他病例，不能据病例相对均值单独宣称全局发散。\n\n')
    text += '| 参考位移RMS组 | ++位移误差40→100 | 恶化病例 |\n|---|---:|---:|\n'
    for name, label in [('low_reference_RMS40', '较小40例'), ('middle_reference_RMS40', '中间40例'), ('high_reference_RMS40', '较大40例')]:
        g = pp['reference_RMS_strata'][name]
        text += f'| {label} | {100*g["mean_relative_error40"]:.1f}%→{100*g["mean_relative_error100"]:.1f}% | {g["worsened_cases"]}/40 |\n'
    text += (f'\nFNO的位移恶化病例为{fno["worsened_relative_and_RMS_cases"]}/120；病例均值、节点合并误差及实际RMS总体均改善。'
             '原预定主指标仍是病例等权联合U／S误差，节点合并指标只作解释，不替换排名。\n\n')
    text += ('这支持进一步区分目标／指标权重与表示问题；没有证明某个损失或架构造成变化。'
             '当前训练配方及固定24000端点不变，后续原生网格++／条件化Hash共同对照仍须开发集结果决定。'
             '已有物理指标反推出的参考RMS在两个端点逐病例一致；没有新读目标数组、推理或scaler拟合。\n')
    (d.path('results/geobc_exposure_comparison')/'MIDPOINT_DIAGNOSTIC.md').write_text(text, encoding='utf-8')
    print({'status': report['status'], 'PP_U_worsened_cases': pp['worsened_relative_and_RMS_cases'],
           'PP_U_worsened_CAD': pp['worsened_CAD_mean_count'], 'PP_U_pooled': b,
           'PP_U_strata': pp['reference_RMS_strata'], 'CPU_seconds': report['elapsed_CPU_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--analyze', action='store_true')
    if parser.parse_args().analyze: analyze()
    else: print('{"status":"source_only","reads":0,"writes":0,"GPU_execution":0}')
