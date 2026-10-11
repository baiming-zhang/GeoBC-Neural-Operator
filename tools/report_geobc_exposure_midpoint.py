"""Saved scalar-only100-exposure TRAIN comparison; not an inference stage."""
import argparse
import time


def export():
    started = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from tools.export_geobc_exposure_comparison import complete
    from tools.train_geobc_single_case_diagnostic import terminal_binding
    plan = d.read('notes/geobc_exposure_extension_plan.json'); science = plan['scientific']
    assert plan['scientific_digest'] == d.digest(science)
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    status_path = d.path(plan['output'])/'status.json'; status = d.read(status_path)
    assert status['execution']['scientific_digest'] == plan['scientific_digest']
    assert status['raw_GT_reads'] == status['DEV_FINAL_reads'] == status['scaler_fits'] == 0
    keys = science['TRAIN_keys']; assert len(keys) == 120
    points = {arm: status['arms'][arm].get('endpoints', {}).get('12000') for arm in ('FNO9', 'PP256')}
    if any(not point or 'checkpoint_identity' not in point for point in points.values()):
        print({'status': 'not_yet_ready', 'new_inference_GT_PT_reads': 0, 'export_writes': 0})
        return
    review = d.read(plan['root_review']); baseline = terminal_binding(review['parent_binding'])
    assert baseline['both4800'] and baseline['execution']['scientific_digest'] == science['parent_digest']
    report = dict(status='completed_actual_saved_TRAIN120_midpoint_comparison',
        scientific_digest=plan['scientific_digest'], observed_parent_status=status['status'],
        status_identity=d.identity(status_path, hashed=True),
        baseline_identity=review['parent_binding']['summary'],
        source_sha256=d.sha('tools/report_geobc_exposure_midpoint.py'),
        case_denominator=120, CAD_denominator=40, case_exposures=100,
        arms={}, physical_units=['mm', 'mm', 'mm', 'MPa'], signed_stress_unchanged=True,
        new_inference_or_GT_PT_array_reads=0, this_report_does_not_evaluate_final24000_or_DEV=True,
        interpretation='Same fullTRAIN120 and100 exposures, not newCAD generalization or final solver selection')
    names = dict(U='case_mean_u_relative_l2', S='case_mean_stress_relative_l2',
                 hot='case_mean_stress_hotspot_relative_l2')
    for arm, point in points.items():
        assert status['arms'][arm]['restored_Adam_RNG_and_case_cursor']
        assert status['arms'][arm]['first_actual_cached_fullN_gradient_check']['passed']
        complete(point['TRAIN120'], keys); complete(baseline['arms'][arm]['TRAIN120'], keys)
        assert point['case_exposure_count'] == 100
        assert max(point['epoch_seconds'], point['monotonic_seconds']) <= 1200
        d.stat_matches(point['checkpoint_identity'])  # No checkpoint body is read.
        old = baseline['arms'][arm]['TRAIN120']; new = point['TRAIN120']
        report['arms'][arm] = dict(baseline40={n: old[f] for n, f in names.items()},
            midpoint100={n: new[f] for n, f in names.items()},
            improvement_percentage_points={n: 100*(old[f]-new[f]) for n, f in names.items()},
            new_training_epoch_seconds_before_readout=point['training_epoch_seconds_before_readout'],
            new_training_monotonic_seconds_before_readout=point['training_monotonic_seconds_before_readout'],
            actual_TRAIN_readout_epoch_seconds=point['epoch_seconds'],
            actual_TRAIN_readout_monotonic_seconds=point['monotonic_seconds'],
            own_checkpoint_identity=point['checkpoint_identity'])
    report['PP_minus_FNO_midpoint_gap_percentage_points'] = {n: 100*(report['arms']['PP256']['midpoint100'][n]
        -report['arms']['FNO9']['midpoint100'][n]) for n in names}
    report['elapsed_CPU_seconds'] = time.monotonic()-started
    d.write('notes/geobc_exposure_12000_interim.json', report)
    output = d.path('results/geobc_exposure_comparison'); output.mkdir(parents=True, exist_ok=True)
    text = '# 100次曝光的训练集物理端点\n\n'
    text += '相同TRAIN40 CAD×三载荷、原节点和scaler，从完整4800检查点继续至12000更新，保留Adam／RNG／顺序。本报告只覆盖12000训练端点，不代替24000终态及新开发读出。\n\n'
    text += '| 方法 | 位移误差 | 应力误差 | 热点误差 | 新增训练UTC分钟 |\n|---|---:|---:|---:|---:|\n'
    for arm, label in [('FNO9', 'FNO'), ('PP256', 'Transolver++')]:
        row = report['arms'][arm]; m = row['midpoint100']
        text += f'| {label} | {100*m["U"]:.1f}% | {100*m["S"]:.1f}% | {100*m["hot"]:.1f}% | {row["new_training_epoch_seconds_before_readout"]/60:.2f} |\n'
    text += '\n这些是全120训练病例物理relative-L2均值，不能代表未见几何效果。热点取原应力绝对值最高1%节点。新增训练时间自4800恢复后至12000读出之前，原4800训练及GPU缓存准备另外计费，UTC／monotonic原值分别保留。\n'
    (output/'12000_INTERIM.md').write_text(text, encoding='utf-8')
    print({'status': report['status'], 'midpoint': {n: r['midpoint100'] for n, r in report['arms'].items()},
           'gap_percentage_points': report['PP_minus_FNO_midpoint_gap_percentage_points'], 'CPU_seconds': report['elapsed_CPU_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: export()
    else: print('{"status":"source_only","reads":0,"writes":0,"GPU_execution":0}')
