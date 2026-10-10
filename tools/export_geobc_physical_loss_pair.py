"""One CPU scalar export; no predictions, targets, checkpoints or CUDA imports."""
import argparse
import csv
import io
import math
import os
import time
from geobc import available_cad_pilot_data as d


def run():
    start = time.monotonic()
    plan = d.read('notes/geobc_physical_loss_pair_plan.json')
    review = d.read(plan['root_review'])
    export_review = d.read('notes/geobc_physical_loss_exporter_review.json')
    assert export_review['passed'] and export_review['source_sha256'] == d.sha(__file__)
    assert export_review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    baseline_binding = review['baseline_binding']
    for item in baseline_binding.values(): assert d.identity(item['path'], hashed=True) == item
    base = d.read(baseline_binding['summary']['path'])
    base_external = d.read(baseline_binding['external']['path'])
    folder = d.path(plan['output'])
    result = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    for native, outer in ((base, base_external), (result, external)):
        assert native['closed'] and not native['active'] and outer['closed'] and not outer['active']
        assert all(d.exact_process_exited(outer[k]) for k in ('guard_process', 'child_process'))
    assert result['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert base['selected'] == result['selected'] == plan['scientific']['fixed_selected']
    fields = ['u_relative_l2', 'stress_relative_l2', 'stress_hotspot_relative_l2']

    def finite(v): return type(v) in (float, int) and math.isfinite(v) and v >= 0
    def point(native, step):
        row = native.get('endpoints', {}).get(str(step), {})
        metric = row.get('physical', {})
        complete = (row.get('case_exposures') == row.get('optimizer_steps') == step and
                    metric.get('nodes') == base['original_nodes'] and all(finite(metric.get(k)) for k in fields))
        return row, metric, complete
    qualified = (result['status'] == 'completed' and result.get('fixed4800') is True and result['step'] == 4800
        and result.get('initial_physical_metrics_exactly_match_baseline') is True
        and result.get('first_actual_fullN_gradient_check', {}).get('passed') is True
        and external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
        and result['cached_TRAIN_encoded_reads'] == result['cached_TRAIN_physical_reads'] == 1
        and all(result[k] == 0 for k in ['raw_GT_reads', 'DEV_GT_reads', 'scaler_fits', 'pretrained_checkpoint_reads'])
        and max(result.get('peak_allocated_bytes', 25*2**30), result.get('peak_reserved_bytes', 25*2**30)) <= 24*2**30
        and all(finite(result.get(k)) and result[k] <= 900 for k in ['training_epoch_seconds', 'training_monotonic_seconds'])
        and all(finite(external.get(k)) and external[k] <= 1200 for k in ['elapsed_epoch_seconds', 'elapsed_monotonic_seconds'])
        and all(point(result, step)[2] for step in (0, 40, 4800)))
    assert base['status'] == 'completed' and base['fixed4800'] and base_external['within_external_budget']
    assert all(point(base, step)[2] for step in (0, 40, 4800))
    rows = []
    for label, native in [('encoded_MSE', base), ('physical_relative_squared', result)]:
        for step in (0, 40, 4800):
            saved, metric, complete = point(native, step)
            row = dict(objective=label, step=step, physical_point_complete=complete,
                       U=metric.get(fields[0]) if complete else None,
                       S=metric.get(fields[1]) if complete else None,
                       hot=metric.get(fields[2]) if complete else None)
            row['report_primary'] = .5*(row['U']+row['S']) if complete else None
            row['physical_relative_squared'] = .5*(row['U']**2+row['S']**2) if complete else None
            for axis in ('epoch', 'monotonic'):
                seconds = saved.get('training_'+axis+'_seconds_before_readout')
                row['training_'+axis+'_minutes'] = seconds/60 if finite(seconds) else None
            rows.append(row)
    baseline_row, new_row = rows[2], rows[5]
    gains = {k: baseline_row[k]-new_row[k] for k in ('U', 'S', 'hot', 'report_primary')} if qualified else None
    output = d.path('results/geobc_physical_loss_pair'); output.mkdir(parents=True, exist_ok=True)
    assert not (output/'manifest.json').exists(), 'One export, previous receipt preserved'
    report = dict(status='completed', scientific_qualification=qualified, rows=rows,
        final4800_absolute_gain=gains, case_key=base['selected']['case_key'], arm=base['selected']['arm'],
        case_denominator=1, same_case_updates=4800, actual_GT_PT_model_GPU_reads=0,
        source_evidence={'baseline':baseline_binding, 'new_summary':d.identity(folder/'summary.json', hashed=True),
                         'new_external':d.identity(folder/'external_terminal.json', hashed=True)},
        costs={label:{k:outer.get(k) for k in ['elapsed_epoch_seconds','elapsed_monotonic_seconds']}
               for label,outer in [('encoded_MSE',base_external),('physical_relative_squared',external)]},
        failure=result.get('error'), final_solver_selected=False,
        limits=['One selected known TRAIN case; no unseen-CAD or final-test evidence',
                'Changing objective also changes gradient magnitudes under the same optimizer',
                'Initial physical scalar metrics match exactly; no historical full-weight hash claim',
                'Original signed U mm/S MPa; no off-node or source stress-export semantics claim'])
    text = '# 固定单病例：只改变训练损失\n\n'
    text += f"FNO9／{report['case_key']}，同初始化、同Adam、同4800次单例更新。" + ('对照完成。\n\n' if qualified else '阶段未合格，不给完整对照结论。\n\n')
    text += '| 损失 | 位移误差 | signed应力误差 | 训练分钟UTC／mono |\n|---|---:|---:|---:|\n'
    def number(v): return f'{v:.2f}' if v is not None else '缺失'
    def percent(v): return f'{100*v:.2f}%' if v is not None else '缺失'
    for label, row in [('原归一化MSE', baseline_row), ('物理相对平方误差', new_row)]:
        text += f"| {label} | {percent(row['U'])} | {percent(row['S'])} | {number(row['training_epoch_minutes'])}／{number(row['training_monotonic_minutes'])} |\n"
    if qualified:
        text += f"\n新损失相对原损失：位移下降{100*gains['U']:.2f}、应力下降{100*gains['S']:.2f}个百分点（负数表示变差）。\n"
    text += '\n仅一个已知训练病例；改变损失也改变梯度尺度。结果用于选择后续共享训练方案，不能据此宣称泛化改善、Hash胜负或最终求解器。0／40端点、热点、实际完整阶段成本及来源见metrics.json。\n'
    (output/'BRIEF.md').write_text(text, encoding='utf-8')
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'metrics.csv').write_text(stream.getvalue(), encoding='utf-8')
    os.environ['MPLCONFIGDIR'] = str(output/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    for axis, field, title in zip(axes, ['U', 'S'], ['Displacement', 'Signed stress']):
        for i, row in enumerate([baseline_row, new_row]):
            value = row[field]
            if value is None: axis.text(i, .05, 'Missing', ha='center', transform=axis.get_xaxis_transform())
            else:
                axis.bar(i, 100*value, color=['#637b90', '#338975'][i])
                axis.annotate(f'{100*value:.2f}%', (i, 100*value), xytext=(0, 4), textcoords='offset points', ha='center')
        axis.set_xticks([0, 1], ['Encoded MSE', 'Physical relative squared'], rotation=10)
        axis.set_ylabel('Physical relative L2 (%)'); axis.set_title(title); axis.set_ylim(bottom=0); axis.margins(y=.2)
    figure.suptitle(report['arm']+' / '+report['case_key']+' / same 4800 case visits'+('' if qualified else ' / incomplete stage'))
    figure.savefig(output/'comparison.png', dpi=160); plt.close(figure)
    report.update(CPU_export_seconds=time.monotonic()-start, required_PNG_rendered=(output/'comparison.png').stat().st_size>0,
                  visual_QA='pending_actual_view')
    d.write(output/'metrics.json', report)
    d.write(output/'manifest.json', dict(status='completed', scientific_qualification=qualified,
        scientific_digest=plan['scientific_digest'], source_sha256=d.sha(__file__),
        actual_GT_PT_model_GPU_reads=0, required_PNG_rendered=report['required_PNG_rendered'],
        visual_QA='pending_actual_view', CPU_export_seconds=report['CPU_export_seconds']))
    print({'qualified':qualified, 'final4800': [baseline_row,new_row], 'CPU_seconds':report['CPU_export_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: run()
