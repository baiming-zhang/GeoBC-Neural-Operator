"""CPU scalar report for fixed40/100/200 exposures; never opens arrays or models."""
import argparse
import ast
import csv
import io
import math
import time
from geobc import available_cad_pilot_data as d

OWN = 'tools/export_geobc_exposure_comparison.py'
REVIEW = 'notes/geobc_exposure_exporter_root_review.json'
OUTPUT = 'results/geobc_exposure_comparison'
ARMS = ('FNO9', 'PP256')
MEANS = {'U': 'case_mean_u_relative_l2', 'S': 'case_mean_stress_relative_l2',
         'hot': 'case_mean_stress_hotspot_relative_l2'}
CASE = {'U': 'u_relative_l2', 'S': 'stress_relative_l2', 'hot': 'stress_hotspot_relative_l2'}


def freeze():
    assert not d.path(REVIEW).exists()
    ast.parse(d.path(OWN).read_text(encoding='utf-8'))
    plan = d.read('notes/geobc_exposure_readout_plan.json')
    baseline = d.read('notes/geobc_common_grid_readout_plan.json')
    parent = d.read(baseline['scientific']['parent_plan'])
    pins = {name: d.identity(d.path(folder)/file, hashed=True) for name, folder, file in [
        ('baseline_training', parent['output'], 'summary.json'),
        ('baseline_training_external', parent['output'], 'external_terminal.json'),
        ('baseline_development', baseline['output'], 'summary.json'),
        ('baseline_development_external', baseline['output'], 'external_terminal.json')]}
    value = dict(passed=False, whole_SOURCE_reviewed=False, actual_export_released=False,
        exporter_source_sha256=d.sha(OWN), scientific_digest=plan['scientific_digest'],
        baseline_bindings=pins, baseline_plan=baseline['scientific_digest'],
        preflight='Both actual stages terminal/closed/exact exits; all240-before-GT60 and complete physical endpoints required for ranking',
        reads='Saved scalar JSON only; no GT/arrays/PT/GPU/inference', actual_export=0)
    d.write(REVIEW, value)
    print({'status': 'source_frozen_actual_export0', 'sha256': value['exporter_source_sha256']})


def terminal(plan):
    folder = d.path(plan['output'])
    summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    return summary, external


def complete(metric, keys):
    assert metric['all_cases_complete'] and metric['fixed_denominator'] == len(keys)
    assert set(metric['per_case']) == set(keys)
    for key in keys:
        row = metric['per_case'][key]
        assert row is not None
        assert all(isinstance(row.get(name), (int, float)) and math.isfinite(row[name]) and row[name] >= 0
                   for name in CASE.values())
    assert all(isinstance(metric[name], (int, float)) and math.isfinite(metric[name]) for name in MEANS.values())


def paired(left, right, keys):
    cads = sorted({k.split('/')[0] for k in keys}); assert len(cads) == 20
    rows = {cad: {name: sum(left['per_case'][cad+'/'+load][field] - right['per_case'][cad+'/'+load][field]
                    for load in ('ver', 'hor', 'tor'))/3 for name, field in CASE.items()} for cad in cads}
    return dict(direction='left minus right; negative favors left', per_CAD20=rows,
                better_CAD_counts={name: sum(row[name] < 0 for row in rows.values()) for name in CASE},
                mean_difference={name: sum(row[name] for row in rows.values())/20 for name in CASE})


def export():
    started = time.monotonic(); review = d.read(REVIEW)
    assert review['passed'] and review['whole_SOURCE_reviewed'] and review['actual_export_released']
    assert review['exporter_source_sha256'] == d.sha(OWN)
    plan = d.read('notes/geobc_exposure_readout_plan.json'); parent = d.read(plan['scientific']['parent_plan'])
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    assert all(d.sha(p) == h for p, h in parent['source_pins'].items())
    development, outer = terminal(plan); training, train_outer = terminal(parent)
    baseline_plan = d.read('notes/geobc_common_grid_readout_plan.json')
    assert baseline_plan['scientific_digest'] == review['baseline_plan']
    for pin in review['baseline_bindings'].values():
        assert d.identity(pin['path'], hashed=True) == pin
    base_training = d.read(review['baseline_bindings']['baseline_training']['path'])
    base_development = d.read(review['baseline_bindings']['baseline_development']['path'])
    assert base_training['both4800'] and base_development['protocol_qualified']
    assert baseline_plan['scientific']['case_keys'] == plan['scientific']['case_keys']
    keys = plan['scientific']['case_keys']; train_keys = parent['scientific']['TRAIN_keys']
    assert len(keys) == 60 and len(train_keys) == 120
    good = (training['status'] == development['status'] == 'completed' and training.get('both24000') is True
        and development.get('protocol_qualified') is True and train_outer['within_external_budget']
        and outer['within_external_budget'] and train_outer['exit_code'] == outer['exit_code'] == 0
        and not train_outer['external_timeout'] and not outer['external_timeout'])
    barrier = None
    if good:
        from tools.evaluate_geobc_exposure_endpoints import parent_qualified
        parent_qualified(plan, d.read(plan['root_review']))
        barrier = d.read(d.path(plan['output'])/'prediction_barrier.json')
        expected = {c['name']+':'+key for c in plan['scientific']['candidates'] for key in keys}
        assert len(expected) == barrier['finite_predictions'] == barrier['expected'] == 240
        assert set(barrier['inventory']) == set(development['prediction_inventory']) == expected
        assert barrier['all_cartesian_entries'] and barrier['owned_GT_reads_before_barrier'] == 0
        assert development['committed_finite_predictions'] == 240 and development['owned_GT_array_reads'] == 60
        assert development['raw_GT_reads'] == development['sealed_DEV_FINAL_reads'] == development['training_updates'] == development['scaler_fits'] == 0
        assert development['fixed_prediction_denominator'] == 240 and development['fixed_case_denominator'] == 60
        assert max(outer['elapsed_epoch_seconds'], outer['elapsed_monotonic_seconds']) <= 3600
        for candidate in plan['scientific']['candidates']:
            row = development['candidates'][candidate['name']]
            assert row['cases'] == 60 and max(row['peak_allocated_bytes'], row['peak_reserved_bytes']) <= 24*2**30
            complete(row['full60'], keys)
            for load in ('ver', 'hor', 'tor'):
                complete(row['per_load20'][load], [k for k in keys if k.endswith('/'+load)])
    rows = []; per_development = {}; per_training = {}; per_load = {}
    for arm in ARMS:
        original = base_training['arms'][arm]; base_train = original['TRAIN120']
        base_dev = base_development['candidates'][arm]['full60']
        complete(base_train, train_keys); complete(base_dev, keys)
        for step in (4800, 12000, 24000):
            name = arm+'_step'+str(step); base = step == 4800
            observed = training.get('arms', {}).get(arm, {}).get('endpoints', {}).get(str(step), {})
            train = base_train if base else observed.get('TRAIN120', {})
            dev = base_dev if base else development.get('candidates', {}).get(name, {}).get('full60', {}) if good else {}
            if train.get('all_cases_complete'): complete(train, train_keys)
            original_epoch = original['training_elapsed_epoch_seconds']
            original_mono = original['training_elapsed_monotonic_seconds']
            # Endpoint cost accounts baseline + extension before readout + this readout.
            # Shared cache/setup and whole guard costs remain separate; never zero-fill missing values.
            extension_epoch = observed.get('training_epoch_seconds_before_readout')
            extension_mono = observed.get('training_monotonic_seconds_before_readout')
            accounted_epoch = original_epoch if base else (original_epoch+extension_epoch+observed['epoch_seconds'] if extension_epoch is not None else None)
            accounted_mono = original_mono if base else (original_mono+extension_mono+observed['monotonic_seconds'] if extension_mono is not None else None)
            inference = base_development['candidates'][arm]['inference_seconds'] if base else development.get('candidates', {}).get(name, {}).get('inference_seconds')
            row = dict(model=arm, step=step, per_case_exposures=step//120, qualified=True if base else good,
                **{'TRAIN_'+metric: train.get(field) for metric, field in MEANS.items()},
                **{'DEV_'+metric: dev.get(field) for metric, field in MEANS.items()},
                primary=dev.get('selection_score'),
                cumulative_accounted_epoch_minutes=accounted_epoch/60 if accounted_epoch is not None else None,
                cumulative_accounted_monotonic_minutes=accounted_mono/60 if accounted_mono is not None else None,
                original_training_epoch_seconds=original_epoch, extension_training_before_readout_epoch_seconds=extension_epoch,
                endpoint_TRAIN_readout_epoch_seconds=observed.get('epoch_seconds'),
                model_inference_seconds=inference, parameters=original['counts']['total'])
            rows.append(row); per_training[name] = train or None; per_development[name] = dev or None
            per_load[name] = base_development['candidates'][arm]['per_load20'] if base else development.get('candidates', {}).get(name, {}).get('per_load20') if good else None
    leaders = []; best_TS = []; pairs = {}; deltas = {}
    if good:
        minimum = min(r['primary'] for r in rows); leaders = [r for r in rows if r['primary'] == minimum]
        ts = [r for r in rows if r['model'] == 'PP256']; best_TS = [r for r in ts if r['primary'] == min(t['primary'] for t in ts)]
        for step in (4800, 12000, 24000):
            pairs[str(step)] = paired(per_development['FNO9_step'+str(step)], per_development['PP256_step'+str(step)], keys)
        for arm in ARMS:
            for step in (12000, 24000):
                name = arm+'_step'+str(step)
                deltas[name] = paired(per_development[name], per_development[arm+'_step4800'], keys)
    evidence = dict(review['baseline_bindings'])
    for name, folder in [('extended_training', parent['output']), ('extended_development', plan['output'])]:
        for field in ('summary', 'external_terminal'):
            evidence[name+'_'+field] = d.identity(d.path(folder)/(field+'.json'), hashed=True)
    if good: evidence['barrier'] = d.identity(d.path(plan['output'])/'prediction_barrier.json', hashed=True)
    result = dict(status='completed', scientific_qualification=good, rows=rows,
        eligible_leaders=[r['model']+'_step'+str(r['step']) for r in leaders],
        best_Transolver=[r['model']+'_step'+str(r['step']) for r in best_TS], all_ties_retained=True,
        TRAIN120=per_training, DEV60=per_development, per_load20=per_load,
        paired_CAD20_FNO_minus_PP=pairs, paired_CAD20_extended_minus_4800=deltas,
        source_evidence=evidence, shared_GPU_cache=training.get('GPU_cache'),
        whole_training_epoch_seconds=train_outer.get('elapsed_epoch_seconds'),
        whole_training_monotonic_seconds=train_outer.get('elapsed_monotonic_seconds'),
        whole_development_epoch_seconds=outer.get('elapsed_epoch_seconds'),
        whole_development_monotonic_seconds=outer.get('elapsed_monotonic_seconds'),
        prediction_denominator=240, development_case_denominator=60,
        failure_reasons=[v.get('error') for v in (training, development) if v.get('error')],
        numeric_array_validation_performed_by_exporter=False, GT_PT_array_model_GPU_reads=0,
        final_solver_selected=False, FINAL_labels_read=0,
        limitations=['One seed; previously known development60; availability-conditional CAD pool, not final testing.',
            'Fixed100/200 continuation preserves Adam/RNG/order and loss; cache changes execution cost, not intended mathematics.',
            'Cumulative endpoint costs account original training + extension to readout end; shared cache/setup and outer wall cost are reported separately.',
            'Both observed UTC and monotonic costs are retained without rescaling or attributing their discrepancy.',
            'Signed S source is unchanged; independent off-node FEM accuracy and full stress-export semantics remain unresolved.'])
    output = d.path(OUTPUT); assert not (output/'manifest.json').exists(), 'Existing export preserved'
    output.mkdir(parents=True, exist_ok=True)
    for name, pin in evidence.items():
        assert d.identity(pin['path'], hashed=True) == pin
        d.write(output/'source_evidence'/(name+'.json'), d.read(pin['path']))
    text = '# 相同病例：40／100／200次曝光\n\n'
    text += ('固定端点开发比较已完成；当前领先：'+ '、'.join(result['eligible_leaders'])+'。最终求解器尚未确定。\n\n') if good else '本次存在技术或协议负结果，取消新端点排名；完整240预测／60病例分母及已有4800参照保留。\n\n'
    shown = []
    for row in leaders[:1]+best_TS[:1]+[rows[0]]:
        if row not in shown: shown.append(row)
    if not good: shown = [rows[0], rows[3]]
    text += '| 方法／每例曝光 | 训练U／S | 开发U／S | 累计训练分钟（UTC） | 60例模型推理秒 |\n|---|---:|---:|---:|---:|\n'
    def fmt(value, scale=1): return f'{value*scale:.1f}' if value is not None else '缺失'
    for row in shown:
        text += f"| {row['model']}／{row['per_case_exposures']} | {fmt(row['TRAIN_U'],100)}%／{fmt(row['TRAIN_S'],100)}% | {fmt(row['DEV_U'],100)}%／{fmt(row['DEV_S'],100)}% | {fmt(row['cumulative_accounted_epoch_minutes'])} | {fmt(row['model_inference_seconds'])} |\n"
    if good:
        for arm in ARMS:
            first = next(r for r in rows if r['model']==arm and r['step']==4800)
            last = next(r for r in rows if r['model']==arm and r['step']==24000)
            text += f"\n{arm}从40到200曝光：训练U／S变化{fmt(last['TRAIN_U']-first['TRAIN_U'],100)}／{fmt(last['TRAIN_S']-first['TRAIN_S'],100)}个百分点；开发U／S变化{fmt(last['DEV_U']-first['DEV_U'],100)}／{fmt(last['DEV_S']-first['DEV_S'],100)}个百分点。负值表示改善。\n"
    text += '\n相同TRAIN120、R64／native25查询头、约453万参数、原encoded4损失与TRAIN-only scaler。训练改善与开发改善分别判断；Hash单例亚1%不能与跨CAD误差直接混比。全部六端点、热点／最差病例／每载荷20／逐CAD配对和UTC／mono成本见JSON及CSV。\n\n'
    text += '所有240份有限全原节点预测在共享读取既有GT60之前提交；本报告只读保存的标量，不重复推理或读取数组。开发标签历史已知，reserved FINAL仍未读取，不宣布最终泛化或离节点物理精度。\n' if good else '负结果不使用有效子集均值给新端点排名；需先处理直接技术原因。报告只读已有标量。\n'
    (output/'BRIEF.md').write_text(text, encoding='utf-8')
    stream=io.StringIO();writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'metrics.csv').write_text(stream.getvalue(), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax, metric in zip(axes, ('U','S')):
        for arm, color in [('FNO9','#307eae'),('PP256','#bf6a37')]:
            points=[r for r in rows if r['model']==arm]
            for scope,style in [('TRAIN','--'),('DEV','-')]:
                valid=[r for r in points if r[scope+'_'+metric] is not None]
                if valid: ax.plot([r['per_case_exposures'] for r in valid], [100*r[scope+'_'+metric] for r in valid],
                                 style,marker='o',color=color,label=arm+' '+scope)
        ax.set_title('Displacement relative L2' if metric=='U' else 'Signed stress relative L2')
        ax.set_xlabel('Per-case exposures');ax.set_ylabel('%');ax.set_xticks([40,100,200]);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    fig.suptitle('Fixed exposure endpoints: training and known development' if good else 'Nonqualified: no new development ranking')
    fig.savefig(output/'exposure.png',dpi=160);plt.close(fig)
    assert (output/'exposure.png').stat().st_size > 0
    result['CPU_export_seconds']=time.monotonic()-started;d.write(output/'metrics.json',result)
    d.write(output/'manifest.json',dict(status='completed',actual_export=True,scientific_qualification=good,
        scientific_digest=plan['scientific_digest'],exporter_source_sha256=d.sha(OWN),
        artifacts={name:d.identity(output/name,hashed=True) for name in ('BRIEF.md','metrics.csv','metrics.json','exposure.png')},
        CPU_seconds=result['CPU_export_seconds'],GT_PT_array_model_GPU_reads=0))
    print({'status':'completed','scientific_qualification':good,'CPU_seconds':result['CPU_export_seconds']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--export',action='store_true')
    args=parser.parse_args();assert not (args.freeze and args.export)
    if args.freeze: freeze()
    if args.export: export()
