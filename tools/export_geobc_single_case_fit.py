"""CPU scalar report of one selected TRAIN case; no PT/GT/array/model reads."""
import argparse
import csv
import io
import math
import os
import time
from geobc import available_cad_pilot_data as d

OWN = 'tools/export_geobc_single_case_fit.py'
REVIEW = 'notes/geobc_single_case_exporter_root_review.json'


def run():
    started = time.monotonic(); review = d.read(REVIEW)
    plan = d.read('notes/geobc_single_case_fit_plan.json')
    assert review['passed'] and review['actual_export_released']
    assert review['exporter_source_sha256'] == d.sha(OWN)
    assert review['scientific_digest'] == plan['scientific_digest'] == d.digest(plan['scientific'])
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    folder = d.path(plan['output']); summary = d.read(folder/'summary.json')
    external = d.read(folder/'external_terminal.json')
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert all(d.exact_process_exited(external[k]) for k in ('guard_process','child_process'))
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    selected = summary['selected']; live_review = d.read(plan['root_review'])
    assert live_review['actual_execution_released'] and live_review['selected'] == selected
    sources = {'native':d.identity(folder/'summary.json',hashed=True),
               'external':d.identity(folder/'external_terminal.json',hashed=True)}
    # Original reference is the actually completed full TRAIN120 scalar readout.
    binding = live_review['upstream_bindings']['training']
    assert d.identity(binding['summary']['path'],hashed=True) == binding['summary']
    parent = d.read(binding['summary']['path']); arm = selected['arm']; key = selected['case_key']
    shared = parent['arms'][arm]['TRAIN120']['per_case'][key]
    assert shared == selected['shared_training_case'] and parent['arms'][arm]['case_exposure_counts'][key] == 40
    fields = ['u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2']

    def complete(metric):
        return (isinstance(metric,dict) and metric.get('nodes') == shared['nodes'] and
                all(isinstance(metric.get(k),(float,int)) and math.isfinite(metric[k]) and metric[k] >= 0 for k in fields))

    assert complete(shared)
    endpoints = summary.get('endpoints',{})
    costs = [summary.get(k) for k in ('training_epoch_seconds','training_monotonic_seconds')]
    costs += [external.get(k) for k in ('elapsed_epoch_seconds','elapsed_monotonic_seconds')]
    cost_known = all(isinstance(v,(float,int)) and math.isfinite(v) and v >= 0 for v in costs)
    good = (summary['status'] == 'completed' and summary.get('fixed4800') is True and summary['step'] == 4800
            and external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
            and summary['first_actual_fullN_gradient_check']['passed']
            and summary['cached_TRAIN_encoded_reads'] == summary['cached_TRAIN_physical_reads'] == 1
            and summary['raw_GT_reads'] == summary['DEV_GT_reads'] == summary['scaler_fits'] == summary['pretrained_checkpoint_reads'] == 0
            and cost_known and max(summary['training_epoch_seconds'],summary['training_monotonic_seconds']) <= 18000
            and max(external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds']) <= 20000
            and max(summary['peak_allocated_bytes'],summary['peak_reserved_bytes']) <= 24*2**30
            and all(complete(endpoints.get(str(step),{}).get('physical')) and
                    endpoints[str(step)]['case_exposures'] == endpoints[str(step)]['optimizer_steps'] == step
                    for step in (0,40,4800)))
    rows = [{'role':'shared4800_case40','fixed_endpoint_updates':4800,'case_exposures':40,'total_updates':4800,
             'U':shared[fields[0]],'S':shared[fields[1]],'hot':shared[fields[2]],
             'train_epoch_minutes':parent['arms'][arm]['training_elapsed_epoch_seconds']/60,
             'train_monotonic_minutes':parent['arms'][arm]['training_elapsed_monotonic_seconds']/60,
             'point_available':True,'complete_single_case_stage':good}]
    for step in (0,40,4800):
        point = endpoints.get(str(step),{}); metric = point.get('physical')
        available = complete(metric) and point.get('case_exposures') == point.get('optimizer_steps') == step
        def minutes(name):
            value = point.get(name)
            return value/60 if isinstance(value,(int,float)) and math.isfinite(value) and value >= 0 else None
        rows.append({'role':'independent'+str(step),'fixed_endpoint_updates':step,
            'case_exposures':point.get('case_exposures'),'total_updates':point.get('optimizer_steps'),
            'U':metric[fields[0]] if available else None,'S':metric[fields[1]] if available else None,
            'hot':metric[fields[2]] if available else None,
            'train_epoch_minutes':minutes('training_epoch_seconds_before_readout'),
            'train_monotonic_minutes':minutes('training_monotonic_seconds_before_readout'),
            'point_available':available,'complete_single_case_stage':good})
    gains = None
    if good:
        gains = {field:(rows[0][field]-rows[3][field])/rows[0][field] if rows[0][field] > 0 else None for field in ('U','S','hot')}
    output = d.path('results/geobc_single_case_fit'); output.mkdir(parents=True,exist_ok=True)
    assert not (output/'manifest.json').exists(), 'Previous export preserved; no automatic overwrite'
    result = {'status':'completed','scientific_qualification':good,'case_key':key,'arm':arm,
        'single_case_denominator':1,'fixed_independent_endpoints':[0,40,4800],'rows':rows,
        'qualified4800_relative_gain_vs_shared':gains,'source_evidence':sources,
        'shared_reference_identity':binding['summary'],'selection':selected,
        'whole_epoch_seconds':external.get('elapsed_epoch_seconds'),
        'whole_monotonic_seconds':external.get('elapsed_monotonic_seconds'),
        'failure':summary.get('error'),'GT_PT_model_GPU_reads':0,'final_solver_selected':False,
        'exporter_numeric_array_validation_performed':False,
        'interpretation':['One worst known TRAIN case selected by frozen rule, not population-average or final test',
            '40 compares same-case exposure but different optimizer history;4800 compares total updates but different case exposure',
            'Resident one-case IO differs from shared loading; times are experiment costs, not solver inference speed',
            'Same asinh scaler/joint4 loss/model/init; no exclusive capacity/optimization/representation attribution',
            'Negative stages may retain complete saved diagnostic points; gains and complete4800 conclusions are cancelled',
            'Raw signedS MPa and U vector mm; no independent off-node FEM or source stress-export semantics claim']}
    def percent(value): return f'{100*value:.3g}%' if value is not None else '缺失'
    text = '# 单病例：共享训练与独立拟合\n\n'
    text += f"模型{arm}，已知TRAIN病例{key}。" + ('固定4800诊断完成。\n\n' if good else '整体诊断未合格；保存点仅描述已发生的读出，不给完整4800结论。\n\n')
    text += '| 训练方式 | 该例曝光 | 位移误差 | signed应力误差 | 训练分钟（UTC／mono） |\n|---|---:|---:|---:|---:|\n'
    for index,label in [(0,'共享4800更新'),(2,'独立40更新'),(3,'独立4800更新')]:
        row = rows[index]
        def number(v):return f'{v:.2f}' if v is not None else '缺失'
        exposure = row['case_exposures'] if row['case_exposures'] is not None else '缺失'
        text += f"| {label} | {exposure} | {percent(row['U'])} | {percent(row['S'])} | {number(row['train_epoch_minutes'])}／{number(row['train_monotonic_minutes'])} |\n"
    text += '\n共享训练覆盖120个病例；独立训练只覆盖这一例。40端点优化历史不同，4800端点该例曝光不同，因此不能作单因素因果归因。训练成本包含各自IO与检查点规则，独立常驻输入也不是推理速度收益。\n\n'
    text += '这个诊断判断现有模型能否拟合已知病例，不证明新几何泛化或最终求解器优胜。未训练0端点、热点及完整阶段成本见metrics.json/CSV。\n'
    (output/'BRIEF.md').write_text(text,encoding='utf-8')
    stream = io.StringIO(); writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'metrics.csv').write_text(stream.getvalue(),encoding='utf-8')
    os.environ['MPLCONFIGDIR'] = str(output/'.mplconfig')
    plot = False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
        shown = [rows[i] for i in (0,2,3)]; labels = ['Shared /40 case visits','Independent40','Independent4800']
        for axis,field,title in zip(axes,['U','S'],['Displacement','Signed stress']):
            for index,row in enumerate(shown):
                value = row[field]
                if value is None: axis.text(index,.02,'Missing',ha='center',transform=axis.get_xaxis_transform())
                else:
                    axis.bar(index,value*100,color=['#637b90','#b9824e','#338975'][index])
                    axis.annotate(f'{value*100:.3g}%',(index,value*100),xytext=(0,4),textcoords='offset points',ha='center')
            axis.set_xticks(range(3),labels,rotation=12);axis.set_ylabel('Physical relative L2 (%)')
            axis.set_title(title);axis.set_ylim(bottom=0);axis.margins(y=.18)
        figure.suptitle(arm+' / '+key+' / one known TRAIN case'+('' if good else ' / incomplete stage'))
        figure.savefig(output/'comparison.png',dpi=160);plt.close(figure)
        plot = (output/'comparison.png').stat().st_size > 0
    except Exception as exc:
        result['plot_error'] = type(exc).__name__+': '+str(exc)
    result.update(status='completed' if plot else 'completed_incomplete',required_PNG_rendered=plot,
                  visual_QA='pending_actual_view',CPU_export_seconds=time.monotonic()-started)
    d.write(output/'metrics.json',result)
    d.write(output/'manifest.json',{'status':result['status'],'scientific_qualification':good,
        'required_PNG_rendered':plot,'visual_QA':'pending_actual_view','actual_GT_PT_model_GPU_reads':0,
        'scientific_digest':plan['scientific_digest'],'exporter_sha256':d.sha(OWN),'CPU_export_seconds':result['CPU_export_seconds']})
    print({'status':result['status'],'case':key,'qualified':good,'CPU_seconds':result['CPU_export_seconds']})
    return 0 if plot else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export',action='store_true')
    if parser.parse_args().export: raise SystemExit(run())
