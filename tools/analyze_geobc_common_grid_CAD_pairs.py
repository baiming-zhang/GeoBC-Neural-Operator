"""CPU scalar-only paired CAD analysis of completed common-grid4800 results.

Resampling describes this known development pool, not independent final-test
confidence, seed uncertainty or a capacity-independent architecture claim.
"""
import argparse
import math
import time


def analyze():
    started = (time.time(), time.monotonic())
    from geobc import available_cad_pilot_data as d
    from tools.export_geobc_exposure_comparison import terminal, complete
    plan = d.read('notes/geobc_common_grid_readout_plan.json')
    summary, external = terminal(plan)
    assert summary['status'] == 'completed' and summary['protocol_qualified']
    assert external['within_external_budget'] and external['exit_code'] == 0 and not external['external_timeout']
    assert summary['committed_finite_predictions'] == 120 and summary['owned_GT_array_reads'] == 60
    keys = plan['scientific']['case_keys']; cads = sorted({k.split('/')[0] for k in keys})
    assert len(keys) == 60 and len(cads) == 20
    assert set(keys) == {c+'/'+load for c in cads for load in ('ver', 'hor', 'tor')}
    arms = {name: summary['candidates'][name]['full60'] for name in ('FNO9', 'PP256')}
    for metric in arms.values(): complete(metric, keys)
    fields = dict(U='u_relative_l2', S='stress_relative_l2', hot='stress_hotspot_relative_l2')
    paired = {}
    for cad in cads:
        paired[cad] = {name: sum(arms['PP256']['per_case'][cad+'/'+load][field]
                                   - arms['FNO9']['per_case'][cad+'/'+load][field]
                              for load in ('ver', 'hor', 'tor'))/3 for name, field in fields.items()}
        paired[cad]['joint'] = .5*(paired[cad]['U']+paired[cad]['S'])
    assert all(math.isfinite(x) for row in paired.values() for x in row.values())
    import numpy as np
    names = ['U', 'S', 'hot', 'joint']
    values = np.asarray([[paired[cad][n] for n in names] for cad in cads], dtype=np.float64)
    means = values.mean(axis=0)
    assert math.isclose(means[3], arms['PP256']['selection_score']-arms['FNO9']['selection_score'], abs_tol=1e-12)
    rng = np.random.default_rng(2026)
    samples = rng.integers(0, 20, size=(50000, 20))
    bootstrap = values[samples].mean(axis=1)
    bounds = np.quantile(bootstrap, [.025, .975], axis=0)
    leave_one_out = (values.sum(axis=0)[None, :]-values)/19
    results = {name: dict(mean_gap_percentage_points=float(means[i]*100),
        known_pool_CAD_bootstrap95_range_percentage_points=[float(x*100) for x in bounds[:, i]],
        FNO_better_CAD_count=int((values[:, i] > 0).sum()), PP_better_CAD_count=int((values[:, i] < 0).sum()),
        exact_tie_CAD_count=int((values[:, i] == 0).sum()),
        leave_one_CAD_out_gap_range_percentage_points=[float(leave_one_out[:, i].min()*100), float(leave_one_out[:, i].max()*100)])
        for i, name in enumerate(names)}
    folder = d.path(plan['output'])
    report = dict(status='completed_existing_scalar_CAD_pair_analysis', parent_scientific_digest=plan['scientific_digest'],
        source_sha256=d.sha('tools/analyze_geobc_common_grid_CAD_pairs.py'),
        bindings={n: d.identity(folder/file, hashed=True) for n, file in [('summary', 'summary.json'), ('external', 'external_terminal.json')]},
        case_denominator=60, CAD_denominator=20, exposures_per_case=40,
        direction='PP minus FNO relative-L2 percentage points; positive favors FNO',
        joint_score='.5(case-mean U relativeL2 + case-mean signed S relativeL2); CAD averages all three loads',
        bootstrap_method='50000 paired CAD resamples with replacement, NumPy default_rng2026, percentile2.5/97.5; keep loads together',
        estimates=results, per_CAD20=paired,
        scientific_scope='Describes existing known availability-conditional development20CAD only; not independent final-test or random-seed confidence and not an architecture upper bound',
        tuning_changes=0, numerical_input_GT_PT_reads=0, GPU_or_new_inference=0,
        elapsed_epoch_seconds=time.time()-started[0], elapsed_monotonic_seconds=time.monotonic()-started[1])
    d.write('notes/geobc_common_grid_CAD_pair_analysis.json', report)
    output = d.path('results/geobc_common_grid_CAD_pair_analysis'); output.mkdir(parents=True, exist_ok=True)
    d.write(output/'summary.json', report)
    text = '# 当前共同基线的CAD配对差距\n\n'
    text += ('相同R64表示、约4.53M参数、TRAIN40 CAD×三载荷、每病例40次曝光下，'
             'FNO在已知开发20 CAD的联合位移／应力评分上领先 **'+str(results['joint']['FNO_better_CAD_count'])+'/20 CAD**。\n\n')
    for name, label in [('U', '位移'), ('S', '应力'), ('joint', '联合评分')]:
        r = results[name]; low, high = r['known_pool_CAD_bootstrap95_range_percentage_points']
        text += f'- {label}平均领先 **{r["mean_gap_percentage_points"]:.1f}个百分点**；按CAD配对重采样的95%范围为{low:.1f}–{high:.1f}个百分点。\n'
    r = results['joint']; low, high = r['leave_one_CAD_out_gap_range_percentage_points']
    text += f'\n逐一移除一个CAD后，联合差距仍为{low:.1f}–{high:.1f}个百分点。因此当前领先不由单个CAD决定。\n\n'
    text += ('这描述当前已知开发池和单seed配方。三载荷作为同一CAD的一个单元重采样；'
             '没有把数百万节点当作独立样本。它不代表最终测试、跨seed稳定性或架构上限。'
             '++延长训练和原生表示仍需真实对照，不能根据这项统计提前排除。\n\n')
    text += f'本次只读取已保存标量，CPU计算{report["elapsed_epoch_seconds"]:.3f}秒；没有新推理、真实数组／标签读取或调参。\n'
    (output/'BRIEF.md').write_text(text, encoding='utf-8')
    print({'status': report['status'], 'seconds': report['elapsed_epoch_seconds'], 'estimates': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--analyze', action='store_true')
    if parser.parse_args().analyze: analyze()
    else: print('{"status":"source_only","reads":0,"writes":0,"GPU_execution":0}')
