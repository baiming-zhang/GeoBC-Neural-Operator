"""One scalar-only matched20 knownDEV report; no PT/GT arrays/inference."""
import argparse
import csv
import time

OWN = 'tools/export_geobc_matched20_readout_v79.py'
REVIEW = 'notes/geobc_matched20_readout_v79_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2',
          'case_mean_stress_hotspot_relative_l2')
METHODS = ('FNO100_PHYSICAL_6000', 'FNO_GLOBAL_HASH20', 'TS_CONTINUOUS20')
LABELS = ('FNO20', '条件Hash20', '原Transolver连续解码器20')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.matched20_readout_contract_v79 import PLAN, SOURCE_REVIEW
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical
    p, a, e, binding = terminal(PLAN)
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN)
    assert marker['training_SCI'] == p['scientific']['training_SCI']
    assert marker['source_review_identity'] == d.identity(SOURCE_REVIEW, hashed=True)
    sr = d.read(SOURCE_REVIEW)
    assert p['source_pins'] == sr['source_pins']
    assert a['protocol_qualified'] and a['prediction_barrier_complete']
    assert a['committed_finite_predictions'] == 120
    assert a['owned_GT_read_attempts'] == a['owned_GT_array_reads'] == 60
    assert a['checkpoint_read_attempts'] == a['checkpoint_reads'] == a['checkpoint_body_hash_reads'] == 2
    assert set(a['strict_loaded_checkpoint_tensor_state_verified']) == set(METHODS[1:])
    assert all(a['strict_loaded_checkpoint_tensor_state_verified'].values())
    assert a['raw_GT_reads'] == a['FINAL_numeric_reads'] == a['training_updates'] == a['scaler_fits'] == 0
    assert not a['final_solver_selected'] and not a['automatic_continuation']
    rows, keys = a['candidates'], p['scientific']['case_keys']
    assert len(keys) == len(set(keys)) == 60
    for row in rows.values():
        complete_physical(row['full60'], keys)
    for n in METHODS:
        assert rows[n]['case_exposure_count'] == 20
    assert rows[METHODS[0]]['reused_immutable_scalar_metrics']
    scores = {n: rows[n]['full60']['selection_score'] for n in METHODS}
    leaders = sorted(n for n, score in scores.items() if score == min(scores.values()))
    bm = rows[METHODS[0]]['full60']
    differences = {n: {f: 100*(rows[n]['full60'][f]-bm[f]) for f in FIELDS}
                   for n in METHODS[1:]}
    out = d.path('results/geobc_matched20_readout_v79')
    assert not (out/'manifest.json').exists(), 'One actual immutable export'
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# 同20遍求解器对比', '',
        '**固定20个开发CAD×三载荷、全原节点、原signed U(mm)/S(MPa)，相同TRAIN-only scaler。三个方案每个训练场均曝光20遍。**', '',
        '| 方法 | 位移误差 | 应力误差 | 本轮追加训练分钟 | 60场推理秒 |',
        '|---|---:|---:|---:|---:|']
    for n, label in zip(METHODS, LABELS):
        row = rows[n]; z = row['full60']
        added = f'{row["added_training_interval_epoch_seconds"]/60:.2f}' if n in METHODS[1:] else '—'
        lines.append(f'| {label} | {100*z[FIELDS[0]]:.2f}% | {100*z[FIELDS[1]]:.2f}% | {added} | {row["inference_epoch_seconds"]:.2f} |')
    lines += ['', '本轮同20遍中，位移／应力等权综合误差最低：'+
        '、'.join(LABELS[METHODS.index(n)] for n in leaders)+'。并列全部保留。', '']
    for n, label in zip(METHODS[1:], LABELS[1:]):
        diff = differences[n]
        lines.append(f'{label}相对FNO20：位移{diff[FIELDS[0]]:+.2f}、应力{diff[FIELDS[1]]:+.2f}、热点{diff[FIELDS[2]]:+.2f}个百分点；预声明开发集推进门通过：{a["DEV_gates_against20"][n]["passed"]}。')
    lines += ['', f'完整验证{e["elapsed_epoch_seconds"]:.2f}秒。两种新候选120份有限全节点预测先全部保存并重验，再共享读取已有60份物理标签各一次；旧FNO和其他历史参考只复用冻结标量。', '',
        '表中追加时间是Hash10→20与TS5→20的本轮区间，含原来记录的中间读出／保存，不能当累计总成本。两个父阶段、当前共享缓存、阶段成本和epoch／monotonic时间轴在JSON分别保留，不将未知成本填零。FNO的推理时间来自历史实际读出，未重新推理。', '',
        '**开发集已用于多轮迭代，且只有一个seed、来源可用性条件筛选；不能当最终盲测或宣布架构上限。** 相同曝光仍存在容量、精度与成本差异。没有自动40遍、FINAL调参或最终求解器选择。全60病例、每载荷20、逐CAD配对及热点保留在JSON／CSV。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        matched20_methods=list(METHODS), matched20_leaders=leaders, differences_against_FNO20_pp=differences,
        final_solver_selected=False, independent_blind_test=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['method', 'case', 'U_relative_L2', 'signed_S_relative_L2', 'hotspot_relative_L2', 'new_inference'])
        for n in sorted(rows):
            for key in keys:
                z = rows[n]['full60']['per_case'][key]
                writer.writerow([n, key, z['u_relative_l2'], z['stress_relative_l2'],
                    z['stress_hotspot_relative_l2'], n in METHODS[1:]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, f, title in zip(axes, FIELDS, ('Displacement', 'Signed stress', 'Stress hotspot')):
        ax.bar(['FNO20', 'Hash20', 'TS20'], [100*rows[n]['full60'][f] for n in METHODS],
               color=['#477db3', '#e39745', '#6a9b6c'])
        ax.set_title(title); ax.set_ylabel('Case mean relative L2 (%)'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Matched20 exposure / iterated knownDEV60, not final blind test')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    files = ['BRIEF.md', 'metrics.json', 'per_case.csv', 'comparison.png']
    assert all((out/name).is_file() for name in files)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN),
        terminal_binding=binding, files=files, actual_CPU_seconds=time.perf_counter()-began,
        new_GT_reads=0, PT_body_reads=0, GPU_calls=0, matched20_leaders=leaders))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, matched20_leaders=leaders))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","GPU":0,"GT_reads":0,"PT_reads":0}')
