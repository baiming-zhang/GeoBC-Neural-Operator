"""One scalar-only actual paired readout report; no PT, GT arrays or GPU."""
import argparse
import csv
import time

OWN = 'tools/export_geobc_fno40_pair_readout_v73.py'
REVIEW = 'notes/geobc_fno40_pair_readout_v73_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')


def cost(row):
    if 'endpoint_training_cost_exact' in row:
        ledger = row['endpoint_training_cost']
        return None if ledger is None else ledger['training_epoch']/60
    for field in ('complete0to20_cost','complete0to10_cost'):
        if field in row:
            return row[field]['training_epoch']/60
    if 'trajectory_training_epoch_seconds' in row:
        return row['trajectory_training_epoch_seconds']/60
    return row['terminal_campaign_training_epoch_seconds']/60


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.fno40_pair_readout_contract_v73 import PLAN, METHODS, terminal, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    p, a, e, binding = terminal(PLAN)
    assert marker['SCI'] == p['scientific_digest'] and a['protocol_qualified']
    assert a['committed_finite_predictions'] == 120 and a['prediction_barrier_complete']
    assert a['owned_GT_read_attempts'] == a['owned_GT_array_reads'] == 60
    assert a['checkpoint_read_attempts'] == a['checkpoint_reads'] == a['checkpoint_body_hash_reads'] == 2
    assert all(a['strict_loaded_checkpoint_tensor_state_verified'].values())
    assert set(a['strict_loaded_checkpoint_tensor_state_verified']) == set(METHODS.values())
    assert a['raw_GT_reads'] == a['FINAL_numeric_reads'] == a['training_updates'] == a['scaler_fits'] == 0
    assert not a['final_solver_selected'] and not a['automatic_continuation']
    rows, keys = a['candidates'], p['scientific']['case_keys']
    for row in rows.values():
        complete_physical(row['full60'],keys)
    names = sorted(rows,key=lambda n:(rows[n]['full60']['selection_score'],n))
    leaders = [n for n in names if rows[n]['full60']['selection_score'] == rows[names[0]]['full60']['selection_score']]
    ts = [n for n in names if 'TRANSOLVER' in n or n.startswith('TS_')]
    assert ts
    best_ts = [n for n in ts if rows[n]['full60']['selection_score'] == rows[ts[0]]['full60']['selection_score']]
    assert leaders == a['eligible_best']
    roles = [('当前最低综合误差',leaders[0]),('最佳已有Transolver',best_ts[0]),('此前FNO20','FNO100_PHYSICAL_6000')]
    compact = []
    for label,name in roles:
        old = next((z for z in compact if z[1]==name),None)
        if old:
            old[0] += '／'+label
        else:
            compact.append([label,name])
    bm = rows['FNO100_PHYSICAL_6000']['full60']
    differences = {n:{f:100*(rows[n]['full60'][f]-bm[f]) for f in FIELDS} for n in METHODS.values()}
    m, tm = rows[leaders[0]]['full60'], rows[best_ts[0]]['full60']
    leader_vs_TS = {f:100*(m[f]-tm[f]) for f in FIELDS}
    training = d.read('results/geobc_fno_learning40_v72/metrics.json')
    out = d.path('results/geobc_fno40_pair_readout_v73')
    assert not (out/'manifest.json').exists(),'One immutable actual export'
    out.mkdir(parents=True,exist_ok=True)
    lines = ['# 30／40遍：训练收益是否迁移到新CAD','',
        '**固定20个新CAD×三载荷、全原节点、signed U(mm)/S(MPa)。两端点预测全部提交后才共享读取评价标签；旧方法仅复用标量。**','',
        '| 角色／方法 | 位移误差 | 应力误差 | 累计训练区间分钟 | 60场推理秒 |',
        '|---|---:|---:|---:|---:|']
    for label,n in compact:
        row = rows[n];z = row['full60'];minutes = cost(row)
        shown = f'{minutes:.2f}' if minutes is not None else f'未知（≤{row["endpoint_training_cost_upper_bound"]["training_epoch"]/60:.2f}）'
        lines.append(f'| {label}：{n} | {100*z[FIELDS[0]]:.2f}% | {100*z[FIELDS[1]]:.2f}% | {shown} | {row["inference_epoch_seconds"]:.2f} |')
    lines += ['',f'最低综合误差方案相对最佳已有TS：位移{leader_vs_TS[FIELDS[0]]:+.2f}、应力{leader_vs_TS[FIELDS[1]]:+.2f}、热点{leader_vs_TS[FIELDS[2]]:+.2f}个百分点。综合分为病例平均位移／应力相对L2的等权均值，热点单列，不用子集平均。','']
    for step,n in METHODS.items():
        diff = differences[n];z=rows[n]['full60']
        lines.append(f'{step//300}遍相对20遍：位移{diff[FIELDS[0]]:+.2f}、应力{diff[FIELDS[1]]:+.2f}、热点{diff[FIELDS[2]]:+.2f}个百分点；当前热点误差{100*z[FIELDS[2]]:.2f}%，预声明DEV推进门通过：{a["DEV_gates_against20"][n]["passed"]}。')
    lines += ['',f'本轮完整验证{e["elapsed_epoch_seconds"]:.2f}秒：120份全节点预测／有限值／身份重验先全部提交，再共享读取已有60份物理GT各一次。两个自身检查点各hash/load一次，旧参考无PT／数组／GT／推理重播。训练更新、原CSV、FINAL与scaler拟合均0。','',
        '训练时间包含各方案原来记录的资格、读出或保存开销，不是纯CUDA时间；准备、技术失败与阶段成本在JSON完整保留。30遍前缀时间没有单独记录，记未知并列40遍上界，不能由更新比例推算。40遍累计训练区间epoch／monotonic分别723.83／662.09秒；时间轴差异原样保留。','',
        '这是同一FNO轨迹的曝光趋势和已有开发集比较，不是等成本架构排名。已知DEV用于过多轮迭代、单seed、可用来源条件选择；不声称独立盲测、收敛、最终solver或独立离网格FEM精度。不会自动80遍，也不会因TRAIN门剔除任何一个端点。全部候选、并列、逐载荷、逐CAD配对与热点在JSON／CSV。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=binding,
        leaders=leaders,best_TS=best_ts,compact_roles=compact,differences_against20_pp=differences,
        leader_vs_TS_pp=leader_vs_TS,training_cost_minutes={n:cost(r) for n,r in rows.items()},
        final_solver_selected=False,independent_blind_test=False))
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['method','case','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2','new_inference'])
        for n in names:
            for k in keys:
                z=rows[n]['full60']['per_case'][k]
                writer.writerow([n,k,z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2'],n in METHODS.values()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,f,title in zip(axes,FIELDS,['Displacement','Signed stress','Stress hotspot']):
        exposures=[20,30,40]
        ax.plot(exposures,[100*training['endpoints'][str(count)][f] for count in exposures],'-o',label='TRAIN300')
        ax.plot(exposures,[100*rows['FNO100_PHYSICAL_'+str(count*300)]['full60'][f] for count in exposures],'-o',label='knownDEV60')
        ax.set_title(title);ax.set_xticks(exposures);ax.set_xlabel('Full case exposures');ax.grid(alpha=.2)
    axes[0].set_ylabel('Physical relative L2 (%)');axes[-1].legend()
    fig.suptitle('Same FNO trajectory / iterated development set, not final test')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','comparison.png']
    assert all((out/k).is_file() for k in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),
        terminal_binding=binding,actual_CPU_seconds=time.perf_counter()-began,files=files,
        GT_reads=0,PT_body_reads=0,GPU_calls=0,leaders=leaders,best_TS=best_ts))
    print(dict(status='completed',CPU_seconds=time.perf_counter()-began,leaders=leaders,best_TS=best_ts))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","GPU_calls":0,"GT_reads":0}')
