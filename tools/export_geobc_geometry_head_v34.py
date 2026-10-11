"""Terminal scalar-only learned geometry/head transfer report; no PT/arrays/GT/GPU reads."""
import argparse
import time

OWN = 'tools/export_geobc_geometry_head_v34.py'
PLAN = 'notes/geobc_geometry_head_v34_plan.json'
REVIEW = 'notes/geobc_geometry_head_v34_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.geometry_head_readout_contract_v34 import approved_contract, check_identity, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256'] == d.sha(OWN)
    plan, _, _, _, qualified = approved_contract(PLAN)
    assert marker['scientific_digest'] == plan['scientific_digest']
    folder = d.path(plan['output']); summary = d.read(folder/'summary.json'); external = d.read(folder/'external_terminal.json')
    assert summary == d.read(folder/'status.json') and summary['status'] == 'completed' and summary['protocol_qualified']
    assert summary['closed'] and not summary['active'] and external['closed'] and not external['active']
    assert external['exit_code'] == 0 and external['within_external_budget'] and not external['external_timeout']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process', 'guard_process'))
    assert summary['execution']['run_process'] == external['child_process']
    assert summary['execution']['guard']['guard_process'] == external['guard_process']
    assert summary['execution']['scientific_digest'] == external['scientific_digest'] == plan['scientific_digest']
    assert summary['committed_finite_predictions'] == 180 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    rows = summary['candidates']; assert len(rows) == 14
    for row in rows.values(): complete_physical(row['full60'], plan['scientific']['case_keys'])
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    names = ('R16_TRANSOLVER_1500', 'R16_FNO16_CONTROL_1500', 'R16_TRANSOLVER_HEADWARM_1500', 'R16_FNO16_HEADWARM_RESET_1500', 'R16_FNO16_CONTINUE_3000')
    labels = ('Original TS fresh5', 'FNO16 teacher5', 'TS learned-head +5', 'FNO16 learned-head reset +5', 'FNO16 intact 5 to10')
    current = qualified[names[2]]; train = current['summary']; points = []
    for name, label in zip(names, labels):
        row = rows[name]; original = d.read(check_identity(row['upstream_terminal_bindings']['summary']))
        arm = original['methods'][row['model_profile']['method']]
        tm = arm['endpoints'][str(row['step'])]['full_TRAIN']; complete_physical(tm, current['plan']['scientific']['TRAIN_keys'])
        points.append(dict(name=name, label=label, TRAIN={f:tm[f] for f in fields}, DEV={f:row['full60'][f] for f in fields},
            marginal_training_epoch_seconds=arm['training_epoch_seconds'],
            teacher_plus_marginal_training_epoch_seconds=arm.get('teacher_plus_marginal_training_epoch_seconds',arm['training_epoch_seconds']),
            inference_epoch_seconds=row['inference_epoch_seconds'], model_profile=row['model_profile']))
    comparisons = {'TS_warm_minus_old_fresh5':(2,0), 'FNO_reset_minus_teacher5':(3,1),
        'FNO_continue_minus_teacher5':(4,1), 'TS_warm_minus_FNO_reset':(2,3), 'TS_warm_minus_FNO_continue':(2,4)}
    deltas = {name:{f:points[a]['DEV'][f]-points[b]['DEV'][f] for f in fields} for name,(a,b) in comparisons.items()}
    best_score = min(v['full60']['selection_score'] for v in rows.values())
    best = sorted(k for k,v in rows.items() if v['full60']['selection_score'] == best_score); assert best == summary['eligible_best']
    tsnames = [n for n in rows if 'TRANSOLVER' in n or 'TS3' in n]
    best_ts_score = min(rows[n]['full60']['selection_score'] for n in tsnames)
    best_ts = sorted(n for n in tsnames if rows[n]['full60']['selection_score'] == best_ts_score)
    out = d.path('results/geobc_geometry_head_v34'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# 学得的几何编码器／查询头：TS与FNO迁移对照', '',
        '**三控制均继承FNO16在TRAIN上的5遍状态，再新增5遍。下表累计纯训练包含一次teacher5，不当fresh5公平比较。**', '',
        '| 方法 | 位移误差 | 应力误差 | teacher+新增纯训练分钟 | 推理秒／60例 |', '|---|---:|---:|---:|---:|']
    for point in points[2:]:
        m = point['DEV']; lines.append(f"| {point['label']} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {point['teacher_plus_marginal_training_epoch_seconds']/60:.2f} | {point['inference_epoch_seconds']:.2f} |")
    lines += ['', f"原TS fresh5 U/S{100*points[0]['DEV'][fields[0]]:.2f}%／{100*points[0]['DEV'][fields[1]]:.2f}%；原FNO16 teacher5 {100*points[1]['DEV'][fields[0]]:.2f}%／{100*points[1]['DEV'][fields[1]]:.2f}%。",
        f"全部历史混合曝光候选中描述性最佳：{'、'.join(best)}；TS系列描述性最佳：{'、'.join(best_ts)}。该排序仅决定后续优先级，不能抹平容量、表示、继承状态与曝光差异。", '', '新几何差值（负值更好）：']
    for label, change in deltas.items():
        lines.append(f"- {label}：位移{100*change[fields[0]]:+.2f}、应力{100*change[fields[1]]:+.2f}、热点{100*change[fields[2]]:+.2f}个百分点。")
    cost=train['inherited_teacher_cost']
    lines += ['', f"本轮三控制整轮{current['external']['elapsed_epoch_seconds']/60:.2f}分钟，DEV{external['elapsed_epoch_seconds']:.2f}秒；teacher旧整轮{cost['stage_epoch_seconds']/60:.2f}分钟／旧纯训练{cost['training_epoch_seconds']:.2f}秒另列。三控制共享一次新GPU缓存和一次teacher PT读取，旧teacher不重播三次。表中纯训练／推理不包括缓存与检查点等整轮成本；UTC与monotonic保留原值。", '',
        'TS与FNO重置控制：各自seed2026完整fresh state先与原控制SHA匹配，再严格复制同一TRAIN teacher的encoder/head；各自operator前后hash保持，Adam全新。完整FNO续训控制：全state、Adam1500步、四RNG、cursor和每case5次曝光保留，继续到3000／10遍。所有组仍可训练，原R16输入、native25、query、参数规模、旧scaler、原物理损失不变。', '',
        '新增1500次更新都使用固定10遍均衡顺序的后5遍，前5遍hash与teacher旧顺序一致；重置控制仅记录新增5遍算子曝光，encoder/head继承teacher5；完整FNO全部参数10遍。旧fresh5使用前5遍顺序，因此与旧fresh5的差同时含初始化、信息曝光和顺序变化；不是纯架构或纯初始化因果消融。TS1.882M与FNO16 4.244M也不是容量匹配。', '',
        'teacher只从原100TRAIN×三载荷得到，不用DEV更新、不产生teacher预测伪GT。所有监督始终是原CSV GRID对齐H5 vertices的signed U(mm)/S(MPa)，旧TRAIN40 scaler无重拟合。FP32最高精度／TF32关闭、30.5GB SI。', '',
        '实际共享全300TRAIN GPU缓存，三arm最大325387节点encoder/operator/head及jointleaf正有限梯度先于任何新增更新，并恢复相应RNG；全部300物理TRAIN读出复用缓存共900次。CPU合成扰动teacher的33337节点含569尾块只检验迁移数学与完整链式梯度，不是真实训练teacher或GPU资格替代。', '',
        '180新wholeN finite immutable预测保存并重验后才共享knownDEV60目标一次；11旧候选只复用物理标量，不重跑旧推理或旧数组，rawCSV／FINAL／scalerfit0。knownDEV20CAD×三载荷已用于迭代，单seed、availabilityconditional官方TRAIN来源，不是新盲测或架构上限。', '',
        '完整14候选、全部ties、每载荷20、配对20CAD、TRAIN300、初始化／teacher／Adam／参数／显存／成本见metrics.json。根据物理误差及实际成本决定下一宏观阶段，不自动40遍或24000步。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points, full_DEV_candidates=rows, paired_CAD20=summary['paired_CAD20'],
        DEV_gates=summary['DEV_gates'], best=best,best_TS=best_ts, macro_differences=deltas,
        full_TRAIN=train['methods'], teacher_cost=train['inherited_teacher_cost'], training_campaign=current['external'], readout_campaign=external))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12.8,4.8),sharey=True)
    for axis, split in zip(axes,('TRAIN','DEV')):
        for i, point in enumerate(points):
            axis.bar([x+(i-2)*.16 for x in range(3)],[100*point[split][f] for f in fields],width=.15,
                color=('#c88320','#245d9c','#bc3a3a','#37a28c','#7c55b5')[i],label=point['label'])
        axis.set_xticks([0,1,2],['Displacement','Signed stress','Hotspot']);axis.set_title(split);axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Learned TRAIN geometry/head / inherited5 and additional5; reset versus intact continuation')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/n).is_file() for n in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=plan['scientific_digest'],source_sha256=d.sha(OWN),
        summary_identity=d.identity(folder/'summary.json',hashed=True),external_identity=d.identity(folder/'external_terminal.json',hashed=True),
        actual_CPU_seconds=time.monotonic()-began,numerical_array_reads=0,PT_body_reads=0,GPU_calls=0,
        actual_visual_accepted=False,final_solver_selected=False,files=['BRIEF.md','metrics.json','comparison.png']))
    print(dict(status='completed',best=best,best_TS=best_ts,CPU_seconds=time.monotonic()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export: run()
