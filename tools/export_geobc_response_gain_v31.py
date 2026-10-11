"""Terminal scalar-only response-gain report; no PT/arrays/GT/GPU reads."""
import argparse
import time

OWN = 'tools/export_geobc_response_gain_v31.py'
PLAN = 'notes/geobc_response_gain_v31_plan.json'
REVIEW = 'notes/geobc_response_gain_v31_exporter_root_review.json'


def run():
    began = time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.response_gain_readout_contract_v31 import approved_contract, check_identity, complete_physical
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
    assert summary['committed_finite_predictions'] == 120 and summary['owned_GT_array_reads'] == summary['owned_GT_read_attempts'] == 60
    assert summary['raw_GT_reads'] == summary['FINAL_numeric_reads'] == summary['training_updates'] == summary['scaler_fits'] == 0
    assert max(summary['elapsed_epoch_seconds'], summary['elapsed_monotonic_seconds'], external['elapsed_epoch_seconds'], external['elapsed_monotonic_seconds']) <= 1800
    rows = summary['candidates']; assert len(rows) == 11
    for row in rows.values(): complete_physical(row['full60'], plan['scientific']['case_keys'])
    fields = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')
    names = ('R16_TRANSOLVER_1500', 'R16_TRANSOLVER_GAIN_1500', 'R16_FNO16_CONTROL_1500', 'R16_FNO16_GAIN_1500')
    labels = ('Transolver physical', 'Transolver response gain', 'FNO16 physical', 'FNO16 response gain')
    current = qualified[names[1]]; train = current['summary']; points = []
    for name, label in zip(names, labels):
        row = rows[name]; original = d.read(check_identity(row['upstream_terminal_bindings']['summary']))
        arm = original['methods'][row['model_profile']['method']]
        tm = arm['endpoints']['1500']['full_TRAIN']; complete_physical(tm, current['plan']['scientific']['TRAIN_keys'])
        points.append(dict(name=name, label=label, TRAIN={f:tm[f] for f in fields}, DEV={f:row['full60'][f] for f in fields},
            training_epoch_seconds=arm['training_epoch_seconds'], inference_epoch_seconds=row['inference_epoch_seconds']))
    deltas = {method:{f:points[i+1]['DEV'][f]-points[i]['DEV'][f] for f in fields} for method,i in [('TS',0),('FNO16',2)]}
    best_score = min(v['full60']['selection_score'] for v in rows.values())
    best = sorted(k for k,v in rows.items() if v['full60']['selection_score'] == best_score); assert best == summary['eligible_best']
    tsnames = [n for n in rows if 'TRANSOLVER' in n or 'TS3' in n]
    best_ts_score = min(rows[n]['full60']['selection_score'] for n in tsnames)
    best_ts = sorted(n for n in tsnames if rows[n]['full60']['selection_score'] == best_ts_score)
    role_names = list(dict.fromkeys([best[0], best_ts[0], 'R64_FNO_W16_1500']))
    out = d.path('results/geobc_response_gain_v31'); assert not out.exists(), 'No report replay'
    out.mkdir(parents=True)
    lines = ['# 输入物理尺度：同5遍TS／FNO对照', '',
        f"**同5遍联合最佳：{'、'.join(best)}；最佳TS：{'、'.join(best_ts)}。尚未最终选型。**", '',
        '| 方法 | 位移误差 | 应力误差 | 纯训练分钟 | 推理秒／60例 |', '|---|---:|---:|---:|---:|']
    for name in role_names:
        row = rows[name]; m = row['full60']
        lines.append(f"| {name} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {row['terminal_campaign_training_epoch_seconds']/60:.2f} | {row['inference_epoch_seconds']:.2f} |")
    lines += ['', '幅值先验减各自原物理控制（负值更好）：']
    for label in ('TS','FNO16'):
        change = deltas[label]
        lines.append(f"- {label}：位移{100*change[fields[0]]:+.2f}、应力{100*change[fields[1]]:+.2f}、热点{100*change[fields[2]]:+.2f}个百分点。")
    lines += ['', f"新双模型训练整轮{current['external']['elapsed_epoch_seconds']/60:.2f}分钟，DEV整轮{external['elapsed_epoch_seconds']:.2f}秒。UTC与monotonic保留原值；图表／表格纯训练和推理时间不包括缓存、检查点等整轮成本。", '',
        '只改输出幅值的条件化：已知输入F(N)、M(Nmm)、E(MPa)、L(mm)构造U尺度 norm(F)/(E L)+norm(M)/(E L²)，S尺度 norm(F)/L²+norm(M)/L³，除以固定输入schema参考常数。该正尺度是量纲一致的幅值先验，不是精确FEM相似律；不增加缺失信息、不拟合新scaler、不造标签符号、不加epsilon或截断。', '',
        '旧scaler编码输出先解码为signed物理响应、乘正尺度、再编码；原物理相对U/S损失、全部原节点、原native25、R16输入／query／头、容量、Adam和均衡顺序不变。每模型完整初始权重与自己的控制一致，但初始输出函数会因尺度变化而不同，不能声称两者函数相同。', '',
        '原版DeepJEB，100TRAIN×三载荷／1500更新／5遍，knownDEV20CAD×三载荷，旧TRAIN40 scaler，单seed2026、最高FP32主干、FP64标量变换、30.5GB SI。原signed U(mm)/S(MPa)、CSV GRID对齐H5 vertices、物理损失保持。', '',
        '实际全300TRAIN缓存及最大325387节点encoder/operator/head和两leaf正有限梯度资格先于更新；CPU33337节点含569尾块对单次完整反传检查不是GPU资格替代。120新模型全N finite预测保存并重验后才共享knownDEV60目标一次，九个旧候选只复用物理标量，不重跑推理或读旧数组。FINAL/rawCSV/scalerfit0。', '',
        '完整11候选、各载荷20、配对20CAD、TRAIN300、输入尺度分布、参数／显存／成本见metrics.json。knownDEV已参与迭代，不是新盲测或架构上限；没有明显改善则停止该路线，不自动40遍或24000步。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points, full_DEV_candidates=rows, paired_CAD20=summary['paired_CAD20'],
        DEV_gates=summary['DEV_gates'], best=best,best_TS=best_ts, gain_minus_own_control=deltas,
        full_TRAIN=train['methods'], input_response_gains=train['input_response_gains'], training_campaign=current['external'], readout_campaign=external))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11.2,4.4),sharey=True)
    for axis, split in zip(axes,('TRAIN','DEV')):
        for i, point in enumerate(points):
            axis.bar([x+(i-1.5)*.19 for x in range(3)],[100*point[split][f] for f in fields],width=.18,
                color=('#c88320','#bc3a3a','#245d9c','#37a28c')[i],label=point['label'])
        axis.set_xticks([0,1,2],['Displacement','Signed stress','Hotspot']);axis.set_title(split);axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Same R16 and five passes / fixed input-only response gain')
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
