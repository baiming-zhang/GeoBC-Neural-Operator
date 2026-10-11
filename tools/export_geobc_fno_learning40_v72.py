"""Scalar-only dose-response report, only after genuine full40 terminal."""
import argparse
import csv
import time

OWN = 'tools/export_geobc_fno_learning40_v72.py'
PLAN = 'notes/geobc_fno_learning40_v72_plan.json'
REVIEW = 'notes/geobc_fno_learning40_v72_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2','case_mean_stress_relative_l2',
          'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.fno_learning40_contract_v72 import approved_contract, terminal, complete_physical
    marker = d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released']
    assert marker['source_sha256'] == d.sha(OWN)
    approved_contract(PLAN)
    p, a, e, binding = terminal(PLAN)
    s, keys = p['scientific'], p['scientific']['TRAIN_keys']
    assert marker['SCI'] == p['scientific_digest'] and a['training_complete']
    assert a['step'] == a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 12000
    assert a['additional_optimizer_step_attempts'] == a['additional_optimizer_step_completions'] == 6000
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['case_exposure_counts'] == {k:40 for k in keys}
    assert a['cached_TRAIN_physical_read_attempts'] == a['cached_TRAIN_physical_reads'] == 300
    assert a['cache_receipt']['encoded_TRAIN_target_reads'] == a['cache_receipt']['encoded_TRAIN_target_read_attempts'] == 0
    assert a['resident_TRAIN_physical_target_reuses'] == 600
    assert a['parent_checkpoint_hash_reads'] == a['parent_checkpoint_loads'] == 1
    assert a['strict_loaded_checkpoint_tensor_state_verified'] and a['restored_model_Adam_fourRNG_cursor_exposure']
    check = a['actual_largest_fullN_gradient_check']
    assert check['passed'] and check['model_unchanged'] and check['Adam_unchanged'] and check['fourRNG_restored']
    assert a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == 0
    endpoints = {20:s['parent_TRAIN300'],30:a['endpoints']['9000']['full_TRAIN'],40:a['endpoints']['12000']['full_TRAIN']}
    for z in endpoints.values():
        complete_physical(z, keys)
    cost = dict(training_epoch=s['complete0to20_cost']['training_epoch']+a['training_epoch_seconds'],
        training_monotonic=s['complete0to20_cost']['training_monotonic']+a['training_monotonic_seconds'],
        stage_epoch=s['complete0to20_cost']['stage_epoch']+e['elapsed_epoch_seconds'],
        stage_monotonic=s['complete0to20_cost']['stage_monotonic']+e['elapsed_monotonic_seconds'])
    delta = {k:100*(endpoints[40][k]-endpoints[20][k]) for k in FIELDS}
    out = d.path('results/geobc_fno_learning40_v72')
    assert not (out/'manifest.json').exists(), 'One export only'
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# FNO：同一轨迹20→40遍训练不足诊断','',
        '**相同100个TRAIN CAD×三载荷、全原节点、signed U(mm)/S(MPa)。保持20遍模型、Adam、四类RNG、顺序与scaler；只增加6000次病例更新。**','',
        '| 累计曝光 | TRAIN位移误差 | TRAIN应力误差 | TRAIN热点误差 |',
        '|---|---:|---:|---:|']
    for step, z in endpoints.items():
        lines.append(f'| {step}遍 | {100*z[FIELDS[0]]:.2f}% | {100*z[FIELDS[1]]:.2f}% | {100*z[FIELDS[2]]:.2f}% |')
    lines += ['',f'40减20：位移{delta[FIELDS[0]]:+.2f}、应力{delta[FIELDS[1]]:+.2f}、热点{delta[FIELDS[2]]:+.2f}个百分点。预声明TRAIN门通过：{a["TRAIN_gate"]["passed"]}。两个端点均保留，训练门不筛掉已完成候选；knownDEV新结果尚未执行，不将训练收益当泛化收益。','',
        f'本次新增训练区间{a["training_epoch_seconds"]:.2f}秒，含中间30遍读出与检查点；完整阶段{e["elapsed_epoch_seconds"]:.2f}秒。累计0→40训练区间epoch／monotonic为{cost["training_epoch"]:.2f}／{cost["training_monotonic"]:.2f}秒，累计训练阶段{cost["stage_epoch"]:.2f}／{cost["stage_monotonic"]:.2f}秒。旧时间轴差异保留，不缩放，不用上一段10→20成本冒充整个0→20。区间不是纯CUDA时间，阶段间分析等待排除。','',
        '恢复自身step6000检查点hash/load各一次，model字节严格匹配，Adam各state6000、四类RNG、cursor和每场20曝光一致；前6000顺序SHA吻合，首最大325387节点联合梯度检查不改变model/Adam/RNG。全部300原物理标签各读一次，30/40常驻复用600次，不读取不用的编码标签、DEV、FINAL、原CSV或重拟合scaler。','',
        '模型与精度不变，FNO9共4,526,858参数、FP32最高精度且TF32关闭。追加曝光是剂量趋势，不是等成本架构排名、收敛或最终solver证明；不自动80遍或延长其他方法。已有knownDEV已用于迭代，后续应全120预测提交再共享GT60一次，无最终盲测声明。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=binding,
        endpoints=endpoints,differences40_minus20_pp=delta,complete0to40_cost=cost,
        old0to20_cost=s['complete0to20_cost'],TRAIN_gate=a['TRAIN_gate'],
        knownDEV_evaluation_actual=False,final_solver_selected=False))
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['passes','case','U_relative_L2','signed_S_relative_L2','hotspot_relative_L2'])
        for step, values in endpoints.items():
            for key in keys:
                z = values['per_case'][key]
                writer.writerow([step,key,z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for key, label in zip(FIELDS,['Displacement','Signed stress','Stress hotspot']):
        ax.plot(list(endpoints),[100*z[key] for z in endpoints.values()],'-o',label=label)
    ax.set_xlabel('Full-node exposures per TRAIN field')
    ax.set_ylabel('Physical relative L2 (%)')
    ax.set_xticks([20,30,40])
    ax.set_title('Same FNO trajectory / TRAIN300, no new CAD score')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160)
    plt.close(fig)
    files = ['BRIEF.md','metrics.json','per_case.csv','comparison.png']
    assert all((out/f).is_file() for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],
        source_sha256=d.sha(OWN),terminal_binding=binding,actual_CPU_seconds=time.perf_counter()-began,
        files=files,GT_reads=0,PT_body_reads=0,GPU_calls=0,TRAIN_gate_passed=a['TRAIN_gate']['passed']))
    print(dict(status='completed',CPU_seconds=time.perf_counter()-began))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:
        export()
    else:
        print('{"status":"source_only","GPU_calls":0,"GT_reads":0}')
