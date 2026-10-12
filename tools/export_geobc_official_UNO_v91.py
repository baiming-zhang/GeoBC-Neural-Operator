"""Scalar-only report of the real fixed TRAIN9 official UNO valid-skip attempt."""
import argparse
import csv
import json
import time

OWN = 'tools/export_geobc_official_UNO_v91.py'
REVIEW = 'notes/geobc_official_UNO_valid_skip_recovery_v91_exporter_root_review.json'
FIELDS = ('case_mean_u_relative_l2', 'case_mean_stress_relative_l2', 'case_mean_stress_hotspot_relative_l2')


def export():
    began = time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.official_uno_valid_skip_contract_v91 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal, complete_physical
    p, a, e, binding = terminal(PLAN); s = p['scientific']; r = d.read(REVIEW); arm = s['candidate']; z = a['arms'][arm]
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256'] == d.sha(OWN) and r['SCI'] == p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 900
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts'] == a['cache_receipt']['physical_TRAIN_target_reads'] == 9
    assert a['cache_receipt']['encoded_TRAIN_target_reads'] == a['raw_GT_reads'] == a['DEV_FINAL_reads'] == a['scaler_fits'] == a['parent_PT_reads'] == 0
    assert a['resident_physical_target_reuses'] == 27 and z['step'] == 900 and z['case_exposure_counts'] == {k: 100 for k in s['selected_TRAIN_keys']}
    assert z['qualification']['passed'] and z['qualification']['original_nodes'] == 242936
    assert z['qualification']['model_unchanged'] and z['qualification']['Adam_unchanged'] and z['qualification']['fourRNG_restored']
    initial = z['initialization']; profile = initial['parameter_profile']
    assert initial['original_initialization']['original_FNO_model_digest'] == s['expected_original_model_digest']
    assert initial['original_initialization']['encoder_head_bytes_kept']
    assert initial['shared_encoder_head_digests'] == initial['original_initialization']['kept_encoder_head_digests']
    assert initial['actual_UNO_config'] == s['UNO_kwargs']
    assert profile['total_real_degrees_of_freedom'] == 9354500 and profile['total_storage_numel'] == 4709636
    assert profile['operator']['complex_storage_numel'] == 4644864 and profile['operator']['dtypes'] == ['torch.complex64', 'torch.float32']
    assert z['qualification']['complex_safe_parameter_gradients'] and a['pre_GT_model_construction_passed']
    assert e['original_campaign_elapsed_epoch_seconds'] <= s['whole_campaign_budget_seconds']
    assert e['original_campaign_elapsed_monotonic_seconds'] is None if not e['same_boot'] else e['original_campaign_elapsed_monotonic_seconds'] <= s['whole_campaign_budget_seconds']
    first = d.read(p['prior_native_identity']['path']); second = d.read(p['prior_second_native_identity']['path'])
    assert first['cache_receipt'] is None and second['cache_receipt'] is None
    assert first['optimizer_step_completions'] == second['optimizer_step_completions'] == 0
    for point in z['endpoints'].values(): complete_physical(point['full_TRAIN9'], s['selected_TRAIN_keys'])
    out = d.path('results/geobc_official_UNO_valid_skip_recovery_v91'); assert not (out/'manifest.json').exists()
    out.mkdir(parents=True, exist_ok=True); gate = a['TRAIN_gate']
    old = s['baseline_scalar_endpoints']['100']['full_TRAIN9']; new = z['endpoints']['100']['full_TRAIN9']
    lines = ['# 官方U-NO：双尺度场算子短筛选', '', '**同三CAD×三载荷／100遍／900次全原节点更新；旧普通FNO仅复用冻结标量。原PointNet／查询头初始bytes、输入、物理联合损失、scaler保持；场算子更换为官方U-NO。**', '',
        '| 场算子 | 位移误差 | signed应力误差 | 热点误差 |', '|---|---:|---:|---:|']
    for label, row in (('原FNO', old), ('官方U-NO／64与32双尺度', new)):
        lines.append('| '+label+' | '+' | '.join(f'{100*row[f]:.2f}%' for f in FIELDS)+' |')
    lines += ['', f'新算子减旧基线：位移{100*gate["U_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点，联合{100*gate["primary_difference"]:+.2f}个百分点。预声明TRAIN门：{"PASS" if gate["passed"] else "FAIL"}。', '',
        f'有效恢复执行{e["elapsed_epoch_seconds"]:.2f}秒／学习{z["training_epoch_seconds"]:.2f}秒。原任务从首次启动到结束{e["original_campaign_elapsed_epoch_seconds"]:.2f}秒，包含两次零标签／零更新接口失败和修复等待，原1200秒绝对截止未重置。跨WSL boot原monotonic生命周期未知，原记录保留；本次执行epoch/mono分别记录，不零填或缩放。', '',
        '官方代码commit b307030d9b7ee530823d6a05bddb8238ba22a743／MIT。61输入→16latent；hidden16、层宽16/32/32/32/16、5层modes12/12/12、64→64→32→32→32→64，水平skip4←0/3←1；已有xyz输入所以禁用默认positional embedding。所有固定参数与源身份保存。', '',
        '参数storage数4,709,636、real自由度9,354,500，含4,644,864个complex64参数；原FNO4,526,858实数参数。容量、谱模式约定、FFT/MLP/双尺度/checkpoint策略均不同，不是单因素归因或等容量对照。复数参数使用Adam fused=False/foreach=False，lr/betas/eps/weight_decay与旧基线相同；不声称等计算成本。', '',
        '初次私有namespace缺绝对neuralop.utils，第二次soft-gating不支持变化通道；两次均在读取标签／资格／优化前失败，完整保存，不能当方法科学负结果。修复loader只别名同一官方utils，不改官方数学；有效配置将channelMLP skip改为官方测试采用的linear，显式新SCI记录，不能称科学配方完全不变。未恢复不存在的旧Adam/RNG/PT；依指定seed重建fresh0。', '',
        '真实242936节点全encoder/head/官方operator梯度（abs-square包含实部与虚部）及context/latent leaves、实际64query坐标链和分块检查通过；资格model／Adam／四RNG保持，原encoder/queryhead初值未更改。新operator初始化与旧FNO函数不同，不作零扩展parity宣称。', '',
        '**仍是九TRAIN场拟合，不是新CAD泛化。** 原物理9标签本次各读一次（前两次0），三个端点常驻复用27；编码／CSV／DEV／FINAL／父PT／scaler拟合0。不是原作者DeepJEB精确配置复现或独立离节点FEM精度。', '',
        '通过门仅支持另冻结短100TRAIN宏观比较；失败关闭固定有效U-NO配置，无width/modes/scale/depth/seed/loss或DEV救结果，不用训练集选最终solver。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    d.write(out/'metrics.json', dict(actual_native=a, actual_external=e, terminal_binding=binding,
        baseline_scalar_endpoints=s['baseline_scalar_endpoints'], TRAIN_gate=gate, generalization_tested=False, final_solver_selected=False))
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file); writer.writerow(['method', 'passes', 'case', 'U_relative_L2', 'signed_S_relative_L2', 'hotspot_relative_L2'])
        for label, endpoints in (('old_plain_scalar', s['baseline_scalar_endpoints']), (arm, z['endpoints'])):
            for passes in (0, 10, 100):
                for key, row in endpoints[str(passes)]['full_TRAIN9']['per_case'].items():
                    writer.writerow([label, passes, key, row['u_relative_l2'], row['stress_relative_l2'], row['stress_hotspot_relative_l2']])
    loss = [json.loads(line) for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss) == 900 and [v['step'] for v in loss] == list(range(1, 901))
    d.write(out/'loss.json', loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.6))
    for field, label in (('physical_relative_squared', 'Official UNO physical loss'),):
        vals = [sum(v[field] for v in loss[i:i+9])/9 for i in range(0, 900, 9)]
        axes[0].plot(range(1, 101), vals, label=label)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Full TRAIN9 passes'); axes[0].legend()
    for offset, row, label in ((-.18, old, 'Old plain FNO'), (.18, new, 'Official UNO')):
        axes[1].bar([i+offset for i in range(3)], [100*row[f] for f in FIELDS], .36, label=label)
    axes[1].set_xticks(range(3), ('U', 'Signed S', 'Hotspot')); axes[1].set_ylabel('Mean relative L2 (%)'); axes[1].set_ylim(0,max(row[f] for row in (old,new) for f in FIELDS)*100+10); axes[1].legend(loc='upper center',ncol=2,fontsize=8)
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.suptitle('TRAIN9 fixed900updates; official UNO two-scale; original inputs and decoder unchanged')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
    d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_sha256=d.sha(OWN), terminal_binding=binding,
        files=['BRIEF.md', 'metrics.json', 'per_case.csv', 'loss.json', 'comparison.png'], actual_CPU_seconds=time.perf_counter()-began,
        new_GT_PT_arrays_GPU=0, TRAIN_gate=gate, generalization_tested=False))
    print(dict(status='completed', actual_CPU_seconds=time.perf_counter()-began, TRAIN_gate=gate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export', action='store_true')
    if parser.parse_args().export: export()
    else: print('{"status":"source_only","new_GT_PT_arrays_GPU":0}')
