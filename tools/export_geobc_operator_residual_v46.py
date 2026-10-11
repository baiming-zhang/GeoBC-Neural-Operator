"""Scalar-only CPU report for the exact completed two-residual comparison."""
import argparse
import time


def export():
    started = time.monotonic()
    import math
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash_physical_readout_contract_v18 import complete_physical
    source = 'tools/export_geobc_operator_residual_v46.py'
    review = d.read('notes/geobc_operator_residual_v46_exporter_root_review.json')
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_export_released']
    assert review['source_sha256'] == d.sha(source)
    p = d.read('notes/geobc_operator_residual_readout_v46_plan.json'); r = d.read(p['root_review']); folder = d.path(p['output'])
    s = p['scientific']; a = d.read(folder/'summary.json'); e = d.read(folder/'external_terminal.json')
    assert r['passed'] and r['actual_execution_released'] and p['source_pins'] == r['source_pins']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert all(d.sha(n) == h for n,h in p['source_pins'].items())
    assert a == d.read(folder/'status.json') and a['status'] == 'completed' and a['protocol_qualified']
    assert a['closed'] and not a['active'] and e['closed'] and not e['active']
    assert e['exit_code'] == 0 and e['within_external_budget'] and not e['external_timeout']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['execution']['run_process'] == e['child_process'] and a['execution']['guard']['guard_process'] == e['guard_process']
    assert a['execution']['scientific_digest'] == e['scientific_digest'] == p['scientific_digest']
    assert a['committed_finite_predictions'] == 120 and a['owned_GT_read_attempts'] == a['owned_GT_array_reads'] == 60
    assert a['raw_GT_reads'] == a['FINAL_numeric_reads'] == a['training_updates'] == a['scaler_fits'] == 0
    assert max(e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds']) <= 1800
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')) <= 30500000000
    keys = s['case_keys']; rows = a['candidates']; named = [x['name'] for x in s['candidates']]
    assert set(rows) == set(named+['FNO100_PHYSICAL_1500','FNO100_PHYSICAL_3000'])
    for row in rows.values():
        complete_physical(row['full60'],keys)
        assert row['full60']['fixed_denominator'] == 60
    def values(name):
        m = rows[name]['full60']
        return [m['case_mean_u_relative_l2'],m['case_mean_stress_relative_l2'],m['case_mean_stress_hotspot_relative_l2']]
    def cost(name):
        row = rows[name]
        if name in named: return row['teacher_old_cost']['training_epoch_seconds']+row['terminal_campaign_training_epoch_seconds']
        if name == 'FNO100_PHYSICAL_3000': return row['trajectory_training_epoch_seconds']
        return row['terminal_campaign_training_epoch_seconds']
    best_residual = sorted(named,key=lambda n:(rows[n]['full60']['selection_score'],n))[0]
    leaders = a['eligible_best']; assert set(leaders) == {n for n,row in rows.items() if row['full60']['selection_score'] == min(x['full60']['selection_score'] for x in rows.values())}
    assert not any(gate['passed'] for gate in a['DEV_gates'].values()), 'This report is for the fixed measured negative endpoints'
    for name in rows:
        assert all(math.isfinite(v) for v in values(name)) and math.isfinite(cost(name)) and cost(name) > 0
    chosen = ['FNO100_PHYSICAL_3000',best_residual,'FNO100_PHYSICAL_1500']
    labels = dict(FNO100_PHYSICAL_3000='Intact FNO 10',FNO100_PHYSICAL_1500='FNO teacher 5',FNO5_MLP_RESIDUAL='FNO5 + MLP5',FNO5_TCNN_RESIDUAL='FNO5 + Hash5')
    text = '# 冻结全局解后的局部残差：5遍筛选\n\n'
    text += '**原版DeepJEB，100 TRAIN CAD×3载荷；knownDEV20 CAD×3载荷，全部原节点。**\n\n'
    text += '| 方法 | 位移误差 | 应力误差 | 热点误差 | 累计更新窗口分钟 |\n|---|---:|---:|---:|---:|\n'
    for n in chosen:
        v = values(n); text += f'| {labels[n]} | {v[0]*100:.2f}% | {v[1]*100:.2f}% | {v[2]*100:.2f}% | {cost(n)/60:.2f} |\n'
    mlp = values('FNO5_MLP_RESIDUAL'); text += f'\n另一残差MLP的位移／应力／热点为{mlp[0]*100:.2f}%／{mlp[1]*100:.2f}%／{mlp[2]*100:.2f}%。四方法完整指标、每载荷20例、配对20CAD及最坏病例保存在metrics.json。\n'
    br,base,full = values(best_residual),values('FNO100_PHYSICAL_1500'),values('FNO100_PHYSICAL_3000')
    text += f'\n最佳残差相对原FNO5：位移{(br[0]-base[0])*100:+.2f}、应力{(br[1]-base[1])*100:+.2f}、热点{(br[2]-base[2])*100:+.2f}个百分点；相对完整FNO10为{(br[0]-full[0])*100:+.2f}／{(br[1]-full[1])*100:+.2f}／{(br[2]-full[2])*100:+.2f}个百分点。\n'
    text += f'\n本轮有限方法池综合领先：{", ".join(labels[n] for n in leaders)}。两残差未通过既定联合／应力各改善3个百分点的推进门槛；不自动追加40遍或24,000步，不宣布最终架构上限。\n'
    tp = d.read(s['training_plan_identity']['path']); train = d.read(d.path(tp['output'])/'summary.json'); te = d.read(d.path(tp['output'])/'external_terminal.json')
    for identity in r['training_binding'].values():
        assert d.identity(identity['path'],hashed=True) == identity
    assert train['status'] == 'completed' and train['closed'] and not train['active'] and te['within_external_budget']
    assert train['methods'][named[0]]['model_profile']['trainable_parameters'] == 26884
    assert train['methods'][named[1]]['model_profile']['trainable_parameters'] == 7745920
    text += f'\n两残差共同TRAIN整轮{te["elapsed_epoch_seconds"]/60:.2f}分钟，包含共享缓存、一次300例冻结全局特征、初始化、更新、原节点读出、保存和退出。knownDEV整轮{e["elapsed_epoch_seconds"]:.2f}秒。MLP／Hash新增更新窗口分别{train["methods"][named[0]]["training_epoch_seconds"]:.2f}／{train["methods"][named[1]]["training_epoch_seconds"]:.2f}秒。表中的累计窗口计入已有teacher5一次；共享新缓存和旧teacher整轮IO另计，不重复归零。UTC／monotonic跨度不同，两轴均保留，不冒充精确纯CUDA时间。\n'
    text += '\n真实最大325,387节点在训练autograd开启时初始预测逐点等于teacher；mutable梯度完整有限、冻结teacher/reference无梯度且前后hash不变，随后两arm各1500次更新。原R64输入网格只供一次冻结特征提取，消费后释放；原节点/stencil保留。采样GPU总峰值约'+f'{train["peak_total_device_used_bytes_sampled"]/1e9:.2f}'+'GB，上限30.5GB；外部workspace精确峰值未知。\n'
    text += '\nteacher此前5遍，mutable decoder额外5遍，reference0；不是fresh5或所有参数10。完整FNO10继承Adam并采用第6–10遍顺序；残差重新初始化Adam及5遍顺序，属于效果／成本比较。MLP 26,884 trainable，官方TCNN实际7,745,920 trainable，另有同尺寸冻结reference；FP32对原生FP16/FP32 master，容量／精度不是单因素因果对照。残差加在encoded4后经旧TRAIN40-only scaler物理解码，signed U(mm)/S(MPa)直接监督，非teacher蒸馏。\n'
    text += '\n先提交两arm×60=120份finite全N encoded/physical不可变预测并验hash，随后共享knownGT60一次。旧FNO5/10只复用已评分标量，不重读PT/GT或重跑推理。knownDEV已用于探索，不是新盲测；FINAL本轮0且先前已揭盲，不用于迭代结论。连续接口的C0取决于显式native25延伸，尚未测独立离节点FEM精度。\n'
    out = d.path('results/geobc_operator_residual_v46'); assert not (out/'manifest.json').exists(); out.mkdir(parents=True,exist_ok=True)
    d.write(out/'metrics.json',dict(readout=a,training=train,training_external=te,readout_external=e,
        source_identities={n:d.identity(folder/n,hashed=True) for n in ('summary.json','status.json','external_terminal.json')},
        training_cost_seconds={n:cost(n) for n in rows},table_names=chosen,all_tied_leaders=leaders,
        exporter_numeric_GT_PT_array_GPU_reads=0,exporter_replays=0))
    (out/'BRIEF.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = ['FNO100_PHYSICAL_1500','FNO5_MLP_RESIDUAL','FNO5_TCNN_RESIDUAL','FNO100_PHYSICAL_3000']
    fig,axes = plt.subplots(1,3,figsize=(12,4.4),constrained_layout=True)
    for index,(ax,title) in enumerate(zip(axes,['Displacement relative L2 (%)','Stress relative L2 (%)','Top 1% abs-stress error (%)'])):
        heights = [values(n)[index]*100 for n in order]
        bars = ax.bar(range(4),heights,color=['#8292a6','#54a9b8','#b178d0','#e19a41'])
        ax.set_xticks(range(4),[labels[n].replace(' + ','\n+ ') for n in order],fontsize=9)
        ax.set_ylim(0,max(heights)*1.2); ax.set_title(title,fontsize=10); ax.spines[['top','right']].set_visible(False)
        for bar,value in zip(bars,heights): ax.text(bar.get_x()+bar.get_width()/2,value+.6,f'{value:.2f}',ha='center',fontsize=9)
    fig.suptitle('Original DeepJEB | 20 known DEV CAD x 3 loads | Full native nodes',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180); plt.close(fig)
    files = [out/'BRIEF.md',out/'metrics.json',out/'comparison.png']; assert all(f.exists() and f.stat().st_size > 0 for f in files)
    elapsed = time.monotonic()-started
    d.write(out/'manifest.json',dict(status='completed',scientific_digest=p['scientific_digest'],actual_CPU_export_seconds=elapsed,
        files={f.name:d.identity(f,hashed=True) for f in files},actual_numeric_GT_PT_array_GPU_reads=0,visual_acceptance_pending=True))
    print(dict(status='completed_CPU_export',CPU_seconds=elapsed,output=d.relative(out),leaders=leaders))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--export',action='store_true'); args = parser.parse_args()
    if args.export: export()
