"""Scalar-only CPU one-page original TS5/10 versus FNO10 comparison."""
import argparse
import time


def export():
    started=time.monotonic()
    import math,csv,json
    from geobc import available_cad_pilot_data as d
    from geobc.global_bc_training_contract_v48 import terminal
    from geobc.original_TS_readout_contract_v51 import training_terminal,complete_physical,METHOD
    src='tools/export_geobc_original_TS_v51.py';review=d.read('notes/geobc_original_TS_v51_exporter_root_review.json')
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_export_released'] and review['source_sha256']==d.sha(src)
    binding,p,a,e=terminal('notes/geobc_original_TS_v51_plan.json');s=p['scientific'];r=d.read(p['root_review'])
    assert a['protocol_qualified'] and a['committed_finite_predictions']==a['owned_GT_read_attempts']==a['owned_GT_array_reads']==60
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    assert max(e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=1800
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    for n,h in p['source_pins'].items():assert d.sha(n)==h
    tb,tp,train,te,m=training_terminal();assert tb==r['training_binding']
    rows=a['candidates'];order=['FNO100_PHYSICAL_1500','FNO100_PHYSICAL_3000','R16_TRANSOLVER_1500',METHOD]
    assert set(rows)==set(order)
    for row in rows.values():complete_physical(row['full60'],s['case_keys'])
    def values(n):
        v=rows[n]['full60'];return [v['case_mean_u_relative_l2'],v['case_mean_stress_relative_l2'],v['case_mean_stress_hotspot_relative_l2']]
    def cost(n):
        return rows[n]['trajectory_training_epoch_seconds'] if n in [METHOD,'FNO100_PHYSICAL_3000'] else rows[n]['terminal_campaign_training_epoch_seconds']
    for n in rows:assert all(math.isfinite(v) for v in values(n)) and math.isfinite(cost(n)) and cost(n)>0
    scores={n:v['full60']['selection_score'] for n,v in rows.items()}
    leaders=sorted(n for n,v in scores.items() if v==min(scores.values()));assert leaders==a['eligible_best']
    tsnames=['R16_TRANSOLVER_1500',METHOD];bestTS=sorted(n for n in tsnames if scores[n]==min(scores[x] for x in tsnames))
    labels=dict(FNO100_PHYSICAL_1500='FNO5',FNO100_PHYSICAL_3000='FNO10',R16_TRANSOLVER_1500='Transolver5',R16_TRANSOLVER_PHYSICAL_3000='Transolver10')
    chosen=['FNO100_PHYSICAL_3000',METHOD,'R16_TRANSOLVER_1500']
    text='# 原版Transolver：5→10遍收敛检查\n\n'
    text+='原版DeepJEB，100 TRAIN CAD×三载荷；knownDEV20 CAD×三载荷、全部原节点。位移／应力保持signed mm／MPa。\n\n'
    text+='| 方法 | 位移误差 | 应力误差 | 热点误差 | 累计更新窗口分钟 |\n|---|---:|---:|---:|---:|\n'
    for n in chosen:
        u,st,hot=values(n);text+=f'| {labels[n]} | {u*100:.2f}% | {st*100:.2f}% | {hot*100:.2f}% | {cost(n)/60:.2f} |\n'
    gate=a['DEV_gates'][METHOD];delta=[values(METHOD)[i]-values('FNO100_PHYSICAL_3000')[i] for i in range(3)]
    text+=f'\nTS5→10位移／应力／联合误差变化：{gate["U_difference"]*100:+.2f}／{gate["S_difference"]*100:+.2f}／{gate["primary_difference"]*100:+.2f}个百分点；原定联合和应力各改善3个百分点的门槛'+('通过' if gate['passed'] else '未通过')+'。\n'
    text+=f'\n同10遍TS相对FNO的位移／应力／热点差距：{delta[0]*100:+.2f}／{delta[1]*100:+.2f}／{delta[2]*100:+.2f}个百分点；正值表示TS更差。累计更新窗口成本比{cost(METHOD)/cost("FNO100_PHYSICAL_3000"):.2f}倍，含检查点保存，非纯CUDA。当前这四方法池综合领先：'+', '.join(labels[n] for n in leaders)+'；保留全部并列。\n'
    text+=f'\n追加1500更新实际{train["training_epoch_seconds"]:.2f}秒；本段训练整轮{te["elapsed_epoch_seconds"]/60:.2f}分钟，含cache/恢复/资格/保存/TRAIN300读出/退出，knownDEV整轮{e["elapsed_epoch_seconds"]:.2f}秒。旧TS5更新窗口{train["parent_cost"]["training_epoch_seconds"]:.2f}秒、单arm{train["parent_cost"]["parent_arm_elapsed_epoch_seconds"]:.2f}秒及共享cache{train["parent_cost"]["old_shared_cache_epoch_seconds"]:.2f}秒分别保留，旧三TS整个campaign不可都归给原版。UTC和monotonic两轴原值保留，不缩放或零填旧成本。采样总GPU峰值{train["peak_total_device_used_bytes_sampled"]/1e9:.2f}GB，上限30.5GB；外部workspace精确峰值未知。\n'
    text+='\n延续原版R16完整模型和Adam1500状态、四类RNG、曝光量及10遍固定顺序后缀；各case从5到10次，总cursor3000，只新增1500。一次own PT hash/CPU load真实资格通过，最大325,387节点三部分/两leaf梯度有限非零，资格前后模型及Adam tensor hash一致、RNG恢复；无GBC分支、容量／训练目标变化或旧更新重放。TRAIN physical300只读一次，端点复用缓存300次。\n'
    text+='\n先提交60份finite全N encoded/physical/hash预测并完整重验，再共享knownGT60一次，不因科学负结果挑选病例。FNO5／10、TS5只复用固定同60的旧标量，没有重读PT/GT或推理。每载荷20例、配对20CAD、四行完整指标及失败分母见metrics.json／CSV；训练曲线是联合物理loss，不能代替单独应力验证。\n'
    text+='\n同10遍的FNO与TS共享划分/scaler/seed/物理损失/Adam/顺序和曝光量，R64 versus R16以及模型容量不同；不是架构单因素因果或最终上限。knownDEV已用于迭代、单seed；FINAL本轮0且先前已揭盲，不用于调参。结果只决定研究优先级，不自动追加40遍或24,000步。连续查询仍依赖显式native25延伸，尚未测独立离节点FEM准确性。\n'
    logs=[json.loads(x) for x in (d.path(tp['output'])/'loss.jsonl').read_text().splitlines()]
    assert len(logs)==1500 and [x['step'] for x in logs]==list(range(1501,3001)) and all(math.isfinite(x['physical_relative_squared']) for x in logs)
    curves=d.read(tp['scientific']['scalar_diagnostic']['path'])['curves']['original_TS5']+[sum(x['physical_relative_squared'] for x in logs[i:i+300])/300 for i in range(0,1500,300)]
    assert len(curves)==10
    out=d.path('results/geobc_original_TS_v51');assert not (out/'manifest.json').exists();out.mkdir(parents=True,exist_ok=True)
    d.write(out/'metrics.json',dict(readout=a,training=train,training_external=te,readout_external=e,training_binding=tb,readout_binding=binding,loss_per_pass=curves,
        table_names=chosen,all_tied_leaders=leaders,all_tied_best_TS=bestTS,TS10_minus_FNO10_physical=delta,training_cost_seconds={n:cost(n) for n in rows},exporter_numeric_GT_PT_array_GPU_reads=0))
    (out/'BRIEF.md').write_text(text,encoding='utf-8')
    with (out/'all_methods.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['name','U_relative_L2','S_relative_L2','stress_hotspot_relative_L2','trajectory_training_window_seconds','inference_monotonic_seconds','case_denominator'])
        for n in order:writer.writerow([n,*values(n),cost(n),rows[n]['inference_seconds'],60])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),constrained_layout=True)
    for i,(ax,title) in enumerate(zip(axes,['Displacement relative L2 (%)','Stress relative L2 (%)','Top 1% abs-stress error (%)'])):
        v=[values(n)[i]*100 for n in order];bars=ax.bar(range(4),v,color=['#92a1b3','#3ca0b4','#bfac9a','#a974c6'])
        ax.set_xticks(range(4),[labels[n] for n in order],fontsize=9);ax.set_ylim(0,max(v)*1.2);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
        for bar,value in zip(bars,v):ax.text(bar.get_x()+bar.get_width()/2,value+.6,f'{value:.2f}',ha='center',fontsize=9)
    fig.suptitle('Original DeepJEB | 20 known DEV CAD x 3 loads | Full native nodes',fontsize=12);fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5),constrained_layout=True);ax.plot(range(1,11),curves,marker='o',label='Original Transolver')
    ax.axvline(5.5,linestyle='--',color='#999999',label='Own checkpoint resumed');ax.set(xlabel='Complete TRAIN pass',ylabel='Mean joint physical squared loss',xticks=range(1,11));ax.legend()
    fig.savefig(out/'training.png',dpi=180);plt.close(fig)
    files=[out/n for n in ['BRIEF.md','metrics.json','all_methods.csv','comparison.png','training.png']];assert all(n.exists() and n.stat().st_size>0 for n in files)
    elapsed=time.monotonic()-started;d.write(out/'manifest.json',dict(status='completed',scientific_digest=p['scientific_digest'],actual_CPU_export_seconds=elapsed,files={n.name:d.identity(n,hashed=True) for n in files},actual_numeric_GT_PT_array_GPU_reads=0,visual_acceptance_pending=True))
    print(dict(status='completed_CPU_export',CPU_seconds=elapsed,leaders=leaders,bestTS=bestTS))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.export:export()
