"""One-page scalar-only CPU report of completed global BC routing screens."""
import argparse
import time


def export():
    started=time.monotonic()
    import math,csv
    from geobc import available_cad_pilot_data as d
    from geobc.global_bc_training_contract_v48 import terminal
    from geobc.global_bc_readout_contract_v49 import training_terminal,complete_physical
    src='tools/export_geobc_global_BC_v49.py'
    review=d.read('notes/geobc_global_BC_v49_exporter_root_review.json')
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_export_released'] and review['source_sha256']==d.sha(src)
    binding,p,a,e=terminal('notes/geobc_global_BC_v49_plan.json');s=p['scientific'];r=d.read(p['root_review']);f=d.path(p['output'])
    assert a['protocol_qualified'] and a['committed_finite_predictions']==120 and a['owned_GT_read_attempts']==a['owned_GT_array_reads']==60
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    assert max(e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=1800
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    for n,h in p['source_pins'].items():assert d.sha(n)==h
    tb,tp,train,te,m=training_terminal();assert tb==r['training_binding']
    rows=a['candidates'];new=[x['name'] for x in s['candidates']]
    assert set(rows)==set(new+['FNO100_PHYSICAL_1500','R16_TRANSOLVER_1500','FNO100_PHYSICAL_3000'])
    for row in rows.values():complete_physical(row['full60'],s['case_keys'])
    def values(n):
        v=rows[n]['full60'];return [v['case_mean_u_relative_l2'],v['case_mean_stress_relative_l2'],v['case_mean_stress_hotspot_relative_l2']]
    def cost(n):
        v=rows[n]
        return v['trajectory_training_epoch_seconds'] if n=='FNO100_PHYSICAL_3000' else v['terminal_campaign_training_epoch_seconds']
    for n in rows:assert all(math.isfinite(v) for v in values(n)) and math.isfinite(cost(n)) and cost(n)>0
    scores={n:r['full60']['selection_score'] for n,r in rows.items()}
    leaders=sorted(n for n,v in scores.items() if v==min(scores.values()));assert leaders==a['eligible_best']
    tsnames=['R16_TRANSOLVER_1500','R16_TRANSOLVER_GLOBAL_BC'];bestTS=sorted(n for n in tsnames if scores[n]==min(scores[x] for x in tsnames))
    chosen=list(dict.fromkeys([leaders[0],bestTS[0],'FNO100_PHYSICAL_1500']))
    labels=dict(FNO9_GLOBAL_BC='FNO5 + 全局BC',R16_TRANSOLVER_GLOBAL_BC='Transolver5 + 全局BC',FNO100_PHYSICAL_1500='原FNO5',R16_TRANSOLVER_1500='原Transolver5',FNO100_PHYSICAL_3000='完整FNO10')
    english=dict(FNO9_GLOBAL_BC='FNO5 + global BC',R16_TRANSOLVER_GLOBAL_BC='TS5 + global BC',FNO100_PHYSICAL_1500='Original FNO5',R16_TRANSOLVER_1500='Original TS5',FNO100_PHYSICAL_3000='Intact FNO10')
    text='# 全局边界条件上下文：两个求解器各5遍\n\n'
    text+='原版DeepJEB：100 TRAIN CAD×3载荷；knownDEV20 CAD×3载荷、全原节点、signed U(mm)/S(MPa)。\n\n'
    text+='| 对照角色 | 位移误差 | 应力误差 | 热点误差 | 更新窗口分钟 |\n|---|---:|---:|---:|---:|\n'
    for n in chosen:
        u,st,hot=values(n);text+=f'| {labels[n]} | {u*100:.2f}% | {st*100:.2f}% | {hot*100:.2f}% | {cost(n)/60:.2f} |\n'
    text+='\n当前五方法池综合领先：'+', '.join(labels[n] for n in leaders)+'。最佳TS：'+', '.join(labels[n] for n in bestTS)+'；单seed、已用于迭代的knownDEV，不是最终求解器或架构上限。\n'
    for n in new:
        g=a['DEV_gates'][n];text+=f'\n{labels[n]}相对同初始化/同5遍的{labels[g["reference"]]}：位移{100*g["U_difference"]:+.2f}、应力{100*g["S_difference"]:+.2f}、联合{100*g["primary_difference"]:+.2f}个百分点；推进门槛'+('通过' if g['passed'] else '未通过')+'（联合和应力各改善至少3个百分点，任一场不恶化超过2个百分点）。\n'
    text+='\n通过只支持优先做容量匹配的零描述符对照；不自动追加40遍/24000步。原native25和输入网格本来就包含载荷/材料，新增分支改变其进入全局32维几何上下文的路径，不是添加新物理信息或强制边界条件。RBE2 CM表示耦合成员，不当作固定SPC。\n'
    text+=f'\n实际两arm训练整轮{te["elapsed_epoch_seconds"]/60:.2f}分钟（共享缓存、输入描述符、初始化和最大全N检查、更新、保存、TRAIN300读出、退出全部计入）；knownDEV整轮{e["elapsed_epoch_seconds"]:.2f}秒。表列仅更新和检查点窗口；共享缓存/描述符一次成本{train["cache_setup_epoch_seconds"]:.2f}/{train["BC_descriptor_epoch_seconds"]:.2f}秒另列于metrics.json，不当纯CUDA时间。UTC与monotonic两轴原值均保留；GPU采样总峰值{train["peak_total_device_used_bytes_sampled"]/1e9:.2f}GB，上限30.5GB，外部workspace精确峰值未知。\n'
    text+='\n两arm均fresh1500更新，每TRAIN case5次，保持各自原完整核心初值、旧TRAIN40-only scaler、相同顺序/Adam/全节点物理损失；增加6496可训练参数及6496冻结初始化参考参数，不能单因素归因给BC信息。真实最大325,387节点开启训练autograd时初始预测与原模型逐点相等，四个训练部分/两leaf梯度通过，冻结参考无梯度且训练前后hash一致。FNO R64与TS R16表示/参数量不同，不是架构容量匹配。FNO10是曝光量上下文，不与fresh5等预算。\n'
    text+='\n两新arm×60=120份finite encoded/physical全节点预测先不可变提交并验hash，随后共享knownGT60一次；旧FNO5/TS5/FNO10只复用已评分标量，不重读PT/GT或重跑推理。每载荷20例、配对20CAD、热点和失败分母、全部5行及训练曲线保存在metrics.json/CSV/PNG。FINAL本轮0且先前已揭盲，不用于本轮迭代。C0连续查询依赖显式native25延伸，尚未测独立离节点FEM精度。\n'
    out=d.path('results/geobc_global_BC_v49');assert not (out/'manifest.json').exists();out.mkdir(parents=True,exist_ok=True)
    curves={}
    for n in new:
        lines=[__import__('json').loads(line) for line in (d.path(tp['output'])/n/'loss.jsonl').read_text().splitlines()]
        assert len(lines)==1500 and [x['step'] for x in lines]==list(range(1,1501))
        assert all(math.isfinite(x['physical_relative_squared']) for x in lines)
        curves[n]=[sum(x['physical_relative_squared'] for x in lines[i:i+300])/300 for i in range(0,1500,300)]
    d.write(out/'metrics.json',dict(readout=a,training=train,training_external=te,readout_external=e,training_binding=tb,readout_binding=binding,
        table_names=chosen,all_tied_leaders=leaders,all_tied_best_TS=bestTS,loss_per_pass=curves,training_cost_seconds={n:cost(n) for n in rows},exporter_numeric_GT_PT_array_GPU_reads=0))
    (out/'BRIEF.md').write_text(text,encoding='utf-8')
    with (out/'all_methods.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['name','U_relative_L2','S_relative_L2','stress_hotspot_relative_L2','training_window_seconds','inference_monotonic_seconds','case_denominator'])
        for n in rows:writer.writerow([n,*values(n),cost(n),rows[n]['inference_seconds'],60])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['FNO100_PHYSICAL_1500','FNO9_GLOBAL_BC','R16_TRANSOLVER_1500','R16_TRANSOLVER_GLOBAL_BC','FNO100_PHYSICAL_3000']
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
    for i,(ax,title) in enumerate(zip(axes,['Displacement relative L2 (%)','Stress relative L2 (%)','Top 1% abs-stress error (%)'])):
        vals=[values(n)[i]*100 for n in order];bars=ax.bar(range(5),vals,color=['#92a1b3','#3ca0b4','#bfac9a','#a974c6','#e1a041'])
        ax.set_xticks(range(5),[english[n].replace(' + ','\n+ ') for n in order],fontsize=8);ax.tick_params(axis='x',rotation=18)
        ax.set_ylim(0,max(vals)*1.2);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.6,f'{v:.2f}',ha='center',fontsize=9)
    fig.suptitle('Original DeepJEB | 20 known DEV CAD x 3 loads | Full native nodes',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5),constrained_layout=True)
    for n,v in curves.items():ax.plot(range(1,6),v,marker='o',label=english[n])
    ax.set(xlabel='Complete TRAIN pass',ylabel='Mean joint physical squared loss',xticks=range(1,6));ax.legend();fig.savefig(out/'training.png',dpi=180);plt.close(fig)
    files=[out/n for n in ['BRIEF.md','metrics.json','all_methods.csv','comparison.png','training.png']]
    assert all(n.exists() and n.stat().st_size>0 for n in files)
    elapsed=time.monotonic()-started
    d.write(out/'manifest.json',dict(status='completed',scientific_digest=p['scientific_digest'],actual_CPU_export_seconds=elapsed,files={n.name:d.identity(n,hashed=True) for n in files},actual_numeric_GT_PT_array_GPU_reads=0,visual_acceptance_pending=True))
    print(dict(status='completed_CPU_export',CPU_seconds=elapsed,leaders=leaders,bestTS=bestTS))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.export:export()
