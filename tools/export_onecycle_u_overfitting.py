"""Export only scalar records from the complete all-method OneCycle U-only run."""
import argparse,csv,json,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
from tools.export_u_only_overfitting import NAMES


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_plot(rows,out,title_suffix='actual stop endpoints'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    low=max(min(min(r['U_percent'] for r in rows)*.5,.2),1e-6)
    high=max(max(r['worst_U_percent'] for r in rows)*2,2.)
    for ax,group in zip(axes,('single','five')):
        data=[next(r for r in rows if r['method']==m and r['group']==group) for m in NAMES]
        x=np.arange(len(data));bars=ax.bar(x,[r['U_percent'] for r in data],color='#247fa9',width=.65,label='Mean case U')
        for index,r in enumerate(data):
            ax.annotate(f"{r['U_percent']:.4g}",(index-.08,r['U_percent']),xytext=(-5,7),textcoords='offset points',ha='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
        ax.scatter(x+.31,[r['worst_U_percent'] for r in data],color='#d26a33',marker='D',s=20,label='Worst case U',zorder=4)
        ax.axhline(1,color='#172638',ls='--',label='1% goal')
        ax.set_yscale('log');ax.set_ylim(low,high);ax.set_ylabel('Full original-node U-vector error (%)')
        ax.set_xticks(x,[f"{NAMES[r['method']]}\n{r['updates']} steps" for r in data],rotation=35,ha='right',fontsize=8)
        ax.set_title(group+' / '+title_suffix,pad=35)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,ncol=3,fontsize=8)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def run(root,config_path):
    c=json.loads(config_path.read_text());s=json.loads((root/'summary.json').read_text())
    if not c.get('uniform_training_parameters') or c.get('learning_rate_by_method') or c['learning_rate']!=1e-3 or c['max_updates_per_arm']!=5000:raise ValueError('Uniform5000 peak1e-3 protocol required')
    terminal=json.loads((root/'terminal.json').read_text());launch=json.loads((root/'launch.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or terminal['exit_code']!=0 or not terminal['within_budget'] or terminal['active'] or not terminal['closed']:raise ValueError('Successful closed process required')
    expected={g+'_'+m for g in ('single','five') for m in NAMES}
    if set(s['arms'])!=expected or set(c['methods'])!=set(NAMES) or s['source_reads']!=5 or s['stress_label_reads'] or s['heldout_reads']:raise ValueError('All16 new U-only arms and shared5 TRAIN cache required')
    rows=[];cases=[];curves=[]
    for name,a in s['arms'].items():
        endpoint=a.get('endpoint');step=a['updates'];reason=a.get('stop_reason')
        if a['status']!='completed' or not endpoint or endpoint['step']!=step or a['output_channels']!=3 or not a.get('actual_full_node_backward') or a.get('partial_backward') or a.get('optimizer_step_in_progress'):raise ValueError('Complete actual stopping endpoint required')
        if reason not in ('update_cap','early_stopping_plateau') or not 0<step<=c['max_updates_per_arm']:raise ValueError('Unexpected stop or incomplete budget-limited arm')
        if reason=='update_cap' and step!=c['max_updates_per_arm']:raise ValueError('Update cap not reached')
        if reason=='early_stopping_plateau':
            e=a['early_stopping'];rule=c['early_stopping']
            if step<rule['min_updates'] or e['checked_step']!=step or e['updates_without_improvement']<rule['patience_updates']:raise ValueError('Early stopping rule was not met')
        if any(abs(a[k]/c['parameter_target']-1)>c['parameter_tolerance'] for k in ('parameters','active_parameters')):raise ValueError('Actual registered and active capacity must match')
        if set(endpoint['per_case'])!=set(c[a['group']]) or a.get('endpoint_predictions_saved')!=len(c[a['group']]) or any(set(v)!={'U_percent'} for v in endpoint['per_case'].values()):raise ValueError('Incomplete U-only case endpoint')
        peak=c.get('learning_rate_by_method',{}).get(a['method'],c['learning_rate'])
        if a['maximum_learning_rate']!=peak:raise ValueError('Undeclared OneCycle peak')
        row=dict(experiment=name,method=a['method'],group=a['group'],parameters=a['parameters'],active_parameters=a['active_parameters'],updates=step,stop_reason=reason,maximum_learning_rate=peak,learning_rate_last_update=a['learning_rate_last_update'],U_percent=endpoint['mean_U_percent'],worst_U_percent=endpoint['worst_field_percent'],all_cases_U_below_1_percent=endpoint['worst_field_percent']<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],best_observed_step=a['best_step'],best_observed_worst_U_percent=a['best']['worst_field_percent'])
        rows.append(row);cases.extend(dict(experiment=name,sample=k,updates=step,U_percent=v['U_percent']) for k,v in endpoint['per_case'].items())
        for line in (root/name/'metrics.jsonl').read_text().splitlines():
            m=json.loads(line)
            curves.append(dict(experiment=name,step=m['step'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],learning_rate_last_update=m.get('learning_rate_last_update')))
    if len(cases)!=48 or not all(math.isfinite(r['U_percent']) and math.isfinite(r['worst_U_percent']) for r in rows):raise ValueError('48 finite endpoint case rows required')
    out=Path(c['report_directory']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported; replot only from exported scalars')
    write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases);write_csv(out/'learning_curves.csv',curves)
    write_plot(rows,out)
    ended=datetime.fromtimestamp(launch['entry_epoch']+terminal['elapsed_seconds']['UTC'],timezone(timedelta(hours=8))).isoformat()
    lines=['# 最新全部8方法U-only：统一OneCycle、5000步上限与早停','',f"北京时间{ended}结束。本轮全部16项重新从seed2026初始化，最多5000次完整原节点更新；以下主表只使用实际停止时的模型端点，不混入旧结果或最佳checkpoint。",'',
        '全部方法统一Adam固定betas(.9,.999)、eps1e-8、weight_decay0；OneCycleLR每次optimizer.step后调用一次，前10%步数余弦升温，后90%余弦下降，峰值均1e-3，div_factor=25、final_div_factor=1000，不循环动量。第1步4e-5，第500步峰值1e-3，第5000步实际4e-8；梯度L2范数裁剪1。所有方法学习率、优化器、调度、最大步数、早停、评价间隔、损失和样本顺序相同。OneCycle按步数调度，不是根据loss自动寻找学习率。', '',
        '每50步完整评价当前组的所有拟合样本，以最差样本U误差监视早停。最早1500步，连续600步未获得至少0.1%相对改善时停止；阈值相对于最近一次达到该幅度的改善计算。此处0.1%为误差自身的相对变化，不是0.1个百分点；监视TRAIN误差，无验证集或held-out调参。达到1%不单独触发早停。步数及停止原因逐项报告。', '',
        '单样本506_533/ver；五样本按506_533/ver、506_25/tor、10_30/ver、10_421/hor、10_460/dia循环。一次更新完整一CAD×载荷全部原节点；达到5000步则single5000遍、five各1000遍，提前停止按实际cursor计数。仅signed Ux/Uy/Uz(mm)和U-vector relative L2平方损失；归一化XYZ+25原条件，无sampleID、物理U、teacher或应力。5份TRAIN NPZ共享各读一次，只用拟合样本计算输出RMS。', '',
        '实际注册与真实full-node backward参与参数均460万±1%。Transolver++保留上游未参与无条件分支的embedding，活跃参数单列；官方eval随机Gumbel单次读出保留，早停监视也受该随机性影响。FP32/TF32；TCNN原生FP16与FP32master。一个seed、预设配方和训练拟合不能证明架构上限、数学完全公平或泛化排名。', '',
        f"全阶段实际成本{terminal['elapsed_seconds']['UTC']:.2f}秒UTC / {terminal['elapsed_seconds']['monotonic']:.2f}秒单调时钟，含导入、缓存、构造、训练、评价、保存和退出。退出码0且在预定双时钟预算内；stress/heldout读取0。完整端点预测及model/Adam/scheduler/四RNG/order/cursor/early-stop恢复状态留本地。导出仅访问标量记录，科学数组读取0。此前3000步/不同峰值轮按用户新要求中断，已用758.16秒UTC / 696.08秒单调时钟，独立保留且不混入本轮主表，也不是本轮免费恢复。", '',
        '|方法|单样本U%|单样本步数|五样本均值U%|五样本最差U%|五样本步数|','|---|---:|---:|---:|---:|---:|']
    for method in NAMES:
        one=next(r for r in rows if r['method']==method and r['group']=='single');five=next(r for r in rows if r['method']==method and r['group']=='five')
        lines.append(f"|{NAMES[method]}|{one['U_percent']:.4f}|{one['updates']}|{five['U_percent']:.4f}|{five['worst_U_percent']:.4f}|{five['updates']}|")
    lines+=['','## 实际更新与停止原因','','|组|方法|更新|停止原因|最后实际LR|活跃参数|所有样本U<1%|UTC秒|','|---|---|---:|---|---:|---:|---|---:|']
    for r in rows:lines.append(f"|{r['group']}|{NAMES[r['method']]}|{r['updates']}|{r['stop_reason']}|{r['learning_rate_last_update']:.5g}|{r['active_parameters']}|{'是' if r['all_cases_U_below_1_percent'] else '否'}|{r['UTC_seconds']:.2f}|")
    lines+=['','![实际停止端点比较](comparison.png)','','[16项汇总](summary.csv) · [48行逐样本](per_case.csv) · [全程评价标量](learning_curves.csv) · [配置与成本](metrics.json)','','历史对照单独保留：[1000步固定LR](../matched_capacity_u_overfitting/README.md) · [Transolver系列1000步warmup/cosine](../stabilized_u_overfitting/README.md)。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(dict(config=c,terminal_seconds=terminal['elapsed_seconds'],summary=rows,per_case=cases,source_reads=5,stress_label_reads=0,heldout_reads=0,scientific_array_reads_by_exporter=0),indent=2)+'\n')
    print(json.dumps(dict(arms=len(rows),per_case_rows=len(cases),scientific_array_reads=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();run(args.run,args.config)
