"""Export terminal overfitting scalar metrics; never open labels or predictions."""
import argparse,csv,json,sys
from pathlib import Path


def run(root):
    state=json.loads((root/'summary.json').read_text());terminal=json.loads((root/'terminal.json').read_text())
    if state['status']!='completed' or state['active'] or not state['closed'] or terminal['exit_code']!=0 or not terminal['within_budget'] or terminal['active'] or not terminal['closed']:
        raise ValueError('Requires closed successful terminal stage, irrespective of precision goal')
    if len(state['arms'])!=16:raise ValueError('All 16 attempted arms required; no subset leaderboard')
    out=Path('results/sample_overfitting');dest=out/'current_metrics.json'
    if dest.exists():raise FileExistsError('Terminal scalar export already exists')
    aggregate=[];cases=[]
    for name,a in state['arms'].items():
        m=a['best']
        aggregate.append({'experiment':name,'group':a['group'],'method':a['method'],'status':a['status'],'goal_met':a['goal_met'],'parameters':a.get('parameters'),
                          'updates':a['updates'],'selected_step':a['best_step'],'mean_U_percent':m['mean_U_percent'] if m else None,
                          'mean_S_percent':m['mean_S_percent'] if m else None,'worst_field_percent':m['worst_field_percent'] if m else None,
                          'UTC_seconds':a['elapsed_seconds']['UTC'],'monotonic_seconds':a['elapsed_seconds']['monotonic'],'stop_reason':a.get('stop_reason',a.get('error'))})
        for key in a['cases']:
            metric=m['per_case'].get(key,{}) if m else {}
            cases.append({'experiment':name,'sample':key,'U_percent':metric.get('U_percent'),'S_percent':metric.get('S_percent'),'selected_step':a['best_step']})
    if len(cases)!=48:raise ValueError('Fixed single/five case denominator')
    for name,rows in [('current_summary.csv',aggregate),('current_per_case.csv',cases)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics={'kind':'training_sample_overfitting','all_goal_met':state['all_goal_met'],'source_reads':state['source_reads'],'heldout_reads':state['heldout_reads'],
             'terminal_seconds':terminal['elapsed_seconds'],'aggregate':aggregate,'per_case':cases,'goal':'Every case U3 and signed S strictly below 1% at one shared checkpoint'}
    try:import matplotlib
    except ModuleNotFoundError:
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'.local_deps'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt;import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
    for ax,group in zip(axes,['single','five']):
        rows=[r for r in aggregate if r['group']==group];x=np.arange(len(rows))
        for offset,key,label,color in [(-.18,'mean_U_percent','U mean','#247fa9'),(.18,'mean_S_percent','Signed S mean','#ed3a94')]:
            values=[r[key] if r[key] is not None else float('nan') for r in rows]
            bars=ax.bar(x+offset,values,width=.34,label=label,color=color);ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
        worst=[r['worst_field_percent'] if r['worst_field_percent'] is not None else float('nan') for r in rows]
        ax.scatter(x,worst,color='#172638',marker='_',s=160,label='Worst case/field',zorder=4)
        ax.axhline(1,color='#172638',linestyle='--',linewidth=1,label='1% goal')
        ax.set_yscale('log');ax.set_ylim(.01,300);ax.set_xticks(x,[r['method'] for r in rows],rotation=35,ha='right');ax.set_ylabel('Full original-node error (%)');ax.set_title(group+' sample fitting',color='#172638');ax.legend(fontsize=8,frameon=False)
    fig.savefig(out/'fitting_comparison.png',dpi=180);plt.close(fig)
    text=['\n## 新实验实际结果\n','完整16臂均已尝试；运行完成不等于1%目标通过。每行是同一个最差字段指标最优 checkpoint，不逐样本/字段另挑最低值。','\n|样本组|方法|U均值%|S均值%|最差样本/字段%|两字段全部<1%|实际更新|UTC耗时秒|','|---|---|---:|---:|---:|---|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    for r in aggregate:text.append(f"|{r['group']}|{r['method']}|{fmt(r['mean_U_percent'])}|{fmt(r['mean_S_percent'])}|{fmt(r['worst_field_percent'])}|{'是' if r['goal_met'] else '否'}|{r['updates']}|{r['UTC_seconds']:.2f}|")
    text+=['\n![原节点拟合误差](fitting_comparison.png)','\n逐样本误差见 [current_per_case.csv](current_per_case.csv)，含失败/缺失的固定48行分母。完整阶段成本见 [current_metrics.json](current_metrics.json)。这是训练拟合结果，不是新的泛化结论。']
    with (out/'README.md').open('a',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    dest.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'arms':len(aggregate),'case_rows':len(cases),'all_goal_met':state['all_goal_met'],'GT_body_reads':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
