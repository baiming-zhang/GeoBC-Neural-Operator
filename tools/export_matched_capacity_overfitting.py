"""Report exact 1000-update endpoints of parameter-matched fitting."""
import argparse,csv,json,sys
from pathlib import Path

def write_plot(rows,out):
    """Render already exported scalar rows without model or target reads."""
    try:import matplotlib
    except ModuleNotFoundError:
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'.local_deps'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt;import numpy as np
    names={'mlp':'MLP','ngp4':'NGP4','hashgrid':'TorchHash','tcnn_hash':'TCNN','fno':'FNO','transolver':'Transolver','transolverpp':'Transolver++','ts3':'TS3'}
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
    for ax,group in zip(axes,['single','five']):
        data=[r for r in rows if r['group']==group];x=np.arange(8)
        for shift,key,label,color,padding in [(-.18,'U_percent','U','#247fa9',3),(.18,'S_percent','Signed S','#ed3a94',14)]:
            bars=ax.bar(x+shift,[float(r[key]) for r in data],width=.34,label=label,color=color)
            for annotation in ax.bar_label(bars,fmt='%.2f',padding=padding,fontsize=7):annotation.set_bbox(dict(facecolor='white',edgecolor='none',pad=.2))
        ax.scatter(x,[float(r['worst_field_percent']) for r in data],marker='_',s=160,color='#172638',label='Worst case/field')
        ax.axhline(1,color='#172638',linestyle='--',label='1% goal');ax.set_yscale('log');ax.set_ylim(.01,300);ax.set_xticks(x,[names[r['method']] for r in data],rotation=35,ha='right');ax.set_ylabel('Full original-node error (%)');ax.set_title(('Single sample' if group=='single' else 'Five samples')+' / exact step 1000');ax.legend(frameon=False,fontsize=8)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

def run(root):
    s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text());c=json.loads(Path('configs/matched_capacity_overfitting.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or not t['within_budget'] or t['active'] or not t['closed']:raise ValueError('Actual closed successful terminal required')
    out=Path('results/matched_capacity_overfitting');out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported')
    rows=[];cases=[]
    for name,a in s['arms'].items():
        m=a.get('endpoint')
        if a['status']!='completed' or a['updates']!=1000 or m is None or m['step']!=1000:raise ValueError('All exact1000 endpoints required; no best/partial substitution')
        if abs(a['parameters']/c['parameter_target']-1)>c['parameter_tolerance']:raise ValueError('Parameter mismatch')
        rows.append(dict(experiment=name,method=a['method'],group=a['group'],parameters=a['parameters'],updates=1000,U_percent=m['mean_U_percent'],S_percent=m['mean_S_percent'],worst_field_percent=m['worst_field_percent'],goal_met=m['worst_field_percent']<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic']))
        cases.extend(dict(experiment=name,sample=k,**v) for k,v in m['per_case'].items())
    if len(rows)!=16 or len(cases)!=48:raise ValueError('Fixed denominator incomplete')
    for name,data in [('summary.csv',rows),('per_case.csv',cases)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write_plot(rows,out)
    text=['# 参数量与1000步匹配的单/五样本拟合','\n8种实现约460万有效可训练参数、容差±1%。每项fresh、固定seed2026、相同样本顺序及完整原节点联合U/S损失，Adam固定.001。每次更新一个完整CAD×载荷：单样本1000遍、五样本各200遍。所有方法实际训练1000次，包括提前达1%的方法。表中严格为1000步端点，不是历史最佳或早停结果。','\n容量匹配调整了宽度、深度、Hash容量或FNO查询头；配置见[matched_capacity_overfitting.json](../../configs/matched_capacity_overfitting.json)。架构与表示仍不同，不声称数学意义上的完全公平、普遍架构排名或泛化精度。NGP4保留四层编码，但扩宽其MLP，区别原小容量NGP。TS++仍保留官方随机Gumbel，端点为一次共享完整读出。','\n|组|方法|参数|U均值%|S均值%|最差字段%|全样本两字段<1%|UTC秒|','|---|---|---:|---:|---:|---:|---|---:|']
    text += [f"|{r['group']}|{r['method']}|{r['parameters']}|{r['U_percent']:.4f}|{r['S_percent']:.4f}|{r['worst_field_percent']:.4f}|{'是' if r['goal_met'] else '否'}|{r['UTC_seconds']:.2f}|" for r in rows]
    text+=['\n![匹配容量1000步结果](comparison.png)','\n旧参数未匹配的拟合与追加Hash结果保留在[sample_overfitting](../sample_overfitting/README.md)，不混入此表。']
    (out/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps({'config':c,'terminal_seconds':t['elapsed_seconds'],'summary':rows,'per_case':cases,'scientific_array_reads_by_exporter':0},indent=2)+'\n')
    print(json.dumps({'arms':16,'case_rows':48,'GT_body_reads':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
