"""Export strict U-only reruns and a latest-results index with explicit provenance."""
import argparse,csv,json,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
from tools.export_u_only_overfitting import NAMES


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def merge_case_rows(cases):
    old=list(csv.DictReader(Path('results/matched_capacity_u_overfitting/per_case.csv').open()))
    replacements={(r['experiment'],r['sample']):r for r in cases}
    keys={(r['experiment'],r['sample']) for r in old}
    if len(old)!=48 or len(keys)!=48 or not set(replacements).issubset(keys):raise ValueError('Latest case denominator incomplete')
    return [dict(experiment=r['experiment'],sample=r['sample'],U_percent=float(replacements.get((r['experiment'],r['sample']),r)['U_percent']),protocol='warmup_cosine_clip' if (r['experiment'],r['sample']) in replacements else 'fixed_001',source_report='stabilized_u_overfitting' if (r['experiment'],r['sample']) in replacements else 'matched_capacity_u_overfitting') for r in old]


def write_plot(old,rows,out,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),constrained_layout=True)
    values=[float(r['U_percent']) for r in old+rows]+[float(r['worst_U_percent']) for r in rows]
    lower=max(min(min(values)*.4,.2),1e-6);upper=max(max(values)*2,2)
    for ax,group in zip(axes,('single','five')):
        before=[next(r for r in old if r['method']==m and r['group']==group) for m in methods]
        after=[next(r for r in rows if r['method']==m and r['group']==group) for m in methods]
        x=np.arange(len(methods))
        for shift,data,color,label,pad in [(-.2,before,'#adb9c3','Earlier fixed LR .001',3),(.2,after,'#247fa9','New warmup/cosine + clip',13)]:
            bars=ax.bar(x+shift,[float(r['U_percent']) for r in data],width=.36,color=color,label=label)
            ax.bar_label(bars,fmt='%.3f',padding=pad,fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':.2})
        ax.scatter(x+.4,[float(r['worst_U_percent']) for r in after],s=90,marker='_',color='#172638',label='New worst case U')
        ax.axhline(1,ls='--',color='#172638',label='1% goal');ax.set_yscale('log');ax.set_ylim(lower,upper)
        ax.set_xticks(x,[NAMES[m] for m in methods],rotation=30,ha='right');ax.set_ylabel('Full original-node U-vector error (%)');ax.set_title(group+' / exact update 1000',pad=53);ax.legend(frameon=False,fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def run(root,config_path):
    c=json.loads(config_path.read_text());s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text());launch=json.loads((root/'launch.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or not t['within_budget'] or t['active'] or not t['closed']:raise ValueError('Successful closed run required')
    expected={g+'_'+m for g in ('single','five') for m in c['methods']}
    if set(s['arms'])!=expected or s['source_reads']!=5 or s['stress_label_reads']!=0 or s['heldout_reads']!=0:raise ValueError('Complete U-only source and arm contract required')
    rows=[];cases=[]
    for name,a in s['arms'].items():
        m=a.get('endpoint');active=a.get('active_parameters')
        if a['status']!='completed' or a['updates']!=1000 or m is None or m['step']!=1000 or a['output_channels']!=3 or not a.get('actual_full_node_backward'):raise ValueError('Every rerun must have actual exact1000 endpoint')
        if active is None or any(abs(n/c['parameter_target']-1)>c['parameter_tolerance'] for n in (active,a['parameters'])):raise ValueError('Registered and active parameter counts must match')
        if set(m['per_case'])!=set(c[a['group']]) or a.get('endpoint_predictions_saved')!=len(c[a['group']]) or any(set(v)!={'U_percent'} for v in m['per_case'].values()):raise ValueError('Complete U-only endpoint predictions required')
        if not math.isclose(a['learning_rate_last_update'],c['minimum_learning_rate'],rel_tol=1e-9):raise ValueError('Final update did not use scheduled floor')
        row=dict(experiment=name,method=a['method'],group=a['group'],parameters=a['parameters'],active_parameters=active,updates=1000,U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],all_cases_U_below_1_percent=m['worst_field_percent']<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],protocol='warmup_cosine_clip',source_report='stabilized_u_overfitting')
        rows.append(row);cases.extend(dict(experiment=name,sample=k,U_percent=v['U_percent']) for k,v in m['per_case'].items())
    if len(cases)!=6*len(c['methods']):raise ValueError('Case denominator incomplete')
    old=list(csv.DictReader(Path('results/matched_capacity_u_overfitting/summary.csv').open()))
    latest=[]
    for r in old:
        replacement=next((v for v in rows if v['experiment']==r['experiment']),None)
        if replacement:latest.append(replacement)
        else:latest.append(dict(experiment=r['experiment'],method=r['method'],group=r['group'],parameters=int(r['parameters']),active_parameters='',updates=int(r['updates']),U_percent=float(r['U_percent']),worst_U_percent=float(r['worst_U_percent']),all_cases_U_below_1_percent=r['all_cases_U_below_1_percent']=='True',UTC_seconds=float(r['UTC_seconds']),monotonic_seconds=float(r['monotonic_seconds']),protocol='fixed_001',source_report='matched_capacity_u_overfitting'))
    out=Path('results/stabilized_u_overfitting');out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported')
    latest_cases=merge_case_rows(cases)
    write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases);write_csv(out/'latest_all_methods.csv',latest);write_csv(out/'latest_per_case.csv',latest_cases)
    write_plot(old,rows,out,c['methods'])
    completed=datetime.fromtimestamp(launch['entry_epoch']+t['elapsed_seconds']['UTC'],timezone(timedelta(hours=8))).isoformat()
    lines=['# 最新U-only过拟合：训练配方调整后重做','',f"北京时间{completed}完成{len(rows)}项重做，每项fresh seed2026、恰好1000次完整原节点更新。下表直接使用本轮新端点；未重做的方法标注保留旧配方。原固定配方表作为历史保留，不再用其Transolver旧值代表最新结果。",'',
        f"预先固定Adam峰值学习率{c['learning_rate']:g}，前{c['lr_schedule']['warmup_updates']}步线性warmup，再余弦降到{c['minimum_learning_rate']:g}；每步全参数梯度L2范数裁剪{c['gradient_clip_norm']:g}。架构、U-only目标、样本顺序、参数匹配、精度和1000更新规则保持不变。配方同时改变学习率过程和裁剪，是重做结果，不是仅学习率单变量因果实验。",'',
        '单样本506_533/ver；五样本另含506_25/tor、10_30/ver、10_421/hor、10_460/dia。一更新完整一个CAD×载荷，single1000遍、five每样本200遍。输入归一化XYZ+25维原条件，输出signed Ux/Uy/Uz(mm)，完整U向量相对L2平方损失，输出RMS仅用拟合样本；无样本ID、物理U直通、teacher、应力标签或held-out。FP32/TF32，TCNN若在重做范围则仍原生FP16/FP32master；TS++保留官方eval随机Gumbel，单次端点不保证重复读出都通过。', '',
        '所有重做项的注册及实际梯度参与参数数目均检查在460万±1%内；++无条件输入分支的原生embedding不参与，单独报告活跃参数。验收是每样本完整原节点FP64 U误差均<1%，不是pooled均值。只用严格1000步端点，不用最佳checkpoint或提前通过的值替代。新结果不保证每个方法更优或达1%；一个seed和固定配方不代表架构上限或泛化排名。', '',
        f"新阶段总成本{t['elapsed_seconds']['UTC']:.2f}秒UTC / {t['elapsed_seconds']['monotonic']:.2f}秒单调时钟，含导入、缓存、构造、训练、完整读出、保存及退出；终止码0、未扩展固定预算。5份TRAIN缓存各共享读取一次，stress/held-out读取0。完整端点预测和恢复checkpoint留本地。旧阶段成本保留独立，不把新训练伪装为旧阶段免费更新；其他WSL项目时间隔离未知。", '',
        '## 最新全部方法汇总','', '以下是最新完成配置的索引。各行清楚标明是否本轮重做；若范围未覆盖全部8种方法，不是相同学习率配方排名。', '',
        '|方法|单样本U%|五样本U均值%|五样本最差U%|当前配置来源|','|---|---:|---:|---:|---|']
    for method in NAMES:
        one=next(r for r in latest if r['method']==method and r['group']=='single');five=next(r for r in latest if r['method']==method and r['group']=='five')
        lines.append(f"|{NAMES[method]}|{one['U_percent']:.4f}|{five['U_percent']:.4f}|{five['worst_U_percent']:.4f}|{'本轮warmup/cosine+clip重做' if method in c['methods'] else '保留原fixed .001'}|")
    lines+=['','## 本轮完整重做结果','','|组|方法|注册参数|活跃参数|更新|U均值%|最差U%|全样本U<1%|UTC秒|','|---|---|---:|---:|---:|---:|---:|---|---:|']
    for r in rows:lines.append(f"|{r['group']}|{NAMES[r['method']]}|{r['parameters']}|{r['active_parameters']}|1000|{r['U_percent']:.4f}|{r['worst_U_percent']:.4f}|{'是' if r['all_cases_U_below_1_percent'] else '否'}|{r['UTC_seconds']:.2f}|")
    lines+=['','![新旧训练配方比较](comparison.png)','', '[最新全部方法CSV](latest_all_methods.csv) · [最新48行逐样本误差](latest_per_case.csv) · [本轮汇总](summary.csv) · [本轮18行误差](per_case.csv) · [配置和成本](metrics.json)', '', '[原统一fixed .001对照](../matched_capacity_u_overfitting/README.md) · [原版单独fixed .0001诊断](../transolver_fit_diagnostics/README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps({'config':c,'terminal_seconds':t['elapsed_seconds'],'summary':rows,'per_case':cases,'latest_all_methods':latest,'latest_per_case':latest_cases,'source_reads':5,'stress_label_reads':0,'heldout_reads':0,'scientific_array_reads_by_exporter':0},indent=2)+'\n')
    print(json.dumps({'rerun_arms':len(rows),'per_case_rows':len(cases),'latest_all_methods_rows':len(latest),'scientific_array_reads':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(a.run,a.config)
