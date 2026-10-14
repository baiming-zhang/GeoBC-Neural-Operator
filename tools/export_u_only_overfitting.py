"""Export all strict parameter-matched U-only step1000 endpoints, scalar only."""
import argparse,csv,json
from pathlib import Path

NAMES={'mlp':'MLP','ngp4':'NGP4','hashgrid':'TorchHash','tcnn_hash':'TCNN','fno':'FNO','transolver':'Transolver','transolverpp':'Transolver++','ts3':'TS3'}

def write_plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),constrained_layout=True)
    values=[float(r[k]) for r in rows for k in ('U_percent','worst_U_percent')]
    lower=max(min(min(values)*.4,.2),1e-5);upper=max(max(values)*2,2)
    for ax,group in zip(axes,['single','five']):
        data=[r for r in rows if r['group']==group];x=np.arange(len(data))
        bars=ax.bar(x,[float(r['U_percent']) for r in data],width=.55,color='#247fa9',label='Mean U')
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=8)
        ax.scatter(x+.36,[float(r['worst_U_percent']) for r in data],marker='_',s=100,color='#172638',label='Worst case U')
        ax.axhline(1,color='#172638',linestyle='--',label='1% goal');ax.set_yscale('log')
        ax.set_ylim(lower,upper)
        ax.set_xticks(x,[NAMES[r['method']] for r in data],rotation=35,ha='right');ax.set_ylabel('Full original-node U-vector error (%)')
        ax.set_title(('Single sample' if group=='single' else 'Five samples')+' / U-only / step 1000');ax.legend(frameon=False,fontsize=8)
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

def run(root):
    s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text())
    c=json.loads(Path('configs/matched_capacity_u_overfitting.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or not t['within_budget'] or t['active'] or not t['closed']:raise ValueError('Successful closed terminal required')
    if s.get('output_field')!='u' or s.get('stress_label_reads')!=0 or s['heldout_reads']!=0 or s['source_reads']!=5:raise ValueError('U-only source contract violated')
    expected={group+'_'+method for group in ('single','five') for method in c['methods']}
    if set(s['arms'])!=expected:raise ValueError('All 16 configured arms required')
    rows=[];cases=[]
    for name,a in s['arms'].items():
        m=a.get('endpoint')
        if a['status']!='completed' or a['updates']!=1000 or m is None or m['step']!=1000 or a.get('output_channels')!=3 or not a.get('actual_full_node_backward'):raise ValueError('All actual U-only1000 endpoints required')
        if abs(a['parameters']/c['parameter_target']-1)>c['parameter_tolerance']:raise ValueError('Capacity mismatch')
        if set(m['per_case'])!=set(c[a['group']]) or any(set(v)!={'U_percent'} for v in m['per_case'].values()) or a.get('endpoint_predictions_saved')!=len(c[a['group']]):raise ValueError('Complete U-only endpoint predictions required')
        rows.append(dict(experiment=name,method=a['method'],group=a['group'],parameters=a['parameters'],updates=1000,U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],all_cases_U_below_1_percent=m['worst_field_percent']<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic']))
        cases.extend(dict(experiment=name,sample=k,U_percent=v['U_percent']) for k,v in m['per_case'].items())
    if len(cases)!=48:raise ValueError('48 case rows required')
    out=Path('results/matched_capacity_u_overfitting');out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported')
    for filename,data in [('summary.csv',rows),('per_case.csv',cases)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write_plot(rows,out)
    lines=['# 只学习位移U：参数匹配、恰好1000步的完整对照','',
        '8种方法各约460万可训练参数（±1%），单/五样本共16项全部fresh seed2026，各恰好1000次优化更新。输出为signed Ux/Uy/Uz（mm），损失只有完整U向量相对L2平方，不读取应力标签，不从U/S联合模型抽取U代替U-only训练。Adam固定.001，无早停或plateau降学习率；主表严格是1000步端点。',
        '', '单样本506_533/ver；五样本另含506_25/tor、10_30/ver、10_421/hor、10_460/dia。一次更新完整一个原节点CAD×载荷，五样本各200遍。归一化XYZ+原25维条件输入；无样本ID、物理U直通、教师或held-out。输出尺度仅用拟合样本U向量RMS；每样本完整节点误差FP64读出，表中均值为案例误差算术均值，验收每样本U均<1%。',
        '', '模型容量设置沿用[联合场参数匹配配置](../../configs/matched_capacity_overfitting.json)的宽度、深度与编码容量，但末端输出改为3通道；实际参数逐项构造检查。TCNN保留原生对齐，FP16计算/FP32 master；其他模型FP32/TF32。FNO仍R64潜表示+原节点查询头，Transformer保留全节点全局上下文；TS++保留官方eval随机Gumbel，单次端点不表示多次随机读出均通过。一个种子与固定配方仅是训练拟合诊断，不代表泛化或普遍架构排名。',
        '',f"完整训练阶段成本{t['elapsed_seconds']['UTC']:.2f}秒UTC / {t['elapsed_seconds']['monotonic']:.2f}秒单调时钟，含导入、缓存、初始化、训练、完整读出、保存和退出；终止码0且预算内。五份本地TRAIN缓存各共享读取一次，应力标签/held-out读取0；导出只读scalar，不加载模型、GT或科学数组。其他WSL项目的时间隔离未知，耗时不是严格硬件吞吐排名。",
        '', '|组|方法|实际参数|更新|U均值%|最差U%|每样本U<1%|UTC秒|单调秒|','|---|---|---:|---:|---:|---:|---|---:|---:|']
    lines += [f"|{r['group']}|{NAMES[r['method']]}|{r['parameters']}|1000|{r['U_percent']:.4f}|{r['worst_U_percent']:.4f}|{'是' if r['all_cases_U_below_1_percent'] else '否'}|{r['UTC_seconds']:.2f}|{r['monotonic_seconds']:.2f}|" for r in rows]
    lines+=['','![U-only完整对照](comparison.png)','', '[16项汇总](summary.csv) · [48行逐案例误差](per_case.csv) · [配置与完整成本](metrics.json) · [实际参数记录](../sample_overfitting/u_only_parameter_profiles.json)', '', '完整端点节点预测及last.pt留在本地运行目录，不上传科学数组或模型。[此前U/S联合对照](../matched_capacity_overfitting/README.md)独立保留，不能与本轮混成同一目标训练。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps({'config':c,'terminal_seconds':t['elapsed_seconds'],'source_reads':s['source_reads'],'stress_label_reads':0,'heldout_reads':0,'summary':rows,'per_case':cases,'scientific_array_reads_by_exporter':0},indent=2)+'\n')
    print(json.dumps({'arms':16,'case_rows':48,'stress_reads':0,'scientific_array_reads':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
