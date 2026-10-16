"""Export two full5000 TS hybrids once; retain all24 existing scalar endpoints."""
import argparse,json,math
from pathlib import Path
from tools.export_fifty_u_overfitting import read_csv
from tools.export_onecycle_u_overfitting import write_csv
from tools.export_u_only_overfitting import NAMES
LABELS={**NAMES,'transolver_hybrid':'TS + point query','transolver_grid_hybrid':'TS + grid/query'}
LABELS['tcnn_hash']='TCNN FP32'
LABELS['fno']='FNO + query'
GROUPS=('single','five','fifty')

def assemble(summary,config,method,run_name):
 if config['methods']!=[method] or config['groups']!=list(GROUPS) or config.get('early_stopping') or config.get('continuation') or not config.get('fixed_endpoint') or config['max_updates_per_arm']!=5000 or config['learning_rate']!=.001 or config.get('allow_tf32') is not True:raise ValueError('Fresh full5000 common hybrid recipe required')
 if set(summary['arms'])!={g+'_'+method for g in GROUPS}:raise ValueError('All3 hybrid groups required')
 if not summary['precision_runtime']['allow_tf32'] or summary['precision_runtime']['NVIDIA_TF32_OVERRIDE'] not in (None,'1'):raise ValueError('Expected FP32 with TF32 allowed')
 rows=[];cases=[]
 for group in GROUPS:
  name=group+'_'+method;a=summary['arms'][name];m=a.get('endpoint',{});keys=config[group]
  if a['status']!='completed' or a['updates']!=5000 or m.get('step')!=5000 or a.get('stop_reason')!='fixed_update_endpoint' or a.get('partial_backward') or a.get('optimizer_step_in_progress') or not a.get('actual_full_node_backward') or a.get('output_channels')!=3:raise ValueError('Complete5000 full-node U endpoint required')
  if a['case_updates']!={k:5000//len(keys) for k in keys} or a['endpoint_predictions_saved']!=len(keys) or set(m['per_case'])!=set(keys):raise ValueError('Every original case needs full visits/predictions')
  if any(abs(a[k]/config['parameter_target']-1)>config['parameter_tolerance'] for k in ('parameters','active_parameters')):raise ValueError('Actual registered/active capacity mismatch')
  if a['maximum_learning_rate']!=.001 or not math.isclose(a['learning_rate_last_update'],4e-8,rel_tol=1e-10):raise ValueError('Actual OneCycle endpoint mismatch')
  mc=a['model_config']
  if mc['query_head']!=[44,376,376,3] or mc['latent_channels']!=16 or mc['label_input'] or mc['backbone']['method']!='transolver':raise ValueError('Declared TS hybrid architecture mismatch')
  if method=='transolver_grid_hybrid' and (mc['grid_resolution']!=64 or mc['all_grid_cells']!=262144):raise ValueError('Full R64 context required')
  values=[m['per_case'][k]['U_percent'] for k in keys]
  if not all(math.isfinite(v) and v>=0 for v in values) or not math.isclose(sum(values)/len(values),m['mean_U_percent'],rel_tol=1e-10) or max(values)!=m['worst_field_percent']:raise ValueError('Scalar aggregation mismatch')
  rows.append(dict(experiment=name,method=method,group=group,parameters=a['parameters'],active_parameters=a['active_parameters'],updates=5000,maximum_learning_rate=.001,learning_rate_last_update=a['learning_rate_last_update'],U_percent=m['mean_U_percent'],worst_U_percent=max(values),all_cases_U_below_1_percent=max(values)<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],source_run=run_name,fresh_rerun_of_early_stop=False,precision='FP32 storage / TF32 allowed / TS hybrid'))
  cases.extend(dict(experiment=name,sample=k,updates=5000,U_percent=m['per_case'][k]['U_percent'],source_run=run_name) for k in keys)
 return rows,cases

def plot(rows,curves,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 low=max(min(float(r['U_percent']) for r in rows)*.3,1e-7);high=max(float(r['worst_U_percent']) for r in rows)*3
 fig,axes=plt.subplots(1,3,figsize=(21,7),layout='constrained',sharey=True)
 for ax,g in zip(axes,GROUPS):
  selected=[next(r for r in rows if r['method']==m and r['group']==g) for m in LABELS]
  x=np.arange(len(selected));bars=ax.bar(x,[float(r['U_percent']) for r in selected],color=['#287da8']*8+['#de8b28','#4e9b59'],label='Mean U')
  ax.bar_label(bars,labels=[f"{float(r['U_percent']):.3g}" for r in selected],padding=4,fontsize=8)
  ax.scatter(x,[float(r['worst_U_percent']) for r in selected],marker='_',s=80,color='#172638',label='Worst U');ax.axhline(1,ls='--',color='#9b4b33',label='1%')
  ax.set_yscale('log');ax.set_ylim(low,high);ax.set_xticks(x,[LABELS[r['method']] for r in selected],rotation=45,ha='right');ax.set_title(g+' / actual update5000');ax.set_ylabel('Fitted-case U-vector relative L2 (%)');ax.legend(frameon=False,fontsize=8,loc='lower right')
 fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(21,7),layout='constrained',sharey=True)
 for ax,g in zip(axes,GROUPS):
  for m,label in LABELS.items():
   points=sorted([r for r in curves if r['experiment']==g+'_'+m],key=lambda r:int(r['step']));ax.plot([int(r['step']) for r in points],[float(r['U_percent']) for r in points],label=label,lw=1)
  ax.axhline(1,ls='--',color='#9b4b33');ax.set_yscale('log');ax.set_ylim(low,max(float(r['U_percent']) for r in curves)*1.15);ax.set_xlim(0,5000);ax.set_title(g,pad=65);ax.set_xlabel('Full original-node case updates');ax.set_ylabel('Mean fitted-case U error (%)');ax.legend(ncol=3,frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01))
 fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)

def run(point,grid,base):
 rows=[];cases=[];curves=read_csv(base/'learning_curves.csv');costs={};configs={}
 for path,method in ((point,'transolver_hybrid'),(grid,'transolver_grid_hybrid')):
  cfgpath=Path('configs')/(path.name+'.json');c=json.loads(cfgpath.read_text());s=json.loads((path/'summary.json').read_text());t=json.loads((path/'terminal.json').read_text())
  if s['active'] or not s['closed'] or t['active'] or not t['closed'] or t['exit_code']!=0 or not t['within_budget']:raise ValueError('Native successful closure required')
  if s['source_reads']!=50 or s['stress_label_reads'] or s['heldout_reads']:raise ValueError('TRAIN-only source accounting mismatch')
  configs[method]=c;costs[path.name]=t['elapsed_seconds'];new,newcases=assemble(s,c,method,path.name);rows+=new;cases+=newcases
  for r in new:
   points=[json.loads(line) for line in (path/r['experiment']/'metrics.jsonl').read_text().splitlines()]
   if [m['step'] for m in points]!=list(range(0,5001,50)):raise ValueError('Complete eval50 scalar curve required')
   curves.extend(dict(experiment=r['experiment'],step=m['step'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],learning_rate_last_update=m.get('learning_rate_last_update'),source_run=path.name) for m in points)
 a,b=configs.values()
 for key in ('single','five','fifty','seed','learning_rate','lr_schedule','gradient_clip_norm','evaluate_every','max_updates_per_arm','allow_tf32'):
  if a[key]!=b[key]:raise ValueError('Same common training recipe required')
 old=read_csv(base/'all_methods.csv');oldcases=read_csv(base/'all_per_case.csv');allrows=old+rows;allcases=oldcases+cases
 if (len(allrows),len(allcases),len(curves))!=(30,560,3030):raise ValueError('Complete30/560/3030 comparison required')
 out=Path('results/transolver_hybrid_ablations');out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise FileExistsError('Already exported; replot from exported scalars only')
 write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases);write_csv(out/'all_methods.csv',allrows);write_csv(out/'all_per_case.csv',allcases);write_csv(out/'learning_curves.csv',curves);plot(allrows,curves,out)
 total={k:sum(v[k] for v in costs.values()) for k in ('UTC','monotonic')}
 lines=['# Transolver查询头与网格化消融：1/5/50样本','','保留已有24项真实5000步端点，新增两种TS混合实现共6项，各freshseed2026且实际5000步。仅TRAIN U3拟合，非泛化测试，不挑best/partial。','',
 'TS + point query：官方原版非规则点骨干在全部原始点上产生16维隐特征，与原点XYZ+25条件拼接，44→376→376→3 GELU查询头。TS + grid/query：先用与FNO完全相同的64³占用+25维格点均值条件，再用官方原版非规则点TS处理全部262144格点，隐特征三线性插值回原点，与原XYZ+25条件进入同一个查询头。网格TS额外显式输入确定性的规则网格XYZ，为注意力提供位置；这来自网格索引，不含额外标签。FNO由FFT/网格结构表达空间关系。','',
 '两种TS骨干均5层/宽400/8heads/32slices/MLPratio1，latent16；查询头宽376。原版容量匹配TS宽408、直接输出U3；现有FNO查询头宽320。新增头计入约460万±1%注册/活跃参数预算，因此这是同预算的混合架构比较，不是保持每一层参数完全不变的纯加头实验，也不是论文默认配置原样复现。','',
 '统一Adam fused betas(.9,.999), eps1e-8, wd0；OneCycle5000/pct_start.1/peak1e-3/div25/finaldiv1000/cos两阶段/no momentum cycle；实际第1步4e-5、第500峰值.001、第5000步4e-8；clipL2=1，eval50，无早停。每update一完整原节点case，single5000遍、five各1000遍、fifty各100遍。U3signedmm/vectorrelativeL2²，组RMS仅拟合组；无sampleID/teacher/physicsU/stress/heldout输入。','',
 '两新增模型FP32且允许TF32，与原TS/FNO一致；表中TCNN为原生FP32 Cutlass且禁TF32。TS网格损失仍在原节点，不将真实U栅格化输入。两阶段各共享读取50个TRAIN U源一次，总100次；只为网格阶段构建所需格点缓存。所有模型/Adam/scheduler/四RNG/order/cursor及端点预测本地保留。','',
 '|方法|单样本U%|五样本均值%|五样本最差%|50样本均值%|50样本最差%|','|---|---:|---:|---:|---:|---:|']
 for method,label in LABELS.items():
  p,q,r=[next(r for r in allrows if r['method']==method and r['group']==g) for g in GROUPS];lines.append(f"|{label}|{float(p['U_percent']):.6f}|{float(q['U_percent']):.4f}|{float(q['worst_U_percent']):.4f}|{float(r['U_percent']):.4f}|{float(r['worst_U_percent']):.4f}|")
 lines+=['','## 实际执行时间','','时间单位秒，含模型构造、训练、全组评价和保存；不作为纯内核速度。','','|方法|单样本|五样本|50样本|','|---|---:|---:|---:|']
 for method,label in LABELS.items():lines.append('|'+label+'|'+'|'.join(f"{float(next(r for r in allrows if r['method']==method and r['group']==g)['UTC_seconds']):.2f}" for g in GROUPS)+'|')
 lines+=['',f"新增两阶段合计{total['UTC']:.2f}s UTC / {total['monotonic']:.2f}s单调时钟，含共享加载/导入/退出；事件队列等候与第一阶段重叠，不重复累计。旧24项阶段成本保留于原报告。导出只读标量，模型/GT/科学数组读取0。",'', '![全部30端点](comparison.png)','','![原始学习曲线](learning_curves.png)','','[全部30项和时间](all_methods.csv) · [560个逐案例端点](all_per_case.csv) · [新6项](summary.csv) · [原24项](../tcnn_fp32_u_overfitting/README.md)']
 (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(out/'metrics.json').write_text(json.dumps(dict(configs=configs,summary=rows,stage_costs=costs,stage_seconds=total,source_reads=100,stress_label_reads=0,heldout_reads=0,actual_export=1,scientific_array_reads_by_exporter=0,preserved_old_arms=24),indent=2)+'\n')
 print(json.dumps(dict(new_arms=6,all_arms=30,new_per_case=112,all_per_case=560,all_actual_updates=5000)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--point',type=Path,default=Path('runs/transolver_hybrid_u_overfitting'));p.add_argument('--grid',type=Path,default=Path('runs/transolver_grid_hybrid_u_overfitting'));p.add_argument('--base',type=Path,default=Path('results/tcnn_fp32_u_overfitting'));a=p.parse_args();run(a.point,a.grid,a.base)
