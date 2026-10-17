"""Scalar-only MeshGraphNet single/five report; retain all30 earlier endpoints."""
import argparse,json,math
from pathlib import Path
from tools.export_fifty_u_overfitting import read_csv
from tools.export_onecycle_u_overfitting import write_csv
from tools.export_transolver_hybrid_ablations import LABELS as PREVIOUS_LABELS
LABELS={**PREVIOUS_LABELS,'meshgraphnet':'MeshGraphNet'}
GROUPS=('single','five')

def validate_scope_stop(s,t,scope,run,config):
 if s['active'] or not s['closed'] or t['active'] or not t['closed'] or t['timeout'] or not t['original_deadline_respected'] or max(t['elapsed_seconds'].values())>config['whole_seconds']:raise ValueError('Closed run within original time deadlines required')
 selected={g+'_meshgraphnet' for g in GROUPS}
 if set(s['arms'])!=selected:raise ValueError('Only requested single/five arms may have initialized')
 if s['status']=='completed' and t['exit_code']==0:return False
 gate=run/'fifty_meshgraphnet/USER_SCOPE_GATE.json'
 if s.get('error_type')!='FileExistsError' or t['exit_code']!=1 or 'fifty_meshgraphnet' not in s.get('error','') or not gate.is_file() or set(scope['selected_arms'])!=selected:raise ValueError('Only documented user scope stop is acceptable; technical failures remain failures')
 return True

def assemble(s,c):
 if c['methods']!=['meshgraphnet'] or c['max_updates_per_arm']!=5000 or c['learning_rate']!=.001 or c.get('early_stopping') or c.get('continuation') or not c.get('fixed_endpoint'):raise ValueError('Fresh complete5000 MeshGraphNet recipe required')
 rows=[];cases=[]
 for g in GROUPS:
  name=g+'_meshgraphnet';a=s['arms'][name];m=a.get('endpoint',{});keys=c[g]
  if a['status']!='completed' or a['updates']!=5000 or m.get('step')!=5000 or a.get('stop_reason')!='fixed_update_endpoint' or a.get('partial_backward') or a.get('optimizer_step_in_progress') or not a.get('actual_full_node_backward') or a.get('output_channels')!=3:raise ValueError('Complete actual5000 full-node U endpoint required')
  if a['case_updates']!={k:5000//len(keys) for k in keys} or a['endpoint_predictions_saved']!=len(keys) or set(m['per_case'])!=set(keys):raise ValueError('Every original fitted case needs full visits/predictions')
  if any(abs(a[k]/c['parameter_target']-1)>c['parameter_tolerance'] for k in ('parameters','active_parameters')):raise ValueError('Actual registered/active parameter mismatch')
  mc=a['model_config']
  if mc['message_passing_steps']!=15 or mc['latent_size']!=180 or mc['hidden_layers']!=2 or mc['teacher_or_previous_displacement'] or mc['sample_id'] or mc['node_sampling']:raise ValueError('Declared static MeshGraphNet architecture required')
  if a['maximum_learning_rate']!=.001 or not math.isclose(a['learning_rate_last_update'],4e-8,rel_tol=1e-10):raise ValueError('Actual OneCycle endpoint mismatch')
  values=[m['per_case'][k]['U_percent'] for k in keys]
  if not all(math.isfinite(v) and v>=0 for v in values) or not math.isclose(sum(values)/len(values),m['mean_U_percent'],rel_tol=1e-10) or max(values)!=m['worst_field_percent']:raise ValueError('Scalar aggregation mismatch')
  rows.append(dict(experiment=name,method='meshgraphnet',group=g,parameters=a['parameters'],active_parameters=a['active_parameters'],updates=5000,maximum_learning_rate=.001,learning_rate_last_update=a['learning_rate_last_update'],U_percent=m['mean_U_percent'],worst_U_percent=max(values),all_cases_U_below_1_percent=max(values)<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],source_run='meshgraphnet_u_overfitting',fresh_rerun_of_early_stop=False,precision='FP32 storage / TF32 allowed / static MeshGraphNet'))
  cases.extend(dict(experiment=name,sample=k,updates=5000,U_percent=m['per_case'][k]['U_percent'],source_run='meshgraphnet_u_overfitting') for k in keys)
 return rows,cases

def plot(rows,curves,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 low=max(min(float(r['U_percent']) for r in rows)*.3,1e-7);high=max(float(r['worst_U_percent']) for r in rows)*3
 fig,axes=plt.subplots(1,2,figsize=(17,7),layout='constrained',sharey=True)
 for ax,g in zip(axes,GROUPS):
  selected=[next(r for r in rows if r['method']==m and r['group']==g) for m in LABELS];x=np.arange(len(selected));bars=ax.bar(x,[float(r['U_percent']) for r in selected],color=['#287da8']*10+['#df8524'])
  ax.bar_label(bars,labels=[f"{float(r['U_percent']):.3g}" for r in selected],padding=4,fontsize=8);ax.scatter(x,[float(r['worst_U_percent']) for r in selected],s=70,marker='_',color='#172638',label='Worst U');ax.axhline(1,ls='--',color='#9b4b33',label='1%')
  ax.set_yscale('log');ax.set_ylim(low,high);ax.set_title(g+' / actual update5000');ax.set_xticks(x,list(LABELS.values()),rotation=45,ha='right');ax.set_ylabel('Fitted-case U-vector relative L2 (%)');ax.legend(frameon=False,loc='lower right',fontsize=8)
 fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(17,7),layout='constrained',sharey=True)
 for ax,g in zip(axes,GROUPS):
  for m in ('meshgraphnet','fno','transolver','hashgrid','tcnn_hash'):
   pts=sorted([r for r in curves if r['experiment']==g+'_'+m],key=lambda r:int(r['step']));ax.plot([int(r['step']) for r in pts],[float(r['U_percent']) for r in pts],label=LABELS[m],lw=1.2)
  ax.axhline(1,ls='--',color='#9b4b33');ax.set_yscale('log');ax.set_ylim(low,max(float(r['U_percent']) for r in curves)*1.15);ax.set_xlim(0,5000);ax.set_title(g,pad=45);ax.set_xlabel('Full original-case optimizer updates');ax.set_ylabel('Mean fitted-case U error (%)');ax.legend(ncol=3,frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01))
 fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)

def run(root,base):
 c=json.loads(Path('configs/meshgraphnet_u_overfitting.json').read_text());effective=json.loads(Path('configs/meshgraphnet_single_five_u_overfitting.json').read_text());s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text());scope=json.loads((root/'fifty_meshgraphnet/USER_SCOPE_GATE.json').read_text());controlled=validate_scope_stop(s,t,scope,root,c)
 for key in ('single','five','seed','max_updates_per_arm','learning_rate','minimum_learning_rate','lr_schedule','gradient_clip_norm','evaluate_every','model_settings','output_field','allow_tf32'):
  if c[key]!=effective[key]:raise ValueError('Selected arms must retain unchanged training recipe')
 if s['source_reads']!=50 or s['stress_label_reads'] or s['heldout_reads']:raise ValueError('Actual preloaded TRAIN source accounting required')
 graph=json.loads((root/'graph_cache.json').read_text())
 if graph['graph_reads']!=50 or graph['graph_target_reads']:raise ValueError('Topology cache accounting mismatch')
 rows,cases=assemble(s,c);old=read_csv(base/'all_methods.csv');oldcases=read_csv(base/'all_per_case.csv');curves=read_csv(base/'learning_curves.csv')
 for r in rows:
  pts=[json.loads(line) for line in (root/r['experiment']/'metrics.jsonl').read_text().splitlines()]
  if [m['step'] for m in pts]!=list(range(0,5001,50)):raise ValueError('Complete eval50 curve required')
  curves.extend(dict(experiment=r['experiment'],step=m['step'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],learning_rate_last_update=m.get('learning_rate_last_update'),source_run='meshgraphnet_u_overfitting') for m in pts)
 allrows=old+rows;allcases=oldcases+cases
 if (len(allrows),len(allcases),len(curves))!=(32,566,3232):raise ValueError('Complete32/566/3232 scalar index required')
 out=Path(effective['report_directory']);out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise FileExistsError('Already exported; replot from scalars only')
 write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases);write_csv(out/'all_methods.csv',allrows);write_csv(out/'all_per_case.csv',allcases);write_csv(out/'learning_curves.csv',curves);plot(allrows,curves,out)
 lines=['# MeshGraphNet：单/五样本完整5000步','','用户要求先完成单/五样本。新增两项均actual5000/endstep5000，每案例分别5000/1000遍；已有30个端点原样保留，当前32项。50样本MeshGraphNet未训练，以—占位，不用任何中间值代替。','',
 '采用DeepMind官方[Encode-Process-Decode结构](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/core_model.py)的PyTorch静态位移适配：15个独立消息传递模块、宽180、各MLP两隐藏ReLU层；encoder/processor带LayerNorm，decoder不带；先更新边delta、按receiver求和更新node，再分别加残差。参数注册和实际活跃均4,603,863。不是官方TensorFlow运行时、论文默认宽128或动态rollout复现，也没有真实历史位移/速度作为输入。','',
 '节点为全部原始FEM节点，输入normalizedXYZ+25原条件。图由原TETRA10的六条二次边在原中点处分为12段，双向去重，保留所有中间节点；非kNN或抽样点云图。边输入为normalized mesh-coordinate差向量3维及长度1维。MeshGraphNet额外利用真实网格连接，而旧点场模型不使用该连接，属于不同几何表示的比较。无teacher/physicsU/sampleID/额外stress/heldout。','',
 '总预算约460万±1%，freshseed2026，U3signedmm/vectorrelativeL2²；同Adam fused betas(.9,.999)/eps1e-8/wd0、clipL2=1、eval50，OneCycle5000/pct_start.1/peak1e-3/div25/finaldiv1000/cos两阶段/no momentumcycle；实际首步4e-5、第500峰值.001、第5000用4e-8。没有early stop或达到1%停。FP32允许TF32，与TS/FNO一致；TCNN保留其原生FP32禁TF32的标注。节点和边MLP分块/checkpoint只控制显存，测试验证完整图前向与参数梯度，不做独立子图或邻居采样。','',
 '|方法|单样本U%|五样本均值%|五样本最差%|50样本均值%|','|---|---:|---:|---:|---:|']
 for method,label in LABELS.items():
  a,b=[next(r for r in allrows if r['method']==method and r['group']==g) for g in GROUPS];f=next((r for r in allrows if r['method']==method and r['group']=='fifty'),None)
  lines.append(f"|{label}|{float(a['U_percent']):.6f}|{float(b['U_percent']):.4f}|{float(b['worst_U_percent']):.4f}|{float(f['U_percent']):.4f}|" if f else f"|{label}|{float(a['U_percent']):.6f}|{float(b['U_percent']):.4f}|{float(b['worst_U_percent']):.4f}|—|")
 lines+=['','## 实际耗时','','秒；含模型构造、完整图训练、全组评价和存盘，不等同纯GPU训练或单个算子性能。','','|方法|单样本|五样本|50样本|','|---|---:|---:|---:|']
 for method,label in LABELS.items():
  values=[]
  for g in ('single','five','fifty'):
   r=next((r for r in allrows if r['method']==method and r['group']==g),None);values.append(f"{float(r['UTC_seconds']):.2f}" if r else '—')
  lines.append('|'+label+'|'+'|'.join(values)+'|')
 lines+=['',f"本次原生阶段实际成本{t['elapsed_seconds']['UTC']:.2f}s UTC / {t['elapsed_seconds']['monotonic']:.2f}s单调时钟。用户在启动后收窄范围，原进程已共享加载50个TRAIN U案例和50个图缓存；这笔成本如实计入，只有单/五组被训练。图预处理另读50份H5的vertices/cells及已对齐XYZ，不读目标；耗时见configs/meshgraphnet_graphs.json。完整图合成数据预检只测显存/速度，0个目标标签、0个optimizer更新，不作为训练结果。",'',
 '范围控制说明：为不修改live配置、不重置已进行的单样本更新，预先创建第三项目录，阻止50项初始化。两项完整结果落盘后，原进程因这个预设目录FileExistsError退出1；这不是技术失败重播，也不能称原生exit0。报告保留真实terminal，包括由非零退出造成的within_budget=false，并单列原绝对截止时间遵守情况。两个选定训练项均完整5000，模型/Adam/scheduler/四RNG/order/cursor及6份端点预测本地保留，无50样本模型初始化。若任何选定项不足5000或出现其他失败，导出器会拒绝。','',
 '![当前单/五样本比较](comparison.png)','','![原始训练集误差曲线](learning_curves.png)','','[新增2项](summary.csv) · [新增6案例](per_case.csv) · [全部32项及耗时/精度](all_methods.csv) · [原30项](../transolver_hybrid_ablations/README.md)']
 (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(out/'metrics.json').write_text(json.dumps(dict(actual_config=c,effective_config=effective,summary=rows,stage_seconds=t['elapsed_seconds'],native_terminal=t,controlled_user_scope_stop=controlled,original_time_deadlines_respected=t['original_deadline_respected'],source_reads=50,graph_reads=50,stress_label_reads=0,heldout_reads=0,fifty_model_initialized=False,actual_export=1,scientific_array_reads_by_exporter=0,preserved_old_arms=30),indent=2)+'\n')
 print(json.dumps(dict(new_arms=2,all_arms=32,new_per_case=6,all_per_case=566,fifty_not_run=True,controlled_scope_stop=controlled)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('runs/meshgraphnet_u_overfitting'));p.add_argument('--base',type=Path,default=Path('results/transolver_hybrid_ablations'));a=p.parse_args();run(a.run,a.base)
