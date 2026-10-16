"""Replace only three TCNN endpoints using scalar files; preserve FP16 history."""
import argparse,json,math
from pathlib import Path
from tools.export_fifty_u_overfitting import read_csv,plot
from tools.export_onecycle_u_overfitting import write_csv
from tools.export_u_only_overfitting import NAMES

def assemble(s,c):
 if c.get('allow_tf32') is not False or c['model_settings']['tcnn_hash'].get('native_precision')!='fp32' or c.get('early_stopping') or not c.get('fixed_endpoint') or c['max_updates_per_arm']!=5000:raise ValueError('Native FP32, no TF32, full5000 required')
 if set(s['arms'])!={g+'_tcnn_hash' for g in ('single','five','fifty')}:raise ValueError('Exactly three TCNN arms required')
 if s['precision_runtime']!={'allow_tf32':False,'NVIDIA_TF32_OVERRIDE':'0'}:raise ValueError('Runtime precision guard required')
 rows=[];cases=[]
 for g in ('single','five','fifty'):
  name=g+'_tcnn_hash';a=s['arms'][name];m=a.get('endpoint',{});keys=c[g]
  if a['status']!='completed' or a['updates']!=5000 or m.get('step')!=5000 or a.get('stop_reason')!='fixed_update_endpoint' or a.get('partial_backward') or a.get('optimizer_step_in_progress') or not a.get('actual_full_node_backward'):raise ValueError('Complete5000 full-node endpoint required')
  if a['case_updates']!={k:5000//len(keys) for k in keys} or a['endpoint_predictions_saved']!=len(keys) or set(m['per_case'])!=set(keys):raise ValueError('Every fitted case must have complete visits/predictions')
  mc=a['model_config']
  if mc.get('actual_native_dtype')!='torch.float32' or mc['actual_native_hyperparams']['network']['otype']!='CutlassMLP':raise ValueError('Actual native FP32 CutlassMLP required')
  if any(abs(a[k]/c['parameter_target']-1)>c['parameter_tolerance'] for k in ('parameters','active_parameters')):raise ValueError('Capacity mismatch')
  if a['maximum_learning_rate']!=.001 or not math.isclose(a['learning_rate_last_update'],4e-8,rel_tol=1e-10):raise ValueError('Actual LR mismatch')
  v=[m['per_case'][k]['U_percent'] for k in keys]
  if not all(math.isfinite(x) and x>=0 for x in v) or not math.isclose(sum(v)/len(v),m['mean_U_percent'],rel_tol=1e-10) or max(v)!=m['worst_field_percent']:raise ValueError('Scalar aggregation mismatch')
  rows.append(dict(experiment=name,method='tcnn_hash',group=g,parameters=a['parameters'],active_parameters=a['active_parameters'],updates=5000,maximum_learning_rate=.001,learning_rate_last_update=a['learning_rate_last_update'],U_percent=m['mean_U_percent'],worst_U_percent=max(v),all_cases_U_below_1_percent=max(v)<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],source_run='tcnn_fp32_u_overfitting',fresh_rerun_of_early_stop=False))
  cases.extend(dict(experiment=name,sample=k,updates=5000,U_percent=m['per_case'][k]['U_percent'],source_run='tcnn_fp32_u_overfitting') for k in keys)
 return rows,cases

def run(root,config,base):
 c=json.loads(config.read_text());s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text())
 if s['active'] or not s['closed'] or not t['within_budget'] or t['exit_code']!=0 or t['active'] or not t['closed']:raise ValueError('Closed successful native run required')
 if s['source_reads']!=50 or s['stress_label_reads'] or s['heldout_reads']:raise ValueError('TRAIN U-only source accounting required')
 rows,cases=assemble(s,c);old=read_csv(base/'all_methods.csv');oldcases=read_csv(base/'all_per_case.csv');curves=[r for r in read_csv(base/'learning_curves.csv') if not r['experiment'].endswith('_tcnn_hash')]
 oldtcnn=[r for r in old if r['method']=='tcnn_hash']
 allrows=[r for r in old if r['method']!='tcnn_hash']+rows;allcases=[r for r in oldcases if not r['experiment'].endswith('_tcnn_hash')]+cases
 for r in allrows:r['precision']='native FP32 / TF32 disabled / CutlassMLP' if r['method']=='tcnn_hash' else 'FP32 storage / TF32 allowed (historical endpoint)'
 for r in rows:
  points=[json.loads(line) for line in (root/r['experiment']/'metrics.jsonl').read_text().splitlines()]
  if [p['step'] for p in points]!=list(range(0,5001,50)):raise ValueError('Complete scalar curve required')
  curves.extend(dict(experiment=r['experiment'],step=p['step'],U_percent=p['mean_U_percent'],worst_U_percent=p['worst_field_percent'],learning_rate_last_update=p.get('learning_rate_last_update'),source_run='tcnn_fp32_u_overfitting') for p in points)
 if (len(allrows),len(allcases),len(curves))!=(24,448,2424):raise ValueError('Complete24/448/2424 scalar comparison required')
 out=Path(c['report_directory']);out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise FileExistsError('Already exported; replot from scalars only')
 write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases);write_csv(out/'all_methods.csv',allrows);write_csv(out/'all_per_case.csv',allcases);write_csv(out/'learning_curves.csv',curves)
 write_csv(out/'historical_tcnn_fp16.csv',oldtcnn)
 old_label=NAMES['tcnn_hash'];NAMES['tcnn_hash']=old_label+' FP32'
 try:plot(allrows,curves,out)
 finally:NAMES['tcnn_hash']=old_label
 lines=['# TCNN原生FP32：1/5/50样本完整5000步','','用户要求TCNN改为FP32后，这三项均从seed2026重新初始化、完整训练5000次原节点更新，无早停。其余21项保留已交付的真实5000步标量。旧TCNN FP16结果保留在[历史报告](../fifty_u_overfitting/README.md)及[历史标量](historical_tcnn_fp16.csv)。','',
 'TCNN本地源码以TCNN_HALF_PRECISION=0构建独立扩展，运行时核验native dtype=torch.float32及实际CutlassMLP后端。官方FullyFusedMLP路径采用FP16；FP32使用官方CutlassMLP，因此这不是单独改变输出cast。哈希编码/层宽深、约460万实际注册与活跃参数均保留。禁用PyTorch matmul/cuDNN TF32，并设置NVIDIA_TF32_OVERRIDE=0；无AMP。其他21项的历史FP32计算允许TF32，因此整张表不能称为所有方法完全相同数值精度的比较。','',
 'Adam fused betas(.9,.999)、eps1e-8、wd0；OneCycle5000/pct_start.1/peak1e-3/div25/finaldiv1000/cos两阶段/no momentum cycle；实际首步4e-5、500步峰值1e-3、5000步4e-8。clipL2=1，eval50。signed U3向量relativeL2平方损失、normalizedXYZ+25条件；无sampleID/teacher/physicsU/heldout/stress。每次更新覆盖一个完整原节点案例，单/五/50样本每case分别5000/1000/100遍。仅拟合TRAIN，并非泛化或架构上限证明。','',
 '|方法|单样本U%|五样本均值%|五样本最差%|50样本均值%|50样本最差%|','|---|---:|---:|---:|---:|---:|']
 for method,label in NAMES.items():
  a,b,d=[next(r for r in allrows if r['method']==method and r['group']==g) for g in ('single','five','fifty')]
  lines.append(f"|{label}{' FP32' if method=='tcnn_hash' else ''}|{float(a['U_percent']):.4f}|{float(b['U_percent']):.4f}|{float(b['worst_U_percent']):.4f}|{float(d['U_percent']):.4f}|{float(d['worst_U_percent']):.4f}|")
 lines+=['','## TCNN精度修正与实际耗时','','时间包括模型构造、训练、评价和保存；阶段总成本还含共享加载、导入和退出。旧FP16与新FP32的后端也不同，不能把时间比值解释为单纯浮点位宽的影响。','','|组|旧FP16均值U%|新FP32均值U%|旧FP16耗时秒|新FP32耗时秒|','|---|---:|---:|---:|---:|']
 for r in rows:
  p=next(p for p in oldtcnn if p['group']==r['group']);lines.append(f"|{r['group']}|{float(p['U_percent']):.6f}|{r['U_percent']:.6f}|{float(p['UTC_seconds']):.2f}|{r['UTC_seconds']:.2f}|")
 lines += ['',f"本次新增阶段成本{t['elapsed_seconds']['UTC']:.2f}s UTC / {t['elapsed_seconds']['monotonic']:.2f}s单调时钟；50个TRAIN案例各共享读取一次，stress/heldout读取0。构建成本另计，见构建说明。旧实验成本保留于原报告，不重复计算。导出仅访问标量，科学数组/模型/GT读取0。",'', '![全部24项比较](comparison.png)','','![原始学习曲线](learning_curves.png)','','[全部方法及时间](all_methods.csv) · [448个逐案例端点](all_per_case.csv) · [TCNN新增3项](summary.csv)']
 (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (out/'metrics.json').write_text(json.dumps(dict(config=c,summary=rows,stage_seconds=t['elapsed_seconds'],source_reads=50,stress_label_reads=0,heldout_reads=0,actual_export=1,scientific_array_reads_by_exporter=0,preserved_old_arms=21),indent=2)+'\n')
 print(json.dumps(dict(new_arms=3,all_arms=24,all_per_case=448,all_actual_updates=5000)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/tcnn_fp32_u_overfitting.json'));p.add_argument('--base',type=Path,default=Path('results/fifty_u_overfitting'));a=p.parse_args();run(a.run,a.config,a.base)
