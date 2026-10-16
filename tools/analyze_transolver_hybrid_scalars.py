"""Interpret the completed ablation from exported scalars only."""
import csv,json,math,statistics
from pathlib import Path

def analyze(out):
 def read(name):
  with (out/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
 def write(name,rows):
  with (out/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rows=read('all_methods.csv');cases=read('all_per_case.csv');groups=('single','five','fifty');methods=('fno','transolver','transolver_hybrid','transolver_grid_hybrid');labels={'fno':'FNO + query','transolver':'原版TS','transolver_hybrid':'TS + point query','transolver_grid_hybrid':'TS + grid/query'}
 indexed={(r['method'],r['group']):r for r in rows};summary=[];deltas=[];anchors=[]
 cfg=json.loads((out/'metrics.json').read_text())['configs']['transolver_hybrid'];keys=cfg['five']
 for method in methods:
  for group in groups:
   r=indexed[(method,group)];baseline=indexed[('transolver',group)]
   summary.append(dict(method=method,group=group,mean_U_percent=float(r['U_percent']),worst_U_percent=float(r['worst_U_percent']),UTC_seconds=float(r['UTC_seconds']),parameters=int(r['parameters']),mean_relative_change_vs_original_TS_percent=100*(float(r['U_percent'])/float(baseline['U_percent'])-1)))
  vals=[float(r['U_percent']) for r in cases if r['experiment']=='fifty_'+method and r['sample'] in keys]
  anchors.append(dict(method=method,original_five_group_mean_U_percent=float(indexed[(method,'five')]['U_percent']),fifty_model_same_five_mean_U_percent=statistics.mean(vals),fifty_model_same_five_worst_U_percent=max(vals)))
 for method in methods[2:]:
  for group in groups:
   original={r['sample']:float(r['U_percent']) for r in cases if r['experiment']==group+'_transolver'}
   for r in cases:
    if r['experiment']==group+'_'+method:deltas.append(dict(method=method,group=group,sample=r['sample'],original_TS_U_percent=original[r['sample']],hybrid_U_percent=float(r['U_percent']),delta_percentage_points=float(r['U_percent'])-original[r['sample']]))
 write('ablation_comparison.csv',summary);write('per_case_delta_vs_TS.csv',deltas);write('same_five_cases.csv',anchors)
 lines=['## 消融结果解释','','在同一5000更新配方和约460万参数预算下，新增查询头没有带来一致收益：','',
 '|实现|单样本相对原TS均值误差变化|五样本变化|50样本变化|','|---|---:|---:|---:|']
 for method in methods[2:]:lines.append('|'+labels[method]+'|'+'|'.join(f"{next(r for r in summary if r['method']==method and r['group']==g)['mean_relative_change_vs_original_TS_percent']:+.2f}%" for g in groups)+'|')
 lines+=['','上表是相对误差的百分比变化；负值表示改善，不是误差百分点。离散点查询头只在五样本均值上改善；其单样本和50样本均值均变差。网格TS三组均值均高于原版TS，50样本最差误差也从61.4277%升至79.5382%。网格模型在单样本上优于离散点混合模型，但不优于原TS。','',
 'FNO混合实现的五样本与50样本均值仍分别为1.6939%、9.9989%，低于两种TS混合实现。因而仅加相同形式的原点查询头、或再对齐网格化/插值，并未在本轮复制FNO的优势。该结论限定于当前实现、单seed和配方；没有证明FNO普遍优于TS，也没有确定剩余差异究竟来自算子归纳偏置、优化轨迹或其他架构细节。','',
 '两个新增TS模型骨干宽400、头宽376，旧TS宽408、FNO头宽320。总容量匹配引入了宽度调整；网格TS还需确定性的网格坐标作为注意力位置输入。这些差异使其不是严格保持所有其他层不变的单因素因果消融。精度、输入表示、每案例曝光次数及评价/保存开销也必须随表一起解释。','',
 '|新增实现／组|较原TS误差更低的案例数|案例总数|','|---|---:|---:|']
 for method in methods[2:]:
  for group in groups:
   subset=[r for r in deltas if r['method']==method and r['group']==group];lines.append(f"|{labels[method]} / {group}|{sum(r['delta_percentage_points']<0 for r in subset)}|{len(subset)}|")
 lines+=['','[12项核心消融标量](ablation_comparison.csv) · [112个逐案例变化](per_case_delta_vs_TS.csv) · [50样本模型在相同五案例上的比较](same_five_cases.csv)','']
 report=out/'README.md';text=report.read_text(encoding='utf-8');marker='## 消融结果解释';text=text.split(marker)[0].rstrip()+'\n\n'+'\n'.join(lines);report.write_text(text,encoding='utf-8')
 return dict(summary=summary,anchors=anchors,new_cases_below1=sum(float(r['U_percent'])<1 for r in cases if 'hybrid' in r['experiment']))

if __name__=='__main__':print(json.dumps(analyze(Path('results/transolver_hybrid_ablations')),ensure_ascii=False))
