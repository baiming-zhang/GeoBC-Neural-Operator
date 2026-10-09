"""A compact, evidence-driven report for the autonomous GeoBC experiments."""
from pathlib import Path
import json
import math
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geobc.train import write_json,utc_now

def percent(value):
    if not math.isfinite(float(value)): raise ValueError('Nonfinite metric')
    return f'{100*value:.3f}%'

def main():
    plan=json.loads((ROOT/'notes/geobc_exploration_plan.json').read_text())
    dest=ROOT/'results/geobc_exploration'
    dest.mkdir(parents=True,exist_ok=True)
    summaries={}
    jobs=plan.get('historical_jobs',[])+plan['jobs']
    for job in jobs:
        source=ROOT/job['output']
        path=source/'summary.json'
        if not path.exists(): continue
        summary=json.loads(path.read_text())
        if summary.get('status')!='completed': continue
        summaries[job['name']]=summary
        target=dest/'runs'/job['name']; target.mkdir(parents=True,exist_ok=True)
        for filename in ('summary.json','config.json','metrics_best.json','metrics_latest.json','metrics.jsonl','parent_comparison.json'):
            if (source/filename).exists(): shutil.copyfile(source/filename,target/filename)
    lines=['# GeoBC 探索：当前结论与下一步','',
           '研究对象：原版 DeepJEB；使用逐节点坐标对齐的原始有符号应力。','',
           '## 当前结论','',
           '**Hash 的单案例表示能力很强；当前共享模型的跨几何、跨载荷组合迁移很差。**','']
    geo=summaries.get('hash50_unseen_geometry_pilot')
    load=summaries.get('hash50_unseen_load_combinations')
    parent=json.loads((ROOT/'runs/slides_reproduction/hash50_full1000ep/metrics_best.json').read_text())
    if geo and load:
        lines += [f"- 原共享模型训练集平均应力 L2：{percent(parent['case_mean_relative_l2'])}。",
                  f"- 两个未参与该模型训练的 CAD：平均 {percent(geo['metrics']['case_mean_relative_l2'])}。",
                  f"- 三个已见 CAD 的九个新载荷组合：平均 {percent(load['metrics']['case_mean_relative_l2'])}。",'']
    comparisons=[]
    for cad in ('101_479','107_320','114_202'):
        summary=summaries.get('independent_'+cad+'_tor')
        if summary:
            item=summary['parent_comparison'][cad+'/tor']
            comparisons.append((cad,item['parent_shared50']['relative_l2'],item['independent_fit']['relative_l2']))
    if comparisons:
        lines+=['| 案例 | 原共享模型 | 独立拟合 |','|---|---:|---:|']
        lines += [f'| {cad}/tor | {percent(shared)} | {percent(independent)} |' for cad,shared,independent in comparisons]
        lines+=['','这些对照固定了架构、输入和父模型 scaler；独立模型增加了总容量，不能单独证明某个几何模块必需。','']
    lines+=['## 正在验证','',
            '同一 40-CAD 训练／5-CAD 验证划分，对比零几何条件与 PointNet 整体几何描述。两者 decoder 布局、随机采样、预算相同；scaler 只拟合训练标签。',
            '训练与选模型只使用 ver/hor/tor；dia 及五个官方测试 CAD 保持隔离。','']
    for name,summary in summaries.items():
        if 'validation' in summary.get('best_metrics',{}):
            metrics=summary.get('best_metrics',{})
            if 'validation' in metrics:
                lines.append(f"- {name}：验证平均 L2 {percent(metrics['validation']['case_mean_relative_l2'])}；最佳 step {summary['best_step']}。")
    lines += ['', '## 解释边界','',
              '两个 CAD 的结果是迁移 pilot，存在其他历史实验使用；不是最终盲测。九个组合的四类载荷均已在其他训练 CAD 出现，不能称为全局未见载荷类别。',
              '迁移尾部误差可能被 asinh 逆变换放大；不截断预测、不更改标签，也不把训练误差称为泛化误差。',
              '候选模型组合已有的 [PointNet](https://arxiv.org/abs/1612.00593) 点集汇聚与 [Instant-NGP](https://nvlabs.github.io/instant-ngp/) hash 表示；当前尚无新算法或泛化成功的结论。','',
              f"本页随实验更新；累计完成 {len(summaries)}/{len(jobs)} 项。",'']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(dest/'manifest.json',{'scope':'GeoBC exploration; independent of the completed slides suite',
                                   'completed':len(summaries),'planned':len(jobs),
                                   'phase':'ongoing' if len(summaries)<len(jobs) else 'stage_completed',
                                   'summaries':list(summaries),'raw_data_checkpoints_predictions':'retained locally; not included in Git payload',
                                   'updated_at_utc':utc_now()})
    if geo and load and comparisons:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        values=[100*parent['case_mean_relative_l2'],100*geo['metrics']['case_mean_relative_l2'],100*load['metrics']['case_mean_relative_l2']]
        axes[0].bar(['Train (50)','Unseen CAD (2)','New loads (9)'],values,color=['#507eaa','#d78c59','#d78c59'])
        for i,v in enumerate(values):axes[0].text(i,v+2,f'{v:.1f}%',ha='center')
        axes[0].set_ylim(0,max(values)*1.2);axes[0].set_title('Frozen shared field: migration pilot')
        axes[0].set_ylabel('Case-mean stress relative L2 (%)')
        x=np.arange(len(comparisons))
        axes[1].bar(x-.18,[100*b for _,b,_ in comparisons],.36,label='Shared50')
        axes[1].bar(x+.18,[100*c for _,_,c in comparisons],.36,label='Independent')
        axes[1].set_yscale('log');axes[1].set_xticks(x,[a for a,_,_ in comparisons])
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
        for i,(_,shared,independent) in enumerate(comparisons):
            axes[1].text(i-.18,100*shared*1.05,f'{100*shared:.1f}%',ha='center',fontsize=9)
            axes[1].text(i+.18,100*independent*1.12,f'{100*independent:.3f}%',ha='center',fontsize=9)
        axes[1].set_ylim(.035,65)
        axes[1].set_title('Identical field architecture; separate fitting')
        axes[1].set_ylabel('All-node stress relative L2 (%)')
        fig.savefig(dest/'diagnosis.png',dpi=180)
        plt.close(fig)
    print(json.dumps({'completed':len(summaries),'planned':len(jobs),'report':str(dest/'REPORT.md')}))

if __name__=='__main__':main()
