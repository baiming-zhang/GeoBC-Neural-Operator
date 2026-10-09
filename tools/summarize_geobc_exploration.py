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
    checkpoints={}
    jobs=plan.get('historical_jobs',[])+plan['jobs']
    for job in jobs:
        source=ROOT/job['output']
        path=source/'summary.json'
        if not path.exists(): continue
        summary=json.loads(path.read_text())
        if summary.get('status')!='completed': continue
        summaries[job['name']]=summary
        checkpoint_files={}
        for filename in ('best.pt','last.pt'):
            checkpoint=source/filename
            if checkpoint.is_file():
                stat=checkpoint.stat()
                checkpoint_files[filename]={'local_path':str(checkpoint),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if checkpoint_files:
            checkpoints[job['name']]={'best_step':summary.get('best_step'),'files':checkpoint_files,
                                      'code_sha256':summary.get('config',{}).get('contract',{}).get('code_sha256',{}),
                                      'selection':summary.get('config',{}).get('checkpoint_selection')}
        target=dest/'runs'/job['name']; target.mkdir(parents=True,exist_ok=True)
        for filename in ('summary.json','config.json','metrics_best.json','metrics_latest.json','metrics.jsonl','parent_comparison.json'):
            if (source/filename).exists(): shutil.copyfile(source/filename,target/filename)
    write_json(dest/'checkpoint_index.json',{'weights_published':False,'weights_location':'local training directories',
                                            'identity_policy':'paths/bytes/mtime plus small config source hashes; no redundant checkpoint scans',
                                            'runs':checkpoints})
    manifests=dest/'data_manifests'; manifests.mkdir(exist_ok=True)
    for directory in ('prepared_geobc_transfer','prepared_geobc_multiload','geobc_local_geometry','geobc_frozen_u_response'):
        source=ROOT/'data'/directory/'manifest.json'
        if source.is_file():shutil.copyfile(source,manifests/(directory+'.json'))
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
    lines+=['## 几何条件能否改善泛化','',
            '使用同一 40-CAD 训练／5-CAD 验证划分。每组对照的 decoder 布局、随机采样、预算相同；scaler 只拟合训练标签。每项训练10,000次更新，不能与原50案例的1000个完整epoch直接等同。',
            '训练与选模型只使用 ver/hor/tor；五个官方测试 CAD 保持隔离。验证CAD的dia曾作一次模型冻结后的诊断，后续不能再称全新盲测。','']
    paired=[
        ('整体几何／宽度64','hash_bc_no_geometry_10k','hash_bc_pointnet_10k'),
        ('整体几何／宽度128','hash_bc_no_geometry_width128_10k','hash_bc_pointnet_width128_10k'),
        ('局部几何／宽度128，均有PointNet','hash_bc_pointnet_local_zero_width128_10k','hash_bc_pointnet_local_actual_width128_10k'),
        ('局部几何／同配置，种子2027','hash_bc_pointnet_local_zero_width128_seed2027_10k','hash_bc_pointnet_local_actual_width128_seed2027_10k'),
    ]
    available=[(label,summaries.get(zero),summaries.get(actual)) for label,zero,actual in paired]
    available=[row for row in available if row[1] or row[2]]
    if available:
        lines+=['| 同配置对照 | 零几何特征：验证误差 | 实际几何特征：验证误差 |','|---|---:|---:|']
        metric=lambda summary:percent(summary['best_metrics']['validation']['case_mean_relative_l2']) if summary else '待完成'
        lines += [f'| {label} | {metric(zero)} | {metric(actual)} |' for label,zero,actual in available]
        lines += ['', '整体PointNet的收益不足0.04个百分点，加宽解码器后最佳验证误差仍约65.3%；目前不能认为这些改动改善了泛化。',
                  '局部几何对照使用13维离散边界距离、方向和邻域形状特征；只从几何构造，无标签拟合。同组零特征模型保留相同输入槽与参数布局。']
    zero=summaries.get('hash_bc_pointnet_local_zero_width128_10k')
    actual=summaries.get('hash_bc_pointnet_local_actual_width128_10k')
    if zero and actual:
        gap=100*(zero['best_metrics']['validation']['case_mean_relative_l2']-actual['best_metrics']['validation']['case_mean_relative_l2'])
        lines += [f'首个种子的局部特征收益为{gap:.2f}个百分点；第二个种子用于检查收益是否稳定，高误差仍未解决。']
        z2=summaries.get('hash_bc_pointnet_local_zero_width128_seed2027_10k')
        a2=summaries.get('hash_bc_pointnet_local_actual_width128_seed2027_10k')
        if z2 and a2:
            gap2=100*(z2['best_metrics']['validation']['case_mean_relative_l2']-a2['best_metrics']['validation']['case_mean_relative_l2'])
            lines += [f'第二个种子的收益为{gap2:.2f}个百分点。两次均改善平均与热点误差，但最差案例变差；不能称为全面改善。']
    lines += ['', '## 位移的载荷响应基','',
              '已实现ver/hor/tor位移响应的线性组合，并与同容量、允许响应基依赖载荷的控制模型比较。几何输入、训练预算和随机采样保持相同。',
              '45个训练/验证CAD的求解配置支持该设定；源FEM叠加残差见下方冻结诊断。训练只拟合原始uXYZ，使用训练集的零中心线性尺度；应力保留独立监督。']
    physics=[]
    for mode in ('linear_basis','load_dependent_basis'):
        summary=summaries.get('displacement_'+mode+'_local_actual_10k')
        if summary:
            physics.append((mode,summary['best_metrics']['train']['case_mean_relative_l2'],summary['best_metrics']['validation']['case_mean_relative_l2']))
    if physics:
        lines += ['', '| 位移响应基 | 最佳点训练误差 | 验证误差 |','|---|---:|---:|']
        lines += [f"| {'载荷无关，强制线性' if mode=='linear_basis' else '载荷依赖控制'} | {percent(train)} | {percent(val)} |" for mode,train,val in physics]
    relative=[]
    for mode in ('linear_basis','load_dependent_basis'):
        summary=summaries.get('displacement_'+mode+'_relative_case_10k')
        if summary:
            relative.append((mode,summary['best_metrics']['validation']['case_mean_relative_l2'],summary['best_metrics']['validation']['worst_case_relative_l2']))
    if relative:
        lines += ['', '训练集证据：相同相对误差下，扭转载荷仅占原共同RMS损失约1.3%的权重。改为训练案例RMS归一化损失，物理解码尺度保持不变：']
        lines += [f"- {'线性响应基' if mode=='linear_basis' else '载荷依赖对照'}：验证平均{percent(val)}，最差{percent(worst)}。" for mode,val,worst in relative]
        lines += ['训练扭转拟合明显改善，但验证平均收益小、最差案例变差，幅值权重不是全部泛化瓶颈。']
    probe=summaries.get('frozen_validation_dia_probe')
    if probe:
        u=probe['displacement']
        lines += ['', '冻结后仅评估五个验证CAD的dia组合（933,951个原节点）：',
                  f"- 位移平均误差：线性响应基{percent(u['linear_basis']['case_mean_relative_l2'])}，载荷依赖对照{percent(u['load_dependent_basis']['case_mean_relative_l2'])}。",
                  f"- 源FEM位移叠加残差平均{percent(u['oracle']['case_mean_relative_l2'])}，支持近似线性；不能说数值残差为零或已确定其来源。",
                  f"- 独立应力模型的dia平均误差仍为{percent(probe['stress']['case_mean_relative_l2'])}，组合载荷应力预测仍是主要瓶颈。",
                  '这是验证集载荷组合诊断，未读取最终测试CAD。线性模型的最大绝对位移误差仍可能高于对照，完整物理指标保留在原始summary。']
    uz=summaries.get('stress_u_condition_zero_width128_10k')
    ua=summaries.get('stress_u_condition_actual_width128_10k')
    lines += ['', '## 冻结预测位移作为应力条件','',
              '按常规载荷验证均值选择并冻结案例归一化损失的线性位移教师（29.178%）。每CAD只推断一次三组响应，再按当前载荷组合成三维预测位移；应力模型的对应三个槽使用预测值或全零。两者均从头训练，相同41维局部输入、32维整体描述、采样与10k预算。',
              '预测条件生成不读取真实位移或应力；教师本身使用额外的训练位移监督，因此整个pipeline不能声称与stress-only模型有相同监督量或总成本。两个候选教师训练成本合计约223秒，缓存生成和应力训练成本另列。']
    if uz or ua:
        lines += ['', '| 位移条件 | 验证应力均值 | 最差案例 | 热点均值 | 训练耗时 |','|---|---:|---:|---:|---:|']
        for mode,summary in [('零槽',uz),('冻结预测位移',ua)]:
            if summary:
                v=summary['best_metrics']['validation']
                lines += [f"| {mode} | {percent(v['case_mean_relative_l2'])} | {percent(v['worst_case_relative_l2'])} | {percent(v['case_mean_hotspot_relative_l2'])} | {summary['elapsed_seconds']:.1f}s |"]
            else:
                lines += [f'| {mode} | 待完成 | — | — | — |']
    else:
        lines += ['成对实验已排队，等待实现及共享响应缓存。此轮不打开dia或最终测试CAD。']
    if uz and ua:
        lines += ['位移条件改善了均值与最差案例，但热点变差；未通过预定的复验门。下一轮固定预测u3，比较廉价mesh恢复应变六槽与同布局全零槽，检验局部变形变化是否提供不同的信息。它是corner-P1离散代理，不是精确TETRA10应变，也不以本构公式替换源应力标签。']
    lines += ['', '## 解释边界','',
              '两个 CAD 的结果是迁移 pilot，存在其他历史实验使用；不是最终盲测。九个组合的四类载荷均已在其他训练 CAD 出现，不能称为全局未见载荷类别。新模型的最终测试CAD仅指未参与其训练/选模，原50案例历史实验包含这些CAD。',
              '迁移尾部误差可能被 asinh 逆变换放大；不截断预测、不更改标签，也不把训练误差称为泛化误差。',
              '候选模型组合已有的 [PointNet](https://arxiv.org/abs/1612.00593) 点集汇聚与 [Instant-NGP](https://nvlabs.github.io/instant-ngp/) hash 表示；当前尚无新算法或泛化成功的结论。','',
              f"本页随实验更新；累计完成 {len(summaries)}/{len(jobs)} 项，自主研究继续。",'']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    brief=['# GeoBC：一分钟结果','',
           '**目前的主要问题是共享场的泛化，而非单个案例能否拟合。**','',
           '- 难例单独训练：应力误差约0.1%；原共享Hash换CAD或载荷组合：约92%。',
           '- 整体几何编码和解码器加宽没有明显收益；局部边界几何仅将应力验证误差从约65%改善到约64%。']
    if zero and actual:
        brief += ['  局部几何收益在两个种子上重复，但最差案例变差，不能称为全面改善。']
    if probe:
        brief += ['- 组合载荷诊断：强制位移线性响应优于对照（28.4% vs 33.0%），但应力仍约87%；位移物理结构有用，应力迁移仍未解决。']
    if uz and ua:
        z=uz['best_metrics']['validation'];a=ua['best_metrics']['validation']
        brief += ['', f"最新对照：预测位移条件将应力验证均值{percent(z['case_mean_relative_l2'])} → {percent(a['case_mean_relative_l2'])}，最差也改善；热点{percent(z['case_mean_hotspot_relative_l2'])} → {percent(a['case_mean_hotspot_relative_l2'])}，反而变差。下一步检验局部应变条件，最终测试继续保留。"]
    else:
        brief += ['', '正在检验：冻结模型预测的位移，能否帮助应力预测；与相同输入布局的零条件做成对比较。']
    brief += ['', f"完成{len(summaries)}/{len(jobs)}项探索；原30项slides复现已完成。当前研究继续，五个最终测试CAD未参与新模型训练/选择。",'',
              '数据、控制变量及解释范围见 [完整探索报告](REPORT.md)。','']
    (dest/'BRIEF.md').write_text('\n'.join(brief),encoding='utf8')
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
