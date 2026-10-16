"""Export new50 endpoints once; compare with already exported1/5 scalars only."""
import argparse,csv,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone,timedelta
from tools.export_fullcycle_u_overfitting import COMMON
from tools.export_onecycle_u_overfitting import write_csv
from tools.export_u_only_overfitting import NAMES


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def assemble(summary,config,previous):
    common=[k for k in COMMON if k not in ('single','five')]
    if any(config[k]!=previous['config'][k] for k in common):raise ValueError('Same common training recipe required')
    if config.get('early_stopping') or config.get('continuation') or config.get('learning_rate_by_method') or not config.get('fixed_endpoint') or not config.get('uniform_training_parameters') or config['max_updates_per_arm']!=5000 or config['learning_rate']!=.001:raise ValueError('Full5000 uniform recipe without early stopping required')
    keys=config['fifty']
    if len(keys)!=50 or len({k.split('/')[0] for k in keys})!=50 or keys[:5]!=config['five'] or config['single']!=keys[:1]:raise ValueError('Nested50 distinct CADs required')
    if set(summary['arms'])!={'fifty_'+m for m in NAMES}:raise ValueError('All8 fifty-case methods required')
    rows=[];cases=[]
    for method in NAMES:
        name='fifty_'+method;a=summary['arms'][name];m=a.get('endpoint',{})
        if a['status']!='completed' or a['updates']!=5000 or m.get('step')!=5000 or a.get('stop_reason')!='fixed_update_endpoint' or a.get('partial_backward') or a.get('optimizer_step_in_progress') or not a.get('actual_full_node_backward') or a.get('output_channels')!=3:raise ValueError('Complete actual5000 finite full-node U endpoint required')
        if a.get('endpoint_predictions_saved')!=50 or set(m.get('per_case',{}))!=set(keys) or a.get('case_updates')!={k:100 for k in keys}:raise ValueError('All50 original cases need100 complete updates and endpoint predictions')
        if any(abs(a[k]/config['parameter_target']-1)>config['parameter_tolerance'] for k in ('parameters','active_parameters')):raise ValueError('Actual capacity mismatch')
        if a['maximum_learning_rate']!=.001 or not math.isclose(a['learning_rate_last_update'],4e-8,rel_tol=1e-10):raise ValueError('Actual OneCycle endpoint LR mismatch')
        if any(set(v)!={'U_percent'} for v in m['per_case'].values()):raise ValueError('U-only metrics required')
        values=[m['per_case'][k]['U_percent'] for k in keys]
        if not all(math.isfinite(v) and v>=0 for v in values) or not math.isclose(sum(values)/50,m['mean_U_percent'],rel_tol=1e-10) or max(values)!=m['worst_field_percent']:raise ValueError('Endpoint scalar aggregation mismatch')
        rows.append(dict(experiment=name,method=method,group='fifty',parameters=a['parameters'],active_parameters=a['active_parameters'],updates=5000,maximum_learning_rate=.001,learning_rate_last_update=a['learning_rate_last_update'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],all_cases_U_below_1_percent=max(values)<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],source_run='fifty_u_overfitting',fresh_rerun_of_early_stop=False))
        cases.extend(dict(experiment=name,sample=k,updates=5000,U_percent=m['per_case'][k]['U_percent'],source_run='fifty_u_overfitting') for k in keys)
    return rows,cases


def plot(rows,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups=('single','five','fifty');titles=('1 fitted case','5 fitted cases','50 fitted cases')
    positive=[float(r[k]) for r in rows for k in ('U_percent','worst_U_percent') if float(r[k])>0]
    low=max(min(positive)*.3,1e-6);high=max(max(positive)*3,2)
    curve_high=max(high,max(float(r['U_percent']) for r in curves)*1.15)
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained',sharey=True)
    for ax,g,title in zip(axes,groups,titles):
        selected=[next(r for r in rows if r['method']==method and r['group']==g) for method in NAMES]
        x=np.arange(8);bars=ax.bar(x,[float(r['U_percent']) for r in selected],width=.6,color='#247fa9',label='Mean U')
        ax.bar_label(bars,labels=[f"{float(r['U_percent']):.3g}" for r in selected],padding=4,fontsize=8)
        ax.scatter(x,[float(r['worst_U_percent']) for r in selected],marker='_',s=90,color='#172638',label='Worst case U')
        ax.axhline(1,ls='--',color='#9b4b33',label='1% reference');ax.set_yscale('log');ax.set_ylim(low,high)
        ax.set_xticks(x,list(NAMES.values()),rotation=40,ha='right');ax.set_title(title+' / actual update 5000');ax.set_ylabel('Fitted-case U-vector relative L2 (%)')
        ax.legend(frameon=False,fontsize=8,loc='lower right')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained',sharey=True)
    for ax,g,title in zip(axes,groups,titles):
        for method,label in NAMES.items():
            p=sorted((r for r in curves if r['experiment']==g+'_'+method),key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in p],[float(r['U_percent']) for r in p],label=label,lw=1)
        ax.axhline(1,ls='--',color='#9b4b33');ax.set_yscale('log');ax.set_xlim(0,5000);ax.set_ylim(low,max(curve_high,200))
        ax.set_title(title,pad=55);ax.set_xlabel('Complete original-node case updates');ax.set_ylabel('Mean fitted-case U error (%)')
        ax.legend(frameon=False,fontsize=8,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.01))
    fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)


def analyze_exported_scalars(out):
    """Reproducible interpretation of the exported CSVs; no run/model/GT reads."""
    rows=read_csv(out/'all_methods.csv');cases=read_csv(out/'all_per_case.csv');curves=read_csv(out/'learning_curves.csv')
    config=json.loads((out/'metrics.json').read_text())['config'];anchors=[];distribution=[];loads=[]
    for method in NAMES:
        five=next(r for r in rows if r['group']=='five' and r['method']==method)
        subset=[r for r in cases if r['experiment']=='fifty_'+method];values=[float(r['U_percent']) for r in subset]
        common=[float(r['U_percent']) for r in subset if r['sample'] in config['five']]
        tail=next(r for r in curves if r['experiment']=='fifty_'+method and int(r['step'])==4000)
        anchors.append(dict(method=method,original_five_mean_U_percent=float(five['U_percent']),fifty_model_on_same_five_mean_U_percent=statistics.mean(common),fifty_model_on_same_five_worst_U_percent=max(common),fifty_all_mean_U_percent=statistics.mean(values)))
        distribution.append(dict(method=method,cases=50,U_below_1_percent=sum(v<1 for v in values),median_U_percent=statistics.median(values),mean_U_percent=statistics.mean(values),worst_U_percent=max(values),mean_U_at_4000=float(tail['U_percent']),mean_U_at_5000=statistics.mean(values)))
        for load in ('ver','hor','dia','tor'):
            v=[float(r['U_percent']) for r in subset if r['sample'].endswith('/'+load)]
            loads.append(dict(method=method,load=load,cases=len(v),mean_U_percent=statistics.mean(v),worst_U_percent=max(v)))
    write_csv(out/'same_five_cases.csv',anchors);write_csv(out/'error_distribution.csv',distribution);write_csv(out/'per_load.csv',loads)
    section=['## 标量结果解释','','在当前5000次更新预算下，FNO的50样本均值9.9989%、最差17.0790%，在这8个实现中均最低；TorchHash为21.3929%/47.5205%，其余均值约24.93%–29.53%。400个新增逐样本端点中，没有任何一个U<1%；八个单样本端点全部<1%，五样本仍仅TorchHash全案例通过。这是当前配方和名单的拟合结果，不是普遍架构排名。','',
        '仅比较原先相同的五个案例，50样本模型的误差也全部高于原五样本模型，说明整体均值升高并不只是因为新加入案例更难。下表完全从已导出的逐案例标量筛选，没有额外模型推理或GT读取。','',
        '|方法|原五样本模型：五案例均值U%|50样本模型：相同五案例均值U%|50样本模型：相同五案例最差U%|','|---|---:|---:|---:|']
    for r in anchors:section.append(f"|{NAMES[r['method']]}|{r['original_five_mean_U_percent']:.4f}|{r['fifty_model_on_same_five_mean_U_percent']:.4f}|{r['fifty_model_on_same_five_worst_U_percent']:.4f}|")
    section+=['','解释时需同时考虑：每样本只更新100遍，而五样本为1000遍；拟合几何和载荷组成、组内RMS输出尺度也改变。该实验没有控制每样本训练遍数、只增加数据量的单因素对照，不能将误差差异单独归因于容量不足或代码错误，也不能声称这已是收敛上限。', '',
        '末1000步的平均U误差仍有变化：FNO从12.5870%降到9.9989%，Transolver从29.9956%降到25.1500%，++从30.0088%降到25.6559%，TS3从28.2545%降到24.9284%；MLP/NGP4端点则略高于4000步。没有用更好的中间值替换5000步。学习率末段趋零带来的曲线变平不足以证明架构瓶颈；++还保留官方随机Gumbel评价。', '',
        '50案例载荷构成为ver15、hor13、dia9、tor13，不应把各载荷案例看作完全同难度。提供[按载荷标量](per_load.csv)、[50案例误差分布与末段数值](error_distribution.csv)和[共同五案例比较](same_five_cases.csv)供追查，未追加训练或读取科学数组。','']
    path=out/'README.md';text=path.read_text(encoding='utf-8').split('\n## 标量结果解释')[0]
    path.write_text(text.rstrip()+'\n\n'+'\n'.join(section),encoding='utf-8')


def run(root,config_path,base,recovery=None):
    c=json.loads(config_path.read_text());previous=json.loads((base/'metrics.json').read_text())
    s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or t['active'] or not t['closed'] or not t['within_budget'] or not t['original_deadline_respected'] or s['source_reads']!=50 or s['stress_label_reads'] or s['heldout_reads']:raise ValueError('Successful closed50 TRAIN U-only stage required')
    recovery_note=None;stage_costs={'original':t['elapsed_seconds']};source_reads=50;recovered=False
    if recovery is not None:
        r=json.loads((recovery/'summary.json').read_text());rt=json.loads((recovery/'terminal.json').read_text())
        a=r.get('arms',{}).get('fifty_transolverpp',{});old=s['arms']['fifty_transolverpp']
        if old['status']!='failed' or old['updates']!=21 or old['error']!='Declared GPU memory cap' or r['status']!='completed' or r['active'] or not r['closed'] or rt['exit_code']!=0 or not rt['within_budget'] or not rt['original_deadline_respected'] or r['source_reads']!=50 or r['stress_label_reads'] or r['heldout_reads'] or r.get('checkpoint_reads')!=1:raise ValueError('Successful inherited-budget ++ recovery required')
        if a.get('recovered_from_actual_step')!=21 or a.get('added_updates')!=4979 or not a.get('model_Adam_scheduler_fourRNG_cursor_restored') or not a.get('resumed_full_node_backward_finite') or not r.get('restored_without_readout_or_RNG_draw'):raise ValueError('Actual saved21 plus4979 continuation required; no replay')
        if rt['inherited_clock']['original_deadline_epoch']!=json.loads((root/'launch.json').read_text())['deadline_epoch']:raise ValueError('Original whole-stage deadline changed')
        s['arms']['fifty_transolverpp']=a;recovered=True;source_reads+=r['source_reads'];stage_costs['recovery']=rt['elapsed_seconds']
        recovery_note=dict(starting_step=21,added_updates=4979,prior_failed_arm_seconds=old['elapsed_seconds'],queue_wait_seconds=rt['inherited_clock']['queue_wait_seconds'],queue_wait_overlaps_original=True,clock=rt['inherited_clock'],data_residency=r['data_residency'])
    rows,cases=assemble(s,c,previous)
    if recovered:
        for row in rows+cases:
            if row['experiment']=='fifty_transolverpp':row['source_run']='fifty_u_recovery'
    old_rows=read_csv(base/'summary.csv');old_cases=read_csv(base/'per_case.csv');curves=read_csv(base/'learning_curves.csv')
    if len(old_rows)!=16 or len(old_cases)!=48 or any(int(r['updates'])!=5000 for r in old_rows+old_cases):raise ValueError('Existing16 exact5000 scalar endpoints required')
    for r in rows:
        points=[json.loads(line) for line in (root/r['experiment']/'metrics.jsonl').read_text().splitlines()]
        if recovered and r['experiment']=='fifty_transolverpp':points.extend(json.loads(line) for line in (recovery/'metrics.jsonl').read_text().splitlines())
        if [m['step'] for m in points]!=list(range(0,5001,50)):raise ValueError('Complete eval50 scalar curve required')
        curves.extend(dict(experiment=r['experiment'],step=m['step'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],learning_rate_last_update=m.get('learning_rate_last_update'),source_run=r['source_run']) for m in points)
    out=Path(c['report_directory']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported; replot only from exported scalars')
    all_rows=old_rows+rows;all_cases=old_cases+cases
    write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases)
    write_csv(out/'all_methods.csv',all_rows);write_csv(out/'all_per_case.csv',all_cases);write_csv(out/'learning_curves.csv',curves)
    plot(all_rows,curves,out)
    elapsed={k:sum(v[k] for v in stage_costs.values()) for k in ('UTC','monotonic')}
    end_root=recovery if recovered else root;end_t=json.loads((end_root/'terminal.json').read_text());launch=json.loads((end_root/'launch.json').read_text());end=datetime.fromtimestamp(launch['entry_epoch']+end_t['elapsed_seconds']['UTC'],timezone(timedelta(hours=8))).isoformat()
    lines=['# 1 / 5 / 50 样本：全部8方法U-only，统一完整5000步','',f"新50样本阶段于北京时间{end}结束，8项均为实际第5000次完整原节点更新的端点。已有1/5样本16项5000步标量原样保留；不重训或重新读出历史模型/GT，不挑best。",'',
        '50个不同CAD，每CAD一个载荷，固定包含原来的5个案例。先保留5锚点，再按现有归档顺序选剩余TRAIN40案例，最后按ID顺序补8个additionalTRAIN60 CAD，载荷ver/hor/tor循环。名单在读数值标签之前固定；未使用DEV、test或FINAL。完整名单见[配置](../../configs/fifty_u_samples.json)。', '',
        '全部组同seed2026、约460万±1%实际注册/活跃参数、同结构和28通道输入(normalizedXYZ+25原条件)、3通道signed Uxyz(mm)、仅U-vector relativeL2平方损失。Adam betas(.9,.999)、eps1e-8、weight_decay0、fused；梯度L2clip1；OneCycle总5000、pct_start.1、cos两阶段、div25/finaldiv1000、cycle_momentumFalse。第1次实际LR4e-5、第500峰值1e-3、第5000次4e-8。所有方法统一，无早停，无1%提前退出。', '',
        '一次更新覆盖一个CAD×载荷的全部原节点，固定循环。5000更新对应单样本5000遍、五样本各1000遍、50样本各100遍；这是相同更新预算，不能当作相同每样本训练遍数。每50步完整评价全组，50样本评价开销更高；组内U输出RMS仅用对应拟合样本，损失仍逐case相对能量归一化。', '',
        '原阶段50个训练case在GPU共享缓存中各加载一次，所有8方法共享；42个来自已对齐NPZ，8个来自既有对齐TRAIN NPY。后者仅复制interleaved physical.npy的前三列U，不使用第四列S。无sampleID、physicalU、teacher、stress或heldout输入。避免无用途的best模型/预测写盘；每次评价与最终全状态last.pt、四RNG、Adam/scheduler、order/cursor、实际成本、50份端点预测均保留本地。恢复阶段的额外读取、CPU缓存和实际成本另列。', '',
        'FP32/TF32，TCNN原生FP16/FP32master；++保留官方eval随机Gumbel单次读出。统一训练配方不等于数学完全公平；这是单seed训练集拟合比较，不是泛化测试或架构上限。50样本量和载荷组成也发生变化。', '',
        '|方法|1样本U%|5样本均值%|5样本最差%|50样本均值%|50样本最差%|各项步数|','|---|---:|---:|---:|---:|---:|---:|']
    for method,label in NAMES.items():
        one,five,fifty=[next(r for r in all_rows if r['group']==g and r['method']==method) for g in ('single','five','fifty')]
        lines.append(f"|{label}|{float(one['U_percent']):.4f}|{float(five['U_percent']):.4f}|{float(five['worst_U_percent']):.4f}|{fifty['U_percent']:.4f}|{fifty['worst_U_percent']:.4f}|5000|")
    passes=sum(r['all_cases_U_below_1_percent'] for r in rows)
    lines+=['',f"50样本组有{passes}/8方法的所有案例均U<1%。本表保留实际5000端点，即使某方法比小样本或旧中间值更差也不替换。",'',
        f"新50样本任务完整计算阶段成本{elapsed['UTC']:.2f}s UTC / {elapsed['monotonic']:.2f}s单调时钟，包含导入、缓存、模型构造、训练、评价、存盘和退出；预算未延长。source case读取{source_reads}次、stress数值列/heldout读取0。已有1/5完整交付阶段成本{previous['stage_seconds']['total']['UTC']:.2f}s UTC / {previous['stage_seconds']['total']['monotonic']:.2f}s单调时钟单列，包含其早停废弃工作；本次未重算。导出只读标量，模型/GT/科学数组读取0。", '',
        '![完整5000步比较](comparison.png)','','![原始学习曲线，无平滑](learning_curves.png)','','[新增8项](summary.csv) · [新增400个逐样本端点](per_case.csv) · [全部24项](all_methods.csv) · [全部448行逐样本](all_per_case.csv) · [学习曲线标量](learning_curves.csv) · [历史1/5报告](../fullcycle_u_overfitting/README.md)']
    if recovered:lines+=['','Transolver++在完成21次更新后、下一次更新开始前触发30.5GB显存上限。其余7项完成后，原model/Adam/OneCycle/四RNG/order/cursor从保存的21步状态接续4979步，无零模型重放、无额外读出改变RNG。只将共享数据缓存移到CPU，逐完整案例转入GPU并释放allocator缓存；结构、精度、损失、样本顺序、LR与梯度裁剪不变。恢复仍受原whole截止时间及保守的原++起点+10800秒双时钟截止约束，等待也消耗原arm余量。恢复额外共享读取50个TRAIN案例，总100次；等待与原TS3重叠，不重复计成本。恢复前后实际成本均包含于上文，分项见metrics.json。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(dict(config=c,summary=rows,per_case=cases,stage_seconds=elapsed,stage_costs=stage_costs,recovery=recovery_note,previous_stage_seconds=previous['stage_seconds'],source_reads=source_reads,stress_label_reads=0,heldout_reads=0,scientific_array_reads_by_exporter=0,actual_export=1),indent=2)+'\n')
    analyze_exported_scalars(out)
    print(json.dumps(dict(new_arms=8,all_arms=24,new_per_case=400,all_per_case=448,all_actual_updates=5000,scientific_array_reads=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/fifty_u_overfitting.json'));p.add_argument('--base',type=Path,default=Path('results/fullcycle_u_overfitting'));p.add_argument('--recovery-run',type=Path);a=p.parse_args();run(a.run,a.config,a.base,a.recovery_run)
