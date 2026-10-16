"""Scalar-only exact5000 index: retain full endpoints and replace every early stop."""
import argparse,csv,json,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
from tools.export_onecycle_u_overfitting import write_csv,write_plot
from tools.export_u_only_overfitting import NAMES

COMMON=('single','five','seed','max_updates_per_arm','learning_rate','minimum_learning_rate','lr_schedule','gradient_clip_norm','evaluate_every','parameter_target','parameter_tolerance','model_settings','output_field','precision')


def write_curves(curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for ax,group in zip(axes,('single','five')):
        for method in NAMES:
            points=sorted((r for r in curves if r['experiment']==group+'_'+method),key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in points],[float(r['U_percent']) for r in points],lw=1.2,label=NAMES[method])
        ax.axhline(1,color='#172638',ls='--',lw=1)
        ax.set_yscale('log');ax.set_ylim(1e-3,200);ax.set_xlim(0,5000)
        ax.set_xlabel('Complete original-node optimizer updates');ax.set_ylabel('Mean fitted-case U-vector error (%)')
        ax.set_title(group+' / full 5000-update learning curves',pad=65)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=4,frameon=False,fontsize=8)
    fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)


def assemble(base,rerun,original,config):
    if any(original[k]!=config[k] for k in COMMON) or config.get('early_stopping') or not config.get('fixed_endpoint') or config['learning_rate']!=1e-3 or config['max_updates_per_arm']!=5000 or config.get('learning_rate_by_method'):raise ValueError('Same common recipe and disabled early stopping required')
    expected={g+'_'+m for g in ('single','five') for m in NAMES}
    if set(base['arms'])!=expected:raise ValueError('Original16 denominator incomplete')
    replacements={k for k,a in base['arms'].items() if a['updates']<5000}
    if any(base['arms'][k]['stop_reason']!='early_stopping_plateau' for k in replacements):raise ValueError('Only genuine early stops may be replayed')
    if set(rerun['arms'])!=replacements or set(config['selected_arms'])!=replacements:raise ValueError('Every early stop must be replaced exactly once')
    rows=[];cases=[]
    for name in sorted(expected):
        a=(rerun if name in replacements else base)['arms'][name];m=a.get('endpoint')
        if a['status']!='completed' or a['updates']!=5000 or not m or m['step']!=5000 or a['output_channels']!=3 or not a.get('actual_full_node_backward') or a.get('partial_backward') or a.get('optimizer_step_in_progress'):raise ValueError('Every final arm must have a complete actual5000 endpoint')
        if any(abs(a[k]/config['parameter_target']-1)>config['parameter_tolerance'] for k in ('parameters','active_parameters')):raise ValueError('Actual capacity mismatch')
        if set(m['per_case'])!=set(config[a['group']]) or a.get('endpoint_predictions_saved')!=len(config[a['group']]) or any(set(v)!={'U_percent'} for v in m['per_case'].values()):raise ValueError('Complete U-only predictions required')
        if a['maximum_learning_rate']!=1e-3 or not math.isclose(a['learning_rate_last_update'],4e-8,rel_tol=1e-10):raise ValueError('Actual uniform OneCycle endpoints required')
        values=[v['U_percent'] for v in m['per_case'].values()]
        if not all(math.isfinite(v) for v in values) or not math.isclose(sum(values)/len(values),m['mean_U_percent'],rel_tol=1e-10) or max(values)!=m['worst_field_percent']:raise ValueError('Endpoint scalar aggregation mismatch')
        source='fullcycle_u_overfitting' if name in replacements else 'uniform_onecycle_u_overfitting'
        rows.append(dict(experiment=name,method=a['method'],group=a['group'],parameters=a['parameters'],active_parameters=a['active_parameters'],updates=5000,maximum_learning_rate=a['maximum_learning_rate'],learning_rate_last_update=a['learning_rate_last_update'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],all_cases_U_below_1_percent=max(values)<1,UTC_seconds=a['elapsed_seconds']['UTC'],monotonic_seconds=a['elapsed_seconds']['monotonic'],source_run=source,fresh_rerun_of_early_stop=name in replacements))
        cases.extend(dict(experiment=name,sample=k,updates=5000,U_percent=v['U_percent'],source_run=source) for k,v in m['per_case'].items())
    return rows,cases


def interpretation(rows,base):
    def row(name):return next(r for r in rows if r['experiment']==name)
    passes={g:sum(float(r['worst_U_percent'])<1 for r in rows if r['group']==g) for g in ('single','five')}
    lines=['## 结果解释','',f"完整5000步端点：单样本{passes['single']}/8方法通过U<1%，五样本{passes['five']}/8方法达到每样本U<1%。TorchHash五样本均值0.6831%、最差0.7506%；本轮其余五样本方法均未全部通过。该结论只覆盖当前单seed、5个拟合CAD×载荷及这套训练配方。",'']
    for method in ('mlp','ngp4'):
        before=base['arms']['five_'+method]['endpoint'];after=row('five_'+method)
        mean=float(after['U_percent']);worst=float(after['worst_U_percent']);reduction=100*(1-mean/before['mean_U_percent'])
        lines.append(f"{NAMES[method]}五样本：早停均值{before['mean_U_percent']:.4f}% → 完整5000步{mean:.4f}%（相对下降{reduction:.2f}%），最差{before['worst_field_percent']:.4f}% → {worst:.4f}%。低学习率尾段确实改变了结论，不能把旧早停值当作拟合瓶颈；完整5000步后仍未每样本达1%。")
    lines+=['','MLP五样本的主要残差集中于506_533/ver（11.3329%）和10_30/ver（9.6809%），其余三个案例均<1%；NGP4相同两个ver案例为10.1729%/8.6820%，另三个为约1.11%–1.40%。这是误差分布观察，尚未验证其机制，不能直接归因于参数量或几何输入。','',f"更多更新不保证端点单调变好：TCNN单样本早停{base['arms']['single_tcnn_hash']['endpoint']['mean_U_percent']:.4f}% → 完整5000步{float(row('single_tcnn_hash')['U_percent']):.4f}%。主表按要求保留后者，不挑选更好的旧端点。",'', 'Transolver系列本轮单样本全部<1%，与此前1000步配方的失败不同；但步数、学习率过程及峰值共同改变，不能把改善归因于单一因素。即使末段曲线变平，本轮也只证明5000步后的实际表现，不能证明架构上限或独立泛化能力。','']
    return lines


def load_closed(root):
    s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or t['active'] or not t['closed'] or not t['within_budget'] or s['source_reads']!=5 or s['stress_label_reads'] or s['heldout_reads']:raise ValueError('Successful closed TRAIN U-only stage required')
    return s,t


def run(base_root,rerun_root,config_path):
    c=json.loads(config_path.read_text());original=json.loads(Path('configs/uniform_onecycle_u_overfitting.json').read_text())
    base,bt=load_closed(base_root);rerun,rt=load_closed(rerun_root)
    rows,cases=assemble(base,rerun,original,c)
    out=Path(c['report_directory']);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported; replot only from exported scalars')
    curves=[]
    for row in rows:
        source=rerun_root if row['fresh_rerun_of_early_stop'] else base_root
        for line in (source/row['experiment']/'metrics.jsonl').read_text().splitlines():
            m=json.loads(line);curves.append(dict(experiment=row['experiment'],step=m['step'],U_percent=m['mean_U_percent'],worst_U_percent=m['worst_field_percent'],learning_rate_last_update=m.get('learning_rate_last_update'),source_run=row['source_run']))
    write_csv(out/'summary.csv',rows);write_csv(out/'per_case.csv',cases);write_csv(out/'learning_curves.csv',curves)
    write_plot(rows,out,title_suffix='complete update 5000');write_curves(curves,out)
    elapsed={k:bt['elapsed_seconds'][k]+rt['elapsed_seconds'][k] for k in ('UTC','monotonic')}
    launch=json.loads((rerun_root/'launch.json').read_text());ended=datetime.fromtimestamp(launch['entry_epoch']+rt['elapsed_seconds']['UTC'],timezone(timedelta(hours=8))).isoformat()
    lines=['# 最新全部8方法U-only：统一OneCycle，全部完整5000步','',f"北京时间{ended}完成。最终16项每项实际更新5000次，均使用第5000步完整原节点端点。已跑满项目保留；所有早停项目按用户要求从seed2026重新初始化、关闭早停重跑。旧早停值和中间误差不替代主表，不选best checkpoint。",'',
        '所有方法统一Adam betas(.9,.999)、eps1e-8、weight_decay0；每次optimizer后调用OneCycleLR，total_steps5000、pct_start0.1、cos两阶段、cycle_momentumFalse、div_factor25、final_div_factor1000。第1步4e-5、第500步峰值1e-3、第5000步4e-8；全参数梯度L2裁剪1，完整评价每50步。各方法沿用自身原结构，重跑前后样本顺序、输入、U-only损失和seed不变。OneCycle按步数调度，不是根据loss寻找LR。', '',
        '早停原轮最早1500步、以最差样本历史最优600步未改善作停止规则，截断了部分项目的低学习率阶段。例如五样本MLP在1550步、LR8.72e-4停止，但均值误差仍从950步11.02%改善到6.50%；NGP4在2150步、LR7.03e-4停止，2100步均值5.70%。这些不是充分收敛或架构瓶颈的证据。因此全部提前停止项重跑，已完整5000步项不做无效重算；重跑只移除停止判断，优化配方相同。', '',
        '一次更新完整一CAD×载荷全部原节点；single506_533/ver共5000遍；five加506_25/tor、10_30/ver、10_421/hor、10_460/dia各1000遍。signed Ux/Uy/Uz(mm)，U-vector relativeL2平方损失；归一化XYZ+25原条件，无sampleID、物理U、teacher、应力或heldout输入。输出RMS仅拟合样本。实际注册和full-node有限梯度参与参数均460万±1%。++保留原生未参与无条件分支embedding，活跃数单列。', '',
        'FP32/TF32；TCNN原生FP16/FP32master；++保留官方eval随机Gumbel单次读出。训练参数一致不意味着数学完全公平、架构上限或泛化排名。小于1%的验收要求每个完整样本，而不是均值。', '',
        f"原统一轮成本{bt['elapsed_seconds']['UTC']:.2f}s UTC / {bt['elapsed_seconds']['monotonic']:.2f}s单调时钟；新重跑阶段{rt['elapsed_seconds']['UTC']:.2f}s / {rt['elapsed_seconds']['monotonic']:.2f}s。合计{elapsed['UTC']:.2f}s / {elapsed['monotonic']:.2f}s，包含旧早停已用成本、导入、缓存、构造、训练、评价、保存和退出，非仅选中模型的训练耗时。每阶段5TRAIN NPZ共享各读一次，两阶段合计10次；stress/heldout读取0。等待队列与原阶段重叠，不重复累加。更早被用户改配方中断的3000步轮758.16s UTC/696.08s单调时钟独立保留，不冒充技术恢复。两个阶段退出码均0、预算未延长。完整预测和model/Adam/scheduler/四RNG/order/cursor/clock留本地；导出仅访问标量，科学数组读取0。", '',
        '|方法|单样本U%|五样本U均值%|五样本最差U%|单/五步数|','|---|---:|---:|---:|---:|']
    for method in NAMES:
        one=next(r for r in rows if r['method']==method and r['group']=='single');five=next(r for r in rows if r['method']==method and r['group']=='five')
        lines.append(f"|{NAMES[method]}|{one['U_percent']:.4f}|{five['U_percent']:.4f}|{five['worst_U_percent']:.4f}|5000 / 5000|")
    lines+=['']+interpretation(rows,base)+['## 端点来源','','|组|方法|实际步数|活跃参数|来源|全样本U<1%|','|---|---|---:|---:|---|---|']
    for r in rows:lines.append(f"|{r['group']}|{NAMES[r['method']]}|5000|{r['active_parameters']}|{'关闭早停后fresh重跑' if r['fresh_rerun_of_early_stop'] else '保留原轮完整5000'}|{'是' if r['all_cases_U_below_1_percent'] else '否'}|")
    lines+=['','![全部5000步端点](comparison.png)','','![完整学习曲线](learning_curves.png)','','[16项汇总](summary.csv) · [48行逐样本](per_case.csv) · [学习曲线标量](learning_curves.csv) · [配置与实际成本](metrics.json)','','此前[固定1000步对照](../matched_capacity_u_overfitting/README.md)与[Transolver1000步调整](../stabilized_u_overfitting/README.md)作为历史独立保留。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(dict(config=c,original_config=original,summary=rows,per_case=cases,stage_seconds=dict(original=bt['elapsed_seconds'],rerun=rt['elapsed_seconds'],total=elapsed),source_reads_by_stage=[5,5],stress_label_reads=0,heldout_reads=0,scientific_array_reads_by_exporter=0),indent=2)+'\n')
    print(json.dumps(dict(arms=len(rows),per_case_rows=len(cases),fresh_replays=len(c['selected_arms']),all_steps=5000,scientific_array_reads=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-run',type=Path,default=Path('runs/uniform_onecycle_u_overfitting'));p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(a.base_run,a.run,a.config)
