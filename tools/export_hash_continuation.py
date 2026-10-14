"""Scalar-only report for planned five-sample Hash continuation."""
import argparse,csv,json,sys
from pathlib import Path


def run(root):
    s=json.loads((root/'summary.json').read_text());t=json.loads((root/'terminal.json').read_text())
    if s['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or not t['within_budget'] or t['active'] or not t['closed']:
        raise ValueError('Closed successful stage required, precision success remains separate')
    a=s['arms']['five_hashgrid'];m=a['best']
    if len(s['arms'])!=1 or len(m['per_case'])!=5 or not a['actual_model_Adam_scheduler_RNG_cursor_restore']:raise ValueError('Incomplete actual continuation')
    out=Path('results/sample_overfitting');path=out/'hash_continuation_metrics.json'
    if path.exists():raise FileExistsError('Continuation already exported')
    old=json.loads((out/'current_metrics.json').read_text())
    before=next(r for r in old['aggregate'] if r['experiment']=='five_hashgrid')
    rows=[dict(sample=k,**v) for k,v in m['per_case'].items()]
    with (out/'hash_continuation_per_case.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'starting_step':a['starting_step'],'actual_updates':a['updates'],'added_updates':a['added_updates'],'selected_step':a['best_step'],
        'goal_met':a['goal_met'],'mean_U_percent':m['mean_U_percent'],'mean_S_percent':m['mean_S_percent'],'worst_field_percent':m['worst_field_percent'],
        'prior_arm_seconds':{'UTC':before['UTC_seconds'],'monotonic':before['monotonic_seconds']},'additional_stage_seconds':t['elapsed_seconds'],
        'combined_arm_plus_additional_stage_seconds':{'UTC':before['UTC_seconds']+t['elapsed_seconds']['UTC'],'monotonic':before['monotonic_seconds']+t['elapsed_seconds']['monotonic']},
        'prior_full_comparison_stage_seconds':old['terminal_seconds'],'restored_endpoint_percent':a['restored_endpoint_percent'],
        'model_Adam_scheduler_fourRNG_cursor_restore':True,'discarded_uncommitted_partial_backward':a['discarded_incomplete_backward'],
        'training_case_file_reads':s['source_reads'],'checkpoint_reads':a['checkpoint_reads'],'heldout_reads':s['heldout_reads'],'per_case':rows}
    try:import matplotlib
    except ModuleNotFoundError:
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'.local_deps'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt;import numpy as np
    x=np.arange(5);fig,ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
    for shift,key,label,color in [(-.18,'U_percent','U','#247fa9'),(.18,'S_percent','Signed S','#ed3a94')]:
        bars=ax.bar(x+shift,[r[key] for r in rows],width=.34,color=color,label=label)
        labels=ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        for annotation in labels:annotation.set_bbox(dict(facecolor='white',edgecolor='none',pad=.3))
    ax.axhline(1,color='#172638',linestyle='--',label='1% goal');ax.set_xticks(x,[r['sample'] for r in rows]);ax.set_ylabel('Full original-node relative L2 (%)');ax.legend(frameon=False)
    ax.set_ylim(0,max(1.2,max(max(r['U_percent'],r['S_percent']) for r in rows)*1.2));ax.set_title('Five-sample Torch Hash: continued joint U/S fitting',color='#172638',fontweight='bold')
    fig.savefig(out/'hash_continuation.png',dpi=180);plt.close(fig)
    text=['\n## 五样本续训实际结果\n',f"从{a['starting_step']}步保留模型、Adam、scheduler、四RNG与游标继续；新增{a['added_updates']}步，实际总更新{a['updates']}，同一最优checkpoint为{a['best_step']}步。额外stage耗时{t['elapsed_seconds']['UTC']:.2f}秒UTC / {t['elapsed_seconds']['monotonic']:.2f}秒单调时钟，原Hash臂成本另外保留，不与原16臂混成同预算排名。",f"每样本两字段均严格低于1%：{'通过' if a['goal_met'] else '未通过'}；最差字段{m['worst_field_percent']:.4f}%。其他骨干原轮未达标的事实保持不变。",'\n|样本|U %|S %|','|---|---:|---:|']
    text += [f"|{r['sample']}|{r['U_percent']:.4f}|{r['S_percent']:.4f}|" for r in rows]
    text += ['\n![五样本续训逐案例误差](hash_continuation.png)','\n[完整追加成本及指标](hash_continuation_metrics.json)。五样本训练拟合，不是泛化或独立测试精度。']
    with (out/'README.md').open('a',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'goal_met':a['goal_met'],'worst_field_percent':m['worst_field_percent'],'extra_GT_body_reads':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run(p.parse_args().run)
