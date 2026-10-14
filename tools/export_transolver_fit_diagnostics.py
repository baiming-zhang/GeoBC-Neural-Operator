"""Export bounded Transolver correctness and learning-rate evidence, scalar only."""
import csv,json
from pathlib import Path


def run():
    d=json.loads(Path('runs/transolver_fit_diagnostics/diagnostics.json').read_text())
    control=Path('runs/transolver_single_u_learning_rate_control')
    s=json.loads((control/'summary.json').read_text());t=json.loads((control/'terminal.json').read_text())
    a=s['arms']['single_transolver'];m=a['endpoint']
    if d['status']!='completed' or s['active'] or not s['closed'] or t['exit_code']!=0 or not t['within_budget'] or a['updates']!=1000 or m['step']!=1000:raise ValueError('Complete successful diagnostics and strict endpoint required')
    out=Path('results/transolver_fit_diagnostics');out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Already exported')
    old=[json.loads(line) for line in Path('runs/matched_capacity_u_overfitting/single_transolver/metrics.jsonl').read_text().splitlines()]
    new=[json.loads(line) for line in (control/'single_transolver/metrics.jsonl').read_text().splitlines()]
    rows=[{'learning_rate':lr,'step':v['step'],'U_percent':v['mean_U_percent']} for lr,data in [(.001,old),(.0001,new)] for v in data]
    with (out/'learning_rate_curve.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for lr,data,color in [(.001,old,'#ed3a94'),(.0001,new,'#247fa9')]:
        ax.plot([v['step'] for v in data],[v['mean_U_percent'] for v in data],'-o',color=color,label=f'Adam fixed LR {lr:g}',markersize=4)
    ax.axhline(d['data']['constant_U_mean_error_percent'],linestyle=':',color='#172638',label='Best constant U baseline')
    ax.axhline(1,linestyle='--',color='#777777',label='1% goal')
    ax.set_xlabel('Full original-node optimizer updates');ax.set_ylabel('U-vector relative L2 error (%)');ax.legend(frameon=False);ax.set_title('Transolver: same seed, inputs, model and 1000 updates')
    fig.savefig(out/'learning_rate_control.png',dpi=180);plt.close(fig)
    relevant={k:d[k] for k in ('parity','saved_endpoints','real_probes','synthetic_probes','source_reads','stress_label_reads','heldout_reads')}
    relevant['sample']={'key':d['data']['sample'],'nodes':d['data']['nodes'],'U_RMS_mm':d['data']['U_RMS_mm'],'constant_U_mean_error_percent':d['data']['constant_U_mean_error_percent']}
    relevant['diagnostic_seconds']={'scope':d.get('timing_scope','run entry after module imports'),**d['elapsed_seconds']}
    relevant['learning_rate_control']={'config':json.loads(Path('configs/transolver_single_u_learning_rate_control.json').read_text()),'endpoint':m,'updates':1000,'parameters':a['parameters'],'elapsed_seconds':t['elapsed_seconds'],'source_reads':s['source_reads'],'stress_label_reads':s['stress_label_reads'],'heldout_reads':s['heldout_reads']}
    (out/'metrics.json').write_text(json.dumps(relevant,indent=2)+'\n')
    import_note='' if d.get('timing_scope')=='before_module_imports' else '（该首轮计时不含模块导入）'
    lines=['# Transolver拟合异常：接入正确性与学习率诊断','',
        '本报告检查此前U-only单样本Transolver的76.0923%异常结果。原16项固定1000步结果保留，不把本诊断的新配方替换到原主表。', '',
        '直接对照本地固定官方源码路径与适配器，三个模型在513个真实原节点上、相同权重和随机数状态下，前向输出、完整参数梯度及一次Adam更新的差异均为0。TS3使用257节点分块，检查了跨块全局路径；TS++使用单rank并恢复Gumbel RNG。原版另做120,777全原节点的100步训练，检查点开启与关闭曲线完全相同。该覆盖范围内未发现输入拼接、检查点反向、优化器漏更新等接入错误，不宣称所有可能配置均已证明正确。', '',
        '原节点U向量RMS为0.2949119 mm，所有节点预测同一个最佳平均位移时误差为76.1546%。旧原版1000步模型为76.0923%，预测节点空间变化相对范数仅4.05%，属于近常量预测；不是学到了完整位移场。固定1e-3的曲线先停在该基线附近，800步曾到47.7851%，1000步又退回76.0923%，显示未稳定收敛。', '',
        '保存模型、U缩放与评估核对：原版与TS3的误差完全复现，缩放差为0，Adam已执行1000步。TS++因官方eval Gumbel随机性，重读为31.7057%而已存端点31.7058%，不是归一化或报告错误。', '',
        f"仅把原版单样本固定学习率从1e-3改为1e-4，其他科学条件保持相同、fresh seed2026、4,586,786参数、120,777完整节点、恰好1000更新，端点U从{old[-1]['mean_U_percent']:.4f}%变为{m['mean_U_percent']:.4f}%。100步小对照也由76.0821%变为42.7181%。这证明该固定配方对学习率敏感；仍须按新端点实际判断1%目标，不能把短探针或合成目标当作原物理U通过。", '',
        '官方参考训练入口包含调度与部分梯度裁剪：原版exp_elas.py使用AdamW和CosineAnnealingLR；++ train_airplane.py使用Adam、OneCycleLR和clip_grad_norm_(1.0)；TS3 train_volume.py使用AdamW、余弦调度和clip_grad_norm_(1.0)。这些任务配方不直接证明适合本DeepJEB样本，但此前统一固定Adam .001、无调度、无裁剪不是官方收敛配方复现。固定配方对照不能据此断言Transolver架构记忆能力差。', '',
        '另有参数口径修正：++原生embedding在condition=None输入路径不参与，包含1,152个注册可训练参数；本次梯度覆盖的活跃参数为4,609,512，而原表记录注册数4,610,664。两种计数均在460万±1%内，该差异不是76%误差原因。其余原版和TS3参数均参与所检查的反向路径。', '',
        '三个513节点的合成坐标目标诊断显示能学习空间变化，500步训练误差原版4.3431%、++3.7947%、TS3 3.4927%；这不是物理U精度，也没有证明全部过拟合到1%。', '',
        f"正确性/短探针计时{d['elapsed_seconds']['UTC']:.2f}秒UTC / {d['elapsed_seconds']['monotonic']:.2f}秒单调时钟{import_note}；独立学习率控制完整阶段{t['elapsed_seconds']['UTC']:.2f} / {t['elapsed_seconds']['monotonic']:.2f}秒，含导入、数据、训练、读出、保存与退出。各阶段共享读取一份U-only缓存一次，stress/held-out读取0。所有新结果另行计费，原已关闭实验的预算未重置或延长。", '',
        '![学习率对照](learning_rate_control.png)', '',
        '[标量诊断记录](metrics.json) · [学习率曲线CSV](learning_rate_curve.csv) · [控制配置](../../configs/transolver_single_u_learning_rate_control.json) · [原U-only对照](../matched_capacity_u_overfitting/README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'old_U_percent':old[-1]['mean_U_percent'],'new_U_percent':m['mean_U_percent'],'new_updates':1000,'native_parity_passed':all(v['passed'] for v in d['parity'].values()),'scientific_array_reads_by_exporter':0}))

if __name__=='__main__':run()
