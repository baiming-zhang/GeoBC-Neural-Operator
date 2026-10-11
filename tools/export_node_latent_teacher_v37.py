"""Scalar-only one-page teacher/operator diagnosis after exact real terminal."""
import argparse


def terminal_metadata():
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash_physical_readout_contract_v18 import complete_physical
    plan=d.read('notes/geobc_node_latent_teacher_v37_plan.json');r=d.read(plan['root_review'])
    exporter=d.read('notes/node_latent_teacher_v37_exporter_root_review.json')
    assert exporter['passed'] and exporter['whole_source_reviewed'] and exporter['actual_export_released']
    assert exporter['source_sha256']==d.sha('tools/export_node_latent_teacher_v37.py')
    assert plan['scientific_digest']==d.digest(plan['scientific'])==r['scientific_digest']
    assert r['passed'] and r['actual_execution_released'] and plan['source_pins']==r['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    folder=d.path(plan['output']);s=d.read(folder/'summary.json');e=d.read(folder/'external_terminal.json')
    assert s==d.read(folder/'status.json') and s['status']=='completed' and s['protocol_qualified']
    assert s['closed'] and not s['active'] and e['closed'] and not e['active']
    assert e['exit_code']==0 and e['within_external_budget'] and not e['external_timeout']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert s['execution']['run_process']==e['child_process'] and s['execution']['guard']['guard_process']==e['guard_process']
    assert s['execution']['scientific_digest']==e['scientific_digest']==plan['scientific_digest']
    assert e['source_pins']==plan['source_pins']
    assert max(s['elapsed_epoch_seconds'],s['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=1800
    assert max(s[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    assert s['committed_finite_predictions']==180 and s['owned_GT_array_reads']==s['owned_GT_read_attempts']==60
    assert s['raw_GT_reads']==s['FINAL_numeric_reads']==s['training_updates']==s['scaler_fits']==0
    assert s['prediction_barrier_complete'] and s['teacher_comparisons_complete'] and s['eligible_best'] is None
    for row in s['candidates'].values():complete_physical(row['full60'],plan['scientific']['case_keys'])
    tp=d.read(plan['scientific']['training_plan_identity']['path']);tf=d.path(tp['output'])
    t=d.read(tf/'summary.json');te=d.read(tf/'external_terminal.json')
    for k,file in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]:
        assert d.identity(tf/file,hashed=True)==r['training_binding'][k]
    assert t['status']=='completed' and t['closed'] and not t['active'] and te['closed'] and not te['active']
    assert te['exit_code']==0 and te['within_external_budget'] and all(d.exact_process_exited(te[k]) for k in ('child_process','guard_process'))
    assert t['execution']['scientific_digest']==te['scientific_digest']==tp['scientific_digest']==plan['scientific']['training_SCI']
    assert t['FEM_used_for_optimizer_updates'] is False and t['cache_receipt']['physical_TRAIN_target_reads']==300
    for arm in t['methods'].values():
        assert arm['step']==arm['optimizer_step_attempts']==arm['optimizer_step_completions']==1500
        assert arm['actual_largest_fullN_gradient_check']['passed'] and arm['saved_Adam_steps_equal_cursor']
        assert arm['initial_fullTRAIN_teacher_node_latent']['all_cases_complete'] and arm['endpoints']['1500']['full_TRAIN_teacher_node_latent']['all_cases_complete']
    return d,plan,s,e,t,te


def export():
    import time
    began=time.monotonic()
    d,p,s,e,t,te=terminal_metadata()
    out=d.path('results/geobc_node_latent_teacher_v37');assert not (out/'manifest.json').exists(),'No automatic report replay'
    out.mkdir(parents=True,exist_ok=True)
    roles=[('TRAIN_TEACHER_R16_FNO16_1500','FNO16 teacher physical5'),('R16_TRANSOLVER_LATENT_1500','TS frozen-head latent5'),('R16_FNO16_LATENT_1500','FNO16 frozen-head latent5')]
    rows=[]
    for name,label in roles:
        row=s['candidates'][name];field=row['full60'];teacher=row['is_teacher']
        rows.append(dict(name=name,label=label,is_teacher=teacher,DEV_FEM_U=field['case_mean_u_relative_l2'],DEV_FEM_S=field['case_mean_stress_relative_l2'],DEV_FEM_hot=field['case_mean_stress_hotspot_relative_l2'],
            DEV_teacher_node_latent=0. if teacher else row['full60_to_teacher_node_latent']['case_mean_relative_L2'],
            marginal_training_seconds=0. if teacher else row['terminal_campaign_training_epoch_seconds'],
            full60_to_teacher_physical=None if teacher else row['full60_to_teacher_physical']))
    metrics=dict(status='completed',roles=rows,all17_physical_candidates=s['candidates'],paired_CAD20=s['paired_CAD20'],learning_diagnostics=s['learning_diagnostics'],
        fullTRAIN300_arms=t['methods'],teacher_fullTRAIN300_FEM=t['full_TRAIN_teacher_to_original_FEM'],teacher_inherited_cost=t['inherited_teacher_cost'],
        actual_new_TRAIN_campaign_seconds=te['elapsed_epoch_seconds'],actual_DEV_campaign_seconds=e['elapsed_epoch_seconds'],
        teacher_latent_cache_bytes=t['teacher_latent_cache_bytes'],teacher_physical_cache_host_bytes=t['teacher_physical_cache_host_bytes'],
        checkpoint_reads=s['checkpoint_reads'],new_predictions=180,knownDEV_GT_once=60,physicalTRAIN_GT_once=300,FEM_used_for_updates=False,
        scope='Fixed teacher operator approximation plus original FEM accuracy, inheritedteacher5/studentoperatorlatent5, knownDEV not blind.17physical rows exclude olderFNO10/40, no finalsolver selection',FINAL_reads=0,raw_GT_reads=0,scaler_fits=0)
    d.write(out/'metrics.json',metrics)
    text=['# 固定teacher中间场：TS能否学会该算子映射？','',
        '**同一TRAIN teacher的编码器／输出头冻结，只训练新的TS或FNO算子各5遍；这不是10遍FEM监督，也不是fresh5公平选型。**','',
        '下表均为20个knownDEV新CAD×三载荷、全原节点；teacher节点latent误差属于固定特征坐标，不是物理场误差。','',
        '| 方法 | teacher节点latent误差 | FEM位移误差 | FEM应力误差 | 新增纯训练分钟 |','|---|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['label']} | {100*r['DEV_teacher_node_latent']:.2f}% | {100*r['DEV_FEM_U']:.2f}% | {100*r['DEV_FEM_S']:.2f}% | {r['marginal_training_seconds']/60:.2f} |")
    text+=['','TRAIN300上的节点latent拟合（初始→5遍端点）：']
    for name,label in roles[1:]:
        diag=s['learning_diagnostics'][name]
        text.append(f"- {label}：{100*diag['initial_TRAIN_teacher_latent_relative_L2']:.2f}%→{100*diag['endpoint_TRAIN_teacher_latent_relative_L2']:.2f}%，固定诊断门槛{'通过' if diag['passed'] else '未通过'}。")
    text+=['',f"新增训练整轮{te['elapsed_epoch_seconds']/60:.2f}分钟；DEV整轮{e['elapsed_epoch_seconds']/60:.2f}分钟；teacher旧整轮{t['inherited_teacher_cost']['stage_epoch_seconds']/60:.2f}分钟／旧纯训练{t['inherited_teacher_cost']['training_epoch_seconds']:.2f}秒另列。表中teacher新增成本为0，指复用已有训练，绝非teacher免费。推理计时包含额外latent诊断，不与历史纯推理速度比较。",'',
        'teacher→学生的物理U/S误差另存metrics.json；teacher自身也有原FEM误差。学会teacher映射不意味着FEM达到高精度，更不等价新的离节点FEM验证。', '',
        '完整TRAIN300、首真实最大325387节点算子梯度通过；冻结encoder/head无梯度且hash不变，onlyoperator freshAdam1500步。CPU合成33337节点检查只检验完整链式反传，不替代真实GPU。共同teacher-R16节点latent目标按真实blockcenter三线性插值，完整16通道范数、equalcase，无epsilon／裁剪／节点采样；空网格体积不作为损失样本。', '',
        '180个teacher／学生全节点encoded4／physical4不可变有限预测与完整latent标量先保存并重验，再共享knownDEV60 FEM标签一次；14旧候选仅复用物理标量。原物理TRAIN300缓存仍实际读取一次用于FEM读出，不用来更新学生；teacher目标不冒充FEM。原signed U(mm)/S(MPa)、GRID对齐、旧TRAIN40 scaler保留。', '',
        'TRAIN学习门槛是端点后冻结的描述性判断，不冒称训练前注册，未用于筛掉模型或改变DEV分母。knownDEV已迭代、单seed／availabilityconditional来源；原FINAL已揭盲，本阶段不读取。TS1.882M、FNO16 4.244M、teacher同FNO架构及目标坐标均是局限，不能宣布TS架构上限。', '',
        '完整17行、每载荷20、配对20CAD、热点、成本／曝光／冻结参数及未定义teacher零范数见metrics.json；不将本诊断的最低误差称全部历史最好，不自动延长40遍或24000步。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    labels=['TS latent5','FNO16 latent5'];x=[0,1]
    diagnostics=[s['learning_diagnostics'][n] for n,_ in roles[1:]]
    axes[0].bar([i-.18 for i in x],[100*v['initial_TRAIN_teacher_latent_relative_L2'] for v in diagnostics],.36,label='Initial TRAIN')
    axes[0].bar([i+.18 for i in x],[100*v['endpoint_TRAIN_teacher_latent_relative_L2'] for v in diagnostics],.36,label='TRAIN after5')
    axes[0].set_xticks(x,labels);axes[0].set_ylabel('Mean teacher node-latent relative L2 (%)');axes[0].set_title('Operator approximation; frozen encoder/head');axes[0].legend()
    x=[0,1,2]
    axes[1].bar([i-.18 for i in x],[100*r['DEV_FEM_U'] for r in rows],.36,label='FEM U')
    axes[1].bar([i+.18 for i in x],[100*r['DEV_FEM_S'] for r in rows],.36,label='FEM signed S')
    axes[1].set_xticks(x,['Teacher5','TS latent5','FNO16 latent5']);axes[1].set_ylabel('Mean original-FEM relative L2 (%)');axes[1].set_title('KnownDEV60; teacher error remains');axes[1].legend()
    fig.suptitle('Original DeepJEB / teacher5 state + fresh operator5 latent-target updates')
    fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    assert (out/'comparison.png').stat().st_size>10000
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha('tools/export_node_latent_teacher_v37.py'),
        summary_identity=d.identity(d.path(p['output'])/'summary.json',hashed=True),external_identity=d.identity(d.path(p['output'])/'external_terminal.json',hashed=True),
        actual_CPU_seconds=time.monotonic()-began,numerical_array_reads=0,PT_body_reads=0,GPU_calls=0,actual_visual_accepted=False,final_solver_selected=False,
        files={f:d.identity(out/f,hashed=True) for f in ['BRIEF.md','metrics.json','comparison.png']}))
    print(dict(status='completed',actual_CPU_seconds=time.monotonic()-began,report=d.relative(out/'BRIEF.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
