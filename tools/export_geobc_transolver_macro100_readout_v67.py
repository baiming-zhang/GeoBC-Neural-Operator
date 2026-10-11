"""Concise actual knownDEV comparison, complete scalar denominators and no new I/O bodies."""
import argparse
import csv
import time

OWN='tools/export_geobc_transolver_macro100_readout_v67.py'
PLAN='notes/geobc_transolver_macro100_readout_v67_plan.json'
REVIEW='notes/geobc_transolver_macro100_readout_v67_exporter_root_review.json'
FIELDS=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_macro100_readout_contract_v67 import terminal,training_binding,complete_physical,METHOD
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);up=training_binding();s=p['scientific'];keys=s['case_keys']
    assert marker['SCI']==p['scientific_digest'] and a['protocol_qualified'] and a['prediction_barrier_complete']
    assert a['committed_finite_predictions']==a['owned_GT_read_attempts']==a['owned_GT_array_reads']==60
    assert a['checkpoint_read_attempts']==a['checkpoint_reads']==a['checkpoint_body_hash_reads']==1
    assert a['input_array_read_attempts']==a['input_array_reads']
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    barrier=d.read(d.path(p['output'])/'prediction_barrier.json')
    assert barrier['status']=='completed' and barrier['finite_predictions']==60 and barrier['owned_GT_reads_before_barrier']==0
    assert not a['final_solver_selected'] and not a['automatic_continuation']
    rows=a['candidates'];assert set(rows)=={METHOD}|set(up['references']) and len(rows)==8
    for name,row in rows.items():
        complete_physical(row['full60'],keys)
        for load in ('ver','hor','tor'):complete_physical(row['per_load20'][load],[k for k in keys if k.endswith('/'+load)])
        if name!=METHOD:
            assert row['reused_immutable_scalar_metrics'] and not row['inference_replayed'] and row['full60']==up['references'][name]['full60']
    assert rows[METHOD]['loaded_typed_model_state_digest_verified'] and rows[METHOD]['cases']==60
    scores={n:r['full60']['selection_score']for n,r in rows.items()}
    leaders=sorted(n for n,v in scores.items()if v==min(scores.values()));assert leaders==a['eligible_best']
    tsnames=[n for n,r in rows.items()if r['family']in ('TS_ORIGINAL_BF16_GLOBAL_HASH','R16_transolver')]
    assert len(tsnames)==3
    tsleaders=sorted(n for n in tsnames if scores[n]==min(scores[k]for k in tsnames))
    new=rows[METHOD]['full60'];old=rows[s['same5_reference_name']]['full60'];diff={f:100*(new[f]-old[f])for f in FIELDS}
    du=diff[FIELDS[0]]/100;ds=diff[FIELDS[1]]/100;dh=diff[FIELDS[2]]/100
    passed=.5*(du+ds)<=-.03 and max(du,ds)<=.02 and dh<=.02;assert passed==a['DEV_gate']['passed']
    labels={METHOD:'原版TS＋连续解码器 5遍','FNO100_PHYSICAL_6000':'FNO 20遍','FNO100_PHYSICAL_3000':'FNO 10遍',
        'FNO100_PHYSICAL_1500':'FNO 5遍','GLOBALHASH100_PHYSICAL_1500':'FNO条件Hash 5遍','GLOBALHASH100_PHYSICAL_3000':'FNO条件Hash 10遍',
        'R16_TRANSOLVER_1500':'R16原版TS 5遍','R16_TRANSOLVER_PHYSICAL_3000':'R16原版TS 10遍'}
    shown=[];roles={}
    for role,name in [('本组综合领先',leaders[0]),('本组原版TS领先',tsleaders[0]),('本轮方案',METHOD)]:
        if name not in shown:shown.append(name);roles[name]=[]
        roles[name].append(role)
    out=d.path('results/geobc_transolver_macro100_readout_v67');assert not out.exists(),'One immutable actual report';out.mkdir(parents=True)
    text=['# 原版TS与FNO：相同20个新CAD的物理比较','',
        '**相同100TRAIN划分、20knownDEV CAD×三载荷、全原FEM节点；优先级为位移与应力相对L2均值。knownDEV已用于迭代，不是最终盲测。**','',
        '| 角色／方案 | 位移误差 | 应力误差 | 热点误差 |','|---|---:|---:|---:|']
    for name in shown:
        z=rows[name]['full60'];text.append('| '+'／'.join(roles[name])+'：'+labels[name]+' | '+' | '.join(f'{100*z[f]:.2f}%'for f in FIELDS)+' |')
    train=up['summary'];tr=train['models'][up['family']]['endpoints']['5']
    text+=['',f'本轮TS减同5遍FNO条件Hash：位移{diff[FIELDS[0]]:+.2f}、应力{diff[FIELDS[1]]:+.2f}、热点{diff[FIELDS[2]]:+.2f}个百分点；推进门槛{"通过" if passed else "未通过"}，不自动加长训练。','',
        f'本轮TS全TRAIN300位移{100*tr[FIELDS[0]]:.2f}%／应力{100*tr[FIELDS[1]]:.2f}%；欠拟合不能全部归咎新CAD。宏观训练完整阶段{up["external"]["elapsed_epoch_seconds"]:.2f}秒、学习区间{train["learning_interval_epoch_seconds"]:.2f}秒（含资格、0/1/5读出及保存）。当前验证完整{e["elapsed_epoch_seconds"]:.2f}秒，60场实际推理{rows[METHOD]["inference_epoch_seconds"]:.2f}秒；与旧纯训练分钟不可直接混算成本比。','',
        '本组原版TS候选仅本轮连续解码器5遍与已有R16直接TS5/10遍；这不是所有Transolver版本的最终排名。完整八方案指标、每载荷20、配对CAD20、全部ties与成本范围在CSV/JSON/图中。表格只显示三个去重角色，完整分母和差结果保留。','',
        'TS三场共享拟合应力1.54%，同解码器FNO0.76%；小规模可拟合与大范围可迁移是不同能力。本轮TS输入为完整R64，BF16骨干、FP32参数/Adam和返回latent、TCNN FP16；旧直接TS是R16表示，FNO和Hash参数容量不同，10/20遍参考曝光也不同，不能宣布架构上限或严格等成本因果结论。','',
        '60份全节点编码／物理预测先保存、重新核验有限值和SHA，全部完成后GT各读一次。自己的检查点仅hash/load一次并核对typed模型字节；七个旧方案只复用不可变全节点标量，不重推理或再读真值。旧FP32失败、三场训练、缓存和基线失败成本另外保留。','',
        '原版DeepJEB CSV GRID对齐、signed U(mm)/S(MPa)、scaler仅TRAIN；单seed、可用来源条件选择、knownDEV迭代使用，FINAL／原CSV／训练更新／scaler拟合均0。没有独立离节点FEM精度或最终solver声明。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    with (out/'per_case.csv').open('w',encoding='utf8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['method','case','nodes','U_relative_L2','S_relative_L2','hotspot_relative_L2'])
        for name,row in rows.items():
            for key in keys:
                z=row['full60']['per_case'][key];writer.writerow([name,key,z['nodes'],z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,training_binding=up['binding'],reference_bindings=up['reference_bindings'],
        leaders=leaders,originalTS_group_leaders=tsleaders,all_scores=scores,same5_diff_pp=diff,DEV_gate=a['DEV_gate'],shown_role_rows=shown,final_solver_selected=False,independent_blind_test=False))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=sorted(rows,key=lambda n:(scores[n],n));fig,ax=plt.subplots(figsize=(11,5))
    for i,field in enumerate(FIELDS):ax.barh([j+(i-1)*.24 for j in range(len(names))],[100*rows[n]['full60'][field]for n in names],height=.24,label=['U','Signed S','Hotspot'][i])
    ax.set_yticks(range(len(names)),[labels[n]for n in names]);ax.invert_yaxis();ax.set_xlabel('knownDEV60 full-node physical relative L2 (%)');ax.legend();ax.grid(axis='x',alpha=.2)
    # Use ASCII labels for portable font availability; native brief remains Chinese.
    ascii_labels={METHOD:'TS BF16 + continuous / 5','FNO100_PHYSICAL_6000':'FNO / 20','FNO100_PHYSICAL_3000':'FNO / 10','FNO100_PHYSICAL_1500':'FNO / 5',
        'GLOBALHASH100_PHYSICAL_1500':'FNO + Hash / 5','GLOBALHASH100_PHYSICAL_3000':'FNO + Hash / 10','R16_TRANSOLVER_1500':'R16 TS / 5','R16_TRANSOLVER_PHYSICAL_3000':'R16 TS / 10'}
    ax.set_yticks(range(len(names)),[ascii_labels[n]for n in names]);ax.set_title('Same100 TRAIN / same20 knownDEV CADs x3 loads; representation and exposure differ')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','comparison.png'];assert all((out/f).is_file()for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,
        files=files,DEV_gate_passed=passed,PT_body_reads=0,new_GT_array_reads=0,GPU_calls=0))
    print(dict(status='completed',DEV_gate_passed=passed,leaders=leaders,originalTS_group_leaders=tsleaders,CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
