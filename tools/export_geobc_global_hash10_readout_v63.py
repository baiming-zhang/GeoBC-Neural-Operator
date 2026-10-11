"""KnownDEV scalar comparison; no PT, arrays or inference."""
import argparse
import csv
import time

OWN='tools/export_geobc_global_hash10_readout_v63.py'
REVIEW='notes/geobc_global_hash10_readout_v63_exporter_root_review.json'
PLAN='notes/geobc_global_hash10_readout_v63_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash10_readout_contract_v63 import terminal,training_binding,complete_physical,METHOD
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
    rows=a['candidates'];assert set(rows)=={METHOD,s['Hash5_name'],'FNO100_PHYSICAL_3000','FNO100_PHYSICAL_6000'}
    for name,row in rows.items():
        complete_physical(row['full60'],keys)
        for load in ('ver','hor','tor'):complete_physical(row['per_load20'][load],[k for k in keys if k.endswith('/'+load)])
        if name!=METHOD:assert row['reused_immutable_scalar_metrics'] and not row['inference_replayed']
    fu='case_mean_u_relative_l2';fs='case_mean_stress_relative_l2';fh='case_mean_stress_hotspot_relative_l2'
    new=rows[METHOD]['full60'];old=rows[s['Hash5_name']]['full60']
    du=new[fu]-old[fu];ds=new[fs]-old[fs];dh=new[fh]-old[fh];dp=.5*(du+ds)
    passed=dp<=-.03 and max(du,ds)<=.02 and dh<=.02
    assert passed==a['DEV_gate']['passed']
    train=d.read('results/geobc_fno_learning20_v56/metrics.json')
    costs={METHOD:up['complete0to10_cost']['training_epoch']/60,
        'FNO100_PHYSICAL_3000':train['points'][1]['cumulative_cost']['training_epoch']/60,
        'FNO100_PHYSICAL_6000':train['complete0to20_cost']['training_epoch']/60}
    names=[METHOD,'FNO100_PHYSICAL_3000','FNO100_PHYSICAL_6000']
    labels={METHOD:'条件Hash 10遍','FNO100_PHYSICAL_3000':'FNO 10遍','FNO100_PHYSICAL_6000':'FNO 20遍'}
    scores={name:row['full60']['selection_score']for name,row in rows.items()}
    leaders=sorted(n for n,v in scores.items()if v==min(scores.values()))
    assert leaders==a['eligible_best']
    out=d.path('results/geobc_global_hash10_readout_v63');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    text=['# 条件Hash：100CAD训练收益能否迁移','',
        '**同100训练CAD、相同20个新CAD×三载荷、全原FEM节点、物理位移与signed应力。knownDEV已用于迭代，不是最终盲测。**','',
        '| 方法 | 新CAD位移误差 | 新CAD应力误差 | 新CAD热点误差 | 累计纯训练分钟 |',
        '|---|---:|---:|---:|---:|']
    for n in names:
        m=rows[n]['full60'];text.append(f'| {labels[n]} | {100*m[fu]:.2f}% | {100*m[fs]:.2f}% | {100*m[fh]:.2f}% | {costs[n]:.2f} |')
    text+=['',f'Hash10减自身5：位移{100*du:+.2f}、应力{100*ds:+.2f}、热点{100*dh:+.2f}个百分点。自身推进门槛{"通过" if passed else "未通过"}；不自动加长训练。','',
        f'固定U/S均值优先级当前领先：{", ".join(leaders)}；这是候选优先级，最终solver仍未定。容量、精度和20遍曝光成本不同，不能当架构上限或严格等成本胜负。','',
        f'本次读出完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，实际60场推理{rows[METHOD]["inference_epoch_seconds"]:.2f}秒。60份全节点预测先保存并全部重新验证，再读已有物理GT各一次。FNO与Hash5只复用旧标量指标，不新推理或再读真值。','',
        'Hash10训练集U/S/热点为20.90%/39.32%/28.29%，共享拟合仍不足。3场拟合<1%说明局部能力，不等于100CAD收敛或新几何迁移。原5遍技术失败、恢复与等待成本单独保留，表中纯训练分钟不代表完整项目耗时。','',
        '原版DeepJEB GRID对齐、signed U(mm)/S(MPa)，scaler仅TRAIN；单seed、可用来源条件选择、knownDEV迭代使用；FINAL／原CSV读取／新增训练／scaler拟合均0，不声称独立离节点FEM准确率。所有每载荷20、配对CAD20、失败历史与完整分母保留在JSON/CSV。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['method','case','nodes','U_relative_L2','S_relative_L2','hotspot_relative_L2'])
        for name,row in rows.items():
            for key in keys:
                z=row['full60']['per_case'][key];writer.writerow([name,key,z['nodes'],z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2']])
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,DEV_gate=a['DEV_gate'],
        training_minutes=costs,priorities=leaders,final_solver_selected=False,independent_blind_test=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.8))
    for i,(name,label)in enumerate(zip(names,['Hash 10','FNO 10','FNO 20'])):
        axes[0].bar([x+(i-1)*.24 for x in range(3)],[100*rows[name]['full60'][f]for f in [fu,fs,fh]],width=.24,label=label)
    axes[0].set_xticks([0,1,2],['U','S','Hotspot']);axes[0].set_ylabel('knownDEV60 mean relative L2 (%)');axes[0].legend()
    for field,label in [(fu,'U'),(fs,'S'),(fh,'Hotspot')]:
        axes[1].plot([5,10],[100*old[field],100*new[field]],'-o',label=label)
    axes[1].set_xticks([5,10]);axes[1].set_xlabel('Hash complete TRAIN case exposures');axes[1].set_ylabel('knownDEV60 mean relative L2 (%)');axes[1].legend();axes[1].grid(alpha=.2)
    fig.suptitle('Shared physical knownDEV60 / single seed, iterated development pool');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','comparison.png'];assert all((out/f).is_file()for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        actual_CPU_seconds=time.perf_counter()-began,DEV_gate_passed=passed,files=files,PT_body_reads=0,new_GT_array_reads=0,GPU_calls=0))
    print(dict(status='completed',DEV_gate_passed=passed,leaders=leaders,CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
