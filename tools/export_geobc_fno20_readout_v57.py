"""Scalar-only actual closed readout report; no PT, arrays or inference."""
import argparse
import time

OWN='tools/export_geobc_fno20_readout_v57.py'
REVIEW='notes/geobc_fno20_readout_v57_exporter_root_review.json'
PLAN='notes/geobc_fno20_readout_v57_plan.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.fno20_readout_contract_v57 import terminal,training_binding,complete_physical,METHOD
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);up=training_binding();keys=p['scientific']['case_keys']
    assert marker['SCI']==p['scientific_digest'] and a['protocol_qualified']
    assert a['committed_finite_predictions']==a['owned_GT_read_attempts']==a['owned_GT_array_reads']==60
    assert a['checkpoint_read_attempts']==a['checkpoint_reads']==a['checkpoint_body_hash_reads']==1
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    assert a['prediction_barrier_complete'] and not a['final_solver_selected']
    old=a['candidates']['FNO100_PHYSICAL_3000'];new=a['candidates'][METHOD]
    for row in (old,new):complete_physical(row['full60'],keys)
    train=d.read('results/geobc_fno_learning20_v56/metrics.json')
    oldcost=train['points'][1]['cumulative_cost'];newcost=new['complete0to20_cost']
    fields=['case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2']
    differences={k:new['full60'][k]-old['full60'][k] for k in fields}
    passed=a['DEV_gate']['passed']
    out=d.path('results/geobc_fno20_readout_v57');assert not out.exists(),'One immutable actual report'
    out.mkdir(parents=True)
    text=['# FNO：20遍的收益能否迁移到新CAD','',
        '**同100训练CAD×三载荷、同FNO轨迹；knownDEV20个新CAD×三载荷、全部原节点、signed物理单位。**','',
        '| FNO总曝光遍数 | 新CAD位移误差 | 新CAD应力误差 | 新CAD热点误差 | 累计纯训练分钟 |',
        '|---|---:|---:|---:|---:|']
    for count,row,cost in [(10,old,oldcost),(20,new,newcost)]:
        m=row['full60'];text.append(f'| {count} | {100*m[fields[0]]:.2f}% | {100*m[fields[1]]:.2f}% | {100*m[fields[2]]:.2f}% | {cost["training_epoch"]/60:.2f} |')
    decision='验证推进门槛通过，当前这条FNO轨迹值得保留' if passed else '验证推进门槛未通过，不自动继续延长'
    text+=['',f'20减10：位移{100*differences[fields[0]]:+.2f}、应力{100*differences[fields[1]]:+.2f}、热点{100*differences[fields[2]]:+.2f}个百分点。{decision}。','',
        f'本次验证整轮{e["elapsed_epoch_seconds"]:.2f}秒；60份完整预测先保存并复核，再共享读取60份已有物理GT各一次。旧10遍仅复用不可变标量指标；新增训练／FINAL／原CSV读取／scaler拟合均0。','',
        'TRAIN20位移／应力为15.75%／35.74%；训练拟合仍是瓶颈。原slides单场约0.08%是已知场拟合，50场共享平均15.11%，网格误差与原有限元节点误差分别报告。','',
        'knownDEV已用于迭代、单seed、可用来源条件选择；不是独立最终盲测。增加曝光与成本，不构成等预算架构比较或最终求解器结论。三段训练成本由真实0→5、5→10、10→20收据合并，保留epoch／monotonic及所有准备成本。']
    (out/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,
        DEV_gate=a['DEV_gate'],differences=differences,complete0to10_cost=oldcost,complete0to20_cost=newcost,
        final_solver_selected=False,independent_blind_test=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.4),sharey=True)
    for ax,field,title in zip(axes,fields[:2],['Displacement','Signed stress']):
        ax.plot([10,20],[100*train['points'][i]['full_TRAIN'][field]for i in (1,2)],'-o',label='TRAIN300')
        ax.plot([10,20],[100*row['full60'][field]for row in (old,new)],'-o',label='knownDEV60')
        ax.set_title(title);ax.set_xticks([10,20]);ax.set_xlabel('Complete case exposures');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean relative L2 (%)');axes[1].legend();fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/k).is_file()for k in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),
        terminal_binding=b,actual_CPU_seconds=time.perf_counter()-began,DEV_gate_passed=passed,
        files=['BRIEF.md','metrics.json','comparison.png'],new_GT_array_reads=0,PT_body_reads=0,GPU_calls=0))
    print(dict(status='completed',DEV_gate_passed=passed,CPU_seconds=time.perf_counter()-began))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
