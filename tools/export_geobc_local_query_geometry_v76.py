"""One scalar/loss-only TRAIN9 report after genuine matched pair terminal."""
import argparse
import csv
import time

OWN='tools/export_geobc_local_query_geometry_v76.py'
REVIEW='notes/geobc_local_query_geometry_v76_exporter_root_review.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.local_query_geometry_contract_v76 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal,complete_physical
    p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['selected_TRAIN_keys']
    r=d.read(REVIEW)
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256']==d.sha(OWN) and r['SCI']==p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and a['matched_initial_bytes_verified']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==1800
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts']==a['cache_receipt']['physical_TRAIN_target_reads']==9
    assert a['cache_receipt']['encoded_TRAIN_target_reads']==a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['parent_PT_reads']==0
    assert a['resident_physical_target_reuses']==54 and a['cache_receipt']['Gaussian_basis_unique_CADs']==3
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    initials=[]
    for arm in s['arms']:
        v=a['arms'][arm];initials.append(v['initialization'])
        assert v['status']=='completed' and v['step']==900 and v['case_exposure_counts']=={k:100 for k in keys}
        assert v['qualification']['passed'] and v['qualification']['query_check']['passed']
        assert max(v['arm_epoch_seconds'],v['arm_monotonic_seconds'])<=900
        assert set(v['endpoints'])=={'0','10','100'}
        for endpoint in v['endpoints'].values():complete_physical(endpoint['full_TRAIN9'],keys)
    assert initials[0]==initials[1]
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    folder=d.path('results/geobc_local_query_geometry_v76');assert not (folder/'manifest.json').exists()
    folder.mkdir(parents=True,exist_ok=True)
    lines=['# 查询点局部几何：固定九场短对照','',
        '**同一FNO9、4535050参数、完全相同初始权重与Adam、固定3CAD×三载荷。只比较新增64维PointNet特征在查询点的用法：全局均值广播，或局部Gaussian聚合。**','',
        '| 100遍端点 | 位移误差 | 应力误差 | 热点误差 | 完整arm分钟 |',
        '|---|---:|---:|---:|---:|']
    for arm,label in zip(s['arms'],('全局均值对照','局部几何查询')):
        v=a['arms'][arm];z=v['endpoints']['100']['full_TRAIN9']
        lines.append(f'| {label} | {100*z["case_mean_u_relative_l2"]:.2f}% | {100*z["case_mean_stress_relative_l2"]:.2f}% | {100*z["case_mean_stress_hotspot_relative_l2"]:.2f}% | {v["arm_epoch_seconds"]/60:.2f} |')
    gate=a['TRAIN_gate'];passed=gate['passed']
    lines+=['',f'局部减全局：位移{100*gate["U_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点。预声明TRAIN推进门：{"通过" if passed else "未通过"}。',
        '',f'实际完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，包含原输入／物理标签一次加载、GPU静态几何权重、初始化、真实资格检查、训练、读出及检查点。两arm训练区间和资格成本在JSON分别保留，epoch／monotonic分别列，不按比例填未知成本。',
        '', '**这只回答固定训练场的表示／拟合能力，尚未测新CAD迁移；不自动扩大100CAD，不宣布最终solver或离网格FEM精度。** 单seed、同9场和900更新；目标仍是原signed U(mm)/S(MPa)，没有符号修正。',
        '', '最大的真实场完整反向传播通过，编码器／FNO／head及context／latent／tokens都有有限非零梯度。64个真实位置的query/chunk/点云排列／静态权重一致性与坐标梯度通过；native25在该功能检查中固定，不代表其解析延拓。',
        '', '局部权重为全部1024个有效几何点的Gaussian，sigma=.15，没有KNN／硬截断。静态原节点权重只由几何输入生成，三个CAD去重缓存并供各载荷复用；新查询点仍直接重算可微权重。FP32尾部可能下溢。',
        '', 'TRAIN9物理标签各读一次，两个arm的三端点共54次常驻标签复用；编码目标／原CSV／DEV／FINAL／父检查点读取和scaler拟合均0。所有9例、三端点及两成本轴在CSV／JSON；scaler仍为先前TRAIN40拟合。']
    (folder/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(folder/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,TRAIN_gate=gate,
        new_CAD_transfer_tested=False,final_solver_selected=False,larger_experiment_released=False))
    with (folder/'per_case.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file);writer.writerow(['arm','passes','case','U_relative_L2','S_relative_L2','hotspot_relative_L2','stress_RMSE_MPa','nodes'])
        for arm in s['arms']:
            for passes in (0,10,100):
                for key,z in a['arms'][arm]['endpoints'][str(passes)]['full_TRAIN9']['per_case'].items():
                    writer.writerow([arm,passes,key,z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2'],z['stress_RMSE_MPa'],z['nodes']])
    import json
    loss=[json.loads(line) for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss)==1800 and all(sum(z['arm']==arm for z in loss)==900 for arm in s['arms'])
    d.write(folder/'loss.json',loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    labels=['Pooled control','Local query']
    for arm,label in zip(s['arms'],labels):
        rows=[z for z in loss if z['arm']==arm]
        values=[sum(z['physical_relative_squared'] for z in rows[k:k+9])/9 for k in range(0,900,9)]
        axes[0].plot(range(1,101),values,label=label)
    axes[0].set_yscale('log');axes[0].set_xlabel('Complete TRAIN9 passes');axes[0].set_ylabel('Mean physical relative squared loss');axes[0].legend()
    for shift,arm,label in zip((-.18,.18),s['arms'],labels):
        z=a['arms'][arm]['endpoints']['100']['full_TRAIN9']
        axes[1].bar([i+shift for i in range(3)],[100*z[k] for k in ('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')],width=.36,label=label)
    axes[1].set_xticks(range(3),['U','Signed S','Hotspot']);axes[1].set_ylabel('Case mean relative L2 (%)');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Matched TRAIN9 fitting screen; unseen-CAD transfer not tested')
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','loss.json','comparison.png'];assert all((folder/f).is_file() for f in files)
    d.write(folder/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        files=files,actual_CPU_seconds=time.perf_counter()-began,new_GT_reads=0,PT_reads=0,GPU_calls=0,TRAIN_gate=gate))
    print(dict(status='completed',actual_CPU_seconds=time.perf_counter()-began,TRAIN_gate=gate))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
