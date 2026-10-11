"""Scalar/loss-only report of the genuine fixed three-package TRAIN9 screen."""
import argparse
import csv
import time

OWN='tools/export_geobc_decoder_packages_v77.py'
REVIEW='notes/geobc_decoder_packages_v77_exporter_root_review.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.decoder_packages_contract_v77 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal,complete_physical
    r=d.read(REVIEW);p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['selected_TRAIN_keys']
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256']==d.sha(OWN) and r['SCI']==p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and a['matched_body_initial_bytes_verified']
    assert a['optimizer_step_attempts']==a['optimizer_step_completions']==2700
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts']==a['cache_receipt']['physical_TRAIN_target_reads']==9
    assert a['cache_receipt']['encoded_TRAIN_target_reads']==a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['parent_PT_reads']==0
    assert a['resident_physical_target_reuses']==81 and a['cache_receipt']['Gaussian_basis_bytes']==0
    assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    bodies=[]
    for arm in s['arms']:
        v=a['arms'][arm];bodies.append(v['initialization']['shared_body_digests'])
        assert v['status']=='completed' and v['step']==900 and v['case_exposure_counts']=={k:100 for k in keys}
        assert v['qualification']['passed'] and v['qualification']['query_check']['passed']
        assert max(v['arm_epoch_seconds'],v['arm_monotonic_seconds'])<=900
        assert set(v['endpoints'])=={'0','10','100'}
        for z in v['endpoints'].values():complete_physical(z['full_TRAIN9'],keys)
    assert all(v==bodies[0] for v in bodies)
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))<=30500000000
    folder=d.path('results/geobc_decoder_packages_v77');assert not (folder/'manifest.json').exists()
    folder.mkdir(parents=True,exist_ok=True)
    labels=['普通MLP','Fourier坐标MLP','既有TCNN']
    lines=['# 相同FNO骨干：三种解码器短比较','',
        '**固定3CAD×三载荷、每组900次全原节点更新（100遍），seed2026、同Adam／scaler／物理联合U/S损失，编码器和FNO的实际初始权重完全相同。**','',
        '| 解码器 | 位移误差 | 应力误差 | 热点误差 | 完整arm分钟 |',
        '|---|---:|---:|---:|---:|']
    for arm,label in zip(s['arms'],labels):
        v=a['arms'][arm];z=v['endpoints']['100']['full_TRAIN9']
        lines.append(f'| {label} | {100*z["case_mean_u_relative_l2"]:.2f}% | {100*z["case_mean_stress_relative_l2"]:.2f}% | {100*z["case_mean_stress_hotspot_relative_l2"]:.2f}% | {v["arm_epoch_seconds"]/60:.2f} |')
    lines+=['']
    for arm,label in zip(s['arms'][1:],labels[1:]):
        gate=a['TRAIN_gates'][arm]
        lines.append(f'{label}减普通MLP：位移{100*gate["U_difference"]:+.2f}、应力{100*gate["S_difference"]:+.2f}、热点{100*gate["hotspot_difference"]:+.2f}个百分点。预声明TRAIN推进门：{"通过" if gate["passed"] else "未通过"}。')
    counts=[a['arms'][arm]['initialization']['parameter_count'] for arm in s['arms']]
    lines+=['',f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，三组参数分别{counts[0]:,}／{counts[1]:,}／{counts[2]:,}。普通／Fourier为FP32，TCNN为官方原生FP16解码器及FP32 master，骨干FP32；**比较的是解码器包，不是等容量／同精度的单因素编码因果实验。**',
        '', '**这仍是训练场拟合，不是新CAD预测。** 通过TRAIN门只提高后续实验优先级，不自动扩大100CAD、不宣称最终solver或独立离网格FEM精度。短100遍负结果也不是收敛／架构上限。',
        '', 'Fourier仅在原xyz／native25／global32／latent16后追加固定36维sin/cos，频率1、2、4、8、16、32；没有频率搜索。TCNN复用既有12级配置，没有Hash超参数改动。',
        '', '所有三组在最大真实场全N反向传播通过，encoder／operator／head与context／latent梯度有限非零，资格检查未改变model／Adam／四类RNG。64个真实坐标query的梯度与chunk检查通过，native25在功能检查中固定；不是解析BC延拓或真值FEM检验。',
        '', 'TRAIN9物理标签各读一次，三端点×三组共81次常驻标签复用；编码标签／原CSV／DEV／FINAL／父PT读取及scaler拟合均0。每组原始成本、epoch／monotonic、全部9例与三端点保留；成本包含资格、训练、读出和保存，不冒充纯CUDA。']
    (folder/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(folder/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,TRAIN_gates=a['TRAIN_gates'],
        new_CAD_transfer_tested=False,larger_campaign_released=False,final_solver_selected=False))
    with (folder/'per_case.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file);writer.writerow(['arm','passes','case','U_relative_L2','S_relative_L2','hotspot_relative_L2','stress_RMSE_MPa','nodes'])
        for arm in s['arms']:
            for passes in (0,10,100):
                for key,z in a['arms'][arm]['endpoints'][str(passes)]['full_TRAIN9']['per_case'].items():
                    writer.writerow([arm,passes,key,z['u_relative_l2'],z['stress_relative_l2'],z['stress_hotspot_relative_l2'],z['stress_RMSE_MPa'],z['nodes']])
    import json
    loss=[json.loads(line) for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss)==2700 and all(sum(v['arm']==arm for v in loss)==900 for arm in s['arms'])
    d.write(folder/'loss.json',loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8))
    names=['Plain MLP','Fourier MLP','TCNN']
    for arm,label in zip(s['arms'],names):
        rows=[v for v in loss if v['arm']==arm]
        mean=[sum(v['physical_relative_squared'] for v in rows[k:k+9])/9 for k in range(0,900,9)]
        axes[0].plot(range(1,101),mean,label=label)
    axes[0].set_yscale('log');axes[0].set_xlabel('Complete TRAIN9 passes');axes[0].set_ylabel('Mean physical relative squared loss');axes[0].legend()
    for shift,arm,label in zip((-.25,0,.25),s['arms'],names):
        z=a['arms'][arm]['endpoints']['100']['full_TRAIN9']
        axes[1].bar([i+shift for i in range(3)],[100*z[k] for k in ('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')],width=.24,label=label)
    axes[1].set_xticks(range(3),['U','Signed S','Hotspot']);axes[1].set_ylabel('Case mean relative L2 (%)');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Shared FNO body / TRAIN9 fit only / decoder capacity and precision differ')
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','loss.json','comparison.png'];assert all((folder/f).is_file() for f in files)
    d.write(folder/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        files=files,actual_CPU_seconds=time.perf_counter()-began,new_GT_reads=0,PT_reads=0,GPU_calls=0,TRAIN_gates=a['TRAIN_gates']))
    print(dict(status='completed',actual_CPU_seconds=time.perf_counter()-began,TRAIN_gates=a['TRAIN_gates']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
