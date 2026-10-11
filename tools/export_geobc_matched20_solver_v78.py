"""Scalar-only TRAIN continuation report after genuine completed campaign."""
import argparse
import csv
import time

OWN='tools/export_geobc_matched20_solver_v78.py'
REVIEW='notes/geobc_matched20_solver_v78_exporter_root_review.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.matched20_solver_contract_v78 import PLAN
    from geobc.transolver_decoder_contract_v64 import terminal,complete_physical,check_identity
    p,a,e,b=terminal(PLAN);s=p['scientific'];keys=s['TRAIN_keys'];r=d.read(REVIEW)
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256']==d.sha(OWN) and r['SCI']==p['scientific_digest']
    assert a['training_complete'] and a['protocol_qualified'] and not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
    assert a['additional_optimizer_step_attempts']==a['additional_optimizer_step_completions']==7500
    assert a['parent_checkpoint_hash_attempts']==a['parent_checkpoint_hash_reads']==a['parent_checkpoint_load_attempts']==a['parent_checkpoint_loads']==2
    assert a['cache_receipt']['physical_TRAIN_target_read_attempts']==a['cache_receipt']['physical_TRAIN_target_reads']==300 and a['resident_physical_target_reuses']==902
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==a['cache_receipt']['encoded_TRAIN_target_reads']==0
    for arm in s['arms']:
        state=a['arms'][arm['name']]
        assert state['status']=='completed' and state['step']==6000 and state['case_exposure_counts']=={k:20 for k in keys}
        assert state['restored_model_Adam_fourRNG_cursor_exposure'] and state['qualification']['passed']
        for z in state['endpoints'].values():complete_physical(z['full_TRAIN'],keys)
    fp,fa,fe,fb=terminal(s['baseline_FNO20_plan_identity']['path']);assert fb==s['baseline_FNO20_terminal_binding']
    reference=fa['endpoints']['6000']['full_TRAIN'];complete_physical(reference,keys)
    parent_summaries={arm['name']:d.read(check_identity(a['arms'][arm['name']]['parent_native_cost_identity'])) for arm in s['arms']}
    out=d.path('results/geobc_matched20_solver_v78');assert not (out/'manifest.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    rows={name:z['endpoints']['6000']['full_TRAIN'] for name,z in a['arms'].items()};rows['FNO20_reference']=reference
    lines=['# 两种求解器补至20遍：TRAIN结果','',
        '**原100 TRAIN CAD×三载荷，所有方案每场20次完整原节点曝光、相同TRAIN-only scaler及物理联合U/S目标。Hash从10遍、原TS连续解码器从5遍恢复；模型、Adam、四类RNG和顺序前缀保留。**','',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 本轮新增训练区间分钟 |',
        '|---|---:|---:|---:|---:|']
    for name,label in [('FNO_GLOBAL_HASH20','条件Hash20'),('TS_CONTINUOUS20','原TS连续解码器20'),('FNO20_reference','已有FNO20')]:
        z=rows[name];cost=f'{a["arms"][name]["training_epoch_seconds"]/60:.2f}' if name in a['arms'] else '—'
        lines.append(f'| {label} | {100*z["case_mean_u_relative_l2"]:.2f}% | {100*z["case_mean_stress_relative_l2"]:.2f}% | {100*z["case_mean_stress_hotspot_relative_l2"]:.2f}% | {cost} |')
    lines+=['',f'本轮完整campaign{e["elapsed_epoch_seconds"]/60:.2f}分钟，包括共享一次缓存、两个父PT读取／校验、真实恢复资格、训练、TRAIN读出和保存。新增训练区间含中间读出／保存；终点读出另计。表中新增时间不能当各自总训练成本，原失败／父阶段成本身份与epoch／monotonic均在JSON保留。',
        '', '**这里只是TRAIN，不改变新CAD领先者。** 后续必须将两种20遍方案的全部预测提交后共享读验证标签；不因TRAIN好坏删候选。没有自动40遍、重放旧负结果、Hash小调或最终solver声明。',
        '', '恢复后最大真实场全原节点梯度通过，物理U/S/热点见证与父端点吻合，资格检查未改变model／Adam／RNG。各模型仍使用原有容量与FP32／BF16／TCNN精度；同曝光不是等容量、同成本或单因素架构因果比较。',
        '', '物理TRAIN300各读一次，常驻标签复用902次（含两恢复见证）；两个原父PT各hash/load一次。编码标签／原CSV／DEV／FINAL／scaler拟合均0。自己的最后检查点、曝光cursor与Adam状态留本地，不改变父文件。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,FNO20_reference=reference,FNO20_binding=fb,
        immediate_parent_summaries=parent_summaries,knownDEV_actual_completed=False,new_GT_reads=0,final_solver_selected=False))
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file);writer.writerow(['method','case','U_relative_L2','S_relative_L2','hotspot_relative_L2','nodes'])
        for name,z in rows.items():
            for key,value in z['per_case'].items():writer.writerow([name,key,value['u_relative_l2'],value['stress_relative_l2'],value['stress_hotspot_relative_l2'],value['nodes']])
    import json
    loss=[json.loads(line)for line in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss)==7500 and sum(z['arm']=='FNO_GLOBAL_HASH20'for z in loss)==3000 and sum(z['arm']=='TS_CONTINUOUS20'for z in loss)==4500
    d.write(out/'loss.json',loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for arm,label in zip(s['arms'],['Hash continuation','TS continuation']):
        values=[z for z in loss if z['arm']==arm['name']]
        means=[sum(v['physical_relative_squared']for v in values[i:i+300])/300 for i in range(0,len(values),300)]
        axes[0].plot(range(arm['parent_passes']+1,21),means,label=label)
    axes[0].set_xlabel('Total complete TRAIN300 passes');axes[0].set_ylabel('Mean physical relative squared loss');axes[0].set_yscale('log');axes[0].legend()
    for shift,name,label in zip((-.25,0,.25),rows,['Hash20','TS20','FNO20 reference']):
        z=rows[name];axes[1].bar([v+shift for v in range(3)],[100*z[k]for k in ('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')],width=.24,label=label)
    axes[1].set_xticks(range(3),['U','Signed S','Hotspot']);axes[1].set_ylabel('Case mean relative L2 (%)');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Matched20 TRAIN exposure; knownDEV predictions still required')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','loss.json','comparison.png'];assert all((out/z).is_file()for z in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,files=files,
        actual_CPU_seconds=time.perf_counter()-began,new_GT_reads=0,PT_reads=0,GPU_calls=0))
    print(dict(status='completed',actual_CPU_seconds=time.perf_counter()-began,knownDEV_actual=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
