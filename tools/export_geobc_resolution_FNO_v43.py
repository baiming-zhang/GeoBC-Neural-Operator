"""One scalar-only report for the fixed R128 FNO9 physical5 candidate."""
import argparse
import time

OWN='tools/export_geobc_resolution_FNO_v43.py'
PLAN='notes/geobc_resolution_FNO_readout_v43_plan.json'
REVIEW='notes/geobc_resolution_FNO_v43_exporter_root_review.json'


def run():
    began=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.resolution_readout_contract_v43 import approved_contract,check_identity,complete_physical
    marker=d.read(REVIEW)
    assert marker['passed'] and marker['conditional_actual_export_released'] and marker['source_sha256']==d.sha(OWN)
    p,_,_,_,qualified=approved_contract(PLAN);s=p['scientific'];assert marker['scientific_digest']==p['scientific_digest']
    f=d.path(p['output']);a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json') and a['status']=='completed' and a['protocol_qualified'] and a['closed'] and not a['active']
    assert e['closed'] and not e['active'] and e['exit_code']==0 and e['within_external_budget'] and not e['external_timeout']
    assert all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest'] and e['source_pins']==p['source_pins']
    assert a['committed_finite_predictions']==a['owned_GT_array_reads']==a['owned_GT_read_attempts']==60 and a['prediction_barrier_complete']
    assert a['raw_GT_reads']==a['FINAL_numeric_reads']==a['training_updates']==a['scaler_fits']==0
    assert max(a['elapsed_epoch_seconds'],a['elapsed_monotonic_seconds'],e['elapsed_epoch_seconds'],e['elapsed_monotonic_seconds'])<=1800
    for row in a['candidates'].values():complete_physical(row['full60'],s['case_keys'])
    fields=('case_mean_u_relative_l2','case_mean_stress_relative_l2','case_mean_stress_hotspot_relative_l2')
    names=('FNO100_PHYSICAL_1500','R128_FNO9_1500','R64_FNO_W16_1500')
    labels=('FNO width9 R64','FNO width9 R128','FNO width16 R64')
    points=[]
    for name,label in zip(names,labels):
        row=a['candidates'][name];up=d.read(check_identity(row['upstream_terminal_bindings']['summary']))
        arm=up if name=='FNO100_PHYSICAL_1500' else up['methods'][row['model_profile']['method']];tm=arm['endpoints']['1500']['full_TRAIN']
        complete_physical(tm,qualified[names[1]]['plan']['scientific']['TRAIN_keys'])
        points.append(dict(name=name,label=label,TRAIN={k:tm[k] for k in fields},DEV={k:row['full60'][k] for k in fields},
            training_epoch_seconds=arm['training_epoch_seconds'],training_monotonic_seconds=arm['training_monotonic_seconds'],
            inference_epoch_seconds=row['inference_epoch_seconds'],inference_monotonic_seconds=row['inference_seconds'],model_profile=arm['model_profile']))
    original,new,fno=points
    differences={role:{k:new['DEV'][k]-reference['DEV'][k] for k in fields} for role,reference in [('same_FNO9_R64',original),('wider_FNO16_R64',fno)]}
    up=qualified[names[1]];train=up['summary'];arm=up['arm']
    out=d.path('results/geobc_resolution_FNO_v43');assert not out.exists(),'No report replay';out.mkdir(parents=True)
    lines=['# FNO空间分辨率：同参数与初值／5遍物理训练','',
        '**原版DeepJEB，100训练CAD×三载荷／1500步，knownDEV20个新CAD×三载荷，全部原节点。**','',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |','|---|---:|---:|---:|---:|']
    for point in points:
        v=point['DEV'];lines.append(f"| {point['label']} | {v[fields[0]]*100:.2f}% | {v[fields[1]]*100:.2f}% | {v[fields[2]]*100:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    for role,label in [('same_FNO9_R64','相同参数FNO9 R64'),('wider_FNO16_R64','更大容量FNO16 R64')]:
        v=differences[role];lines+=['',f"R128减{label}：位移{v[fields[0]]*100:+.2f}、应力{v[fields[1]]*100:+.2f}、热点{v[fields[2]]*100:+.2f}个百分点。负数表示改善。"]
    ts=a['candidates']['R16_TRANSOLVER_1500']['full60']
    lines+=['',f"原版R16 Transolver同5遍背景：位移{ts[fields[0]]*100:.2f}%／应力{ts[fields[1]]*100:.2f}%。其表示与参数不同，不能作分辨率控制。",'',
        f"R128 TRAIN位移／应力{new['TRAIN'][fields[0]]*100:.2f}%／{new['TRAIN'][fields[1]]*100:.2f}%；整轮训练{up['external']['elapsed_epoch_seconds']/60:.2f}分钟，DEV读出{e['elapsed_epoch_seconds']:.2f}秒。纯训练和缓存／栅格重建／初始化／保存／退出成本分列。",'',
        'R64→R128保持FNO width9／modes12／4层、4526858参数、全部初始权重、encoder／查询头、loss／Adam／1500病例顺序一致。只改变从原节点输入重新栅格化及查询分辨率，不把旧网格上采样、不池化目标。较宽R64 FNO16有14.207M参数，单列容量背景。', '',
        '原signed物理U／S相对平方损失、seed2026、TRAIN40-only scaler；无teacher权重／目标／优化器继承。CSV按GRID对齐H5 vertices，不抽样监督节点。一个真实最大病例的R64栅格重建与既有输入一致；这项检查没有额外目标读取，也不伪称逐一重验了全部300个旧网格。', '',
        '原TRAIN300物理标签缓存一次；只加载native输入，R64 reference三份输入仅最大病例一次。实际全N encoder／operator／head及jointleaf正有限梯度，显存30.5GB资格；动态R128栅格／FFT耗时计入训练。', '',
        '新60份finite完整原节点预测提交并重验后才共享knownDEV GT60一次；三对照只复用标量。knownDEV已用于迭代，不是盲测；FINAL／scalerfit／重新训练0。完整每载荷20、配对20CAD、最坏／热点与TRAIN300见metrics.json。', '',
        '只有联合与应力误差都比同参数R64改善至少3个百分点、任一场不恶化超过2个百分点，才取得继续探索优先级；不是统计显著性或最终选型，不自动追加10／40遍／24000步。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(points=points,full_DEV_candidates=a['candidates'],full_TRAIN_candidate=train['methods'],
        differences=differences,paired_CAD20=a['paired_CAD20'],DEV_gates=a['DEV_gates'],comparison_scope=a['comparison_scope'],
        cache_receipt=train['cache_receipt'],training_campaign=up['external'],readout_campaign=e,
        peak_memory={k:train[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled')}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),sharey=True)
    for axis,split in zip(axes,('TRAIN','DEV')):
        for i,point in enumerate(points):
            axis.bar([x+(i-1)*.25 for x in range(2)],[100*point[split][k] for k in fields[:2]],width=.24,
                color=('#c88320','#37a28c','#245d9c')[i],label=point['label'])
        axis.set_xticks([0,1],['Displacement','Signed stress']);axis.set_title('TRAIN100 x 3' if split=='TRAIN' else 'Known DEV20 x 3');axis.legend(fontsize=8)
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Fresh physical 5 passes / identical FNO9 weights and parameters / R64 vs R128')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    assert all((out/name).is_file() for name in ('BRIEF.md','metrics.json','comparison.png'))
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),
        summary_identity=d.identity(f/'summary.json',hashed=True),external_identity=d.identity(f/'external_terminal.json',hashed=True),
        actual_CPU_seconds=time.monotonic()-began,numerical_array_reads=0,PT_body_reads=0,GPU_calls=0,actual_visual_accepted=False))
    print({'status':'completed','report':d.relative(out),'CPU_seconds':time.monotonic()-began})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    if args.export:run()
    else:print('{"status":"source_only","numerical_array_reads":0,"PT_body_reads":0,"GPU_calls":0}')
