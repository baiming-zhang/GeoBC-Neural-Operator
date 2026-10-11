"""One scalar-only report for the fixed structured3D physical5 candidate."""
import argparse
import time

OWN='tools/export_geobc_structured_TS_v40.py'
PLAN='notes/geobc_structured_TS_readout_v40_plan.json'
REVIEW='notes/geobc_structured_TS_v40_exporter_root_review.json'


def run():
    began=time.monotonic()
    from geobc import available_cad_pilot_data as d
    from geobc.structured_readout_contract_v40 import approved_contract,check_identity,complete_physical
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
    names=('R16_TRANSOLVER_1500','R16_STRUCTURED_TRANSOLVER_1500','R16_FNO16_CONTROL_1500')
    labels=('Original TS R16','Structured3D TS R16','FNO width16 R16')
    points=[]
    for name,label in zip(names,labels):
        row=a['candidates'][name];up=d.read(check_identity(row['upstream_terminal_bindings']['summary']))
        arm=up['methods'][row['model_profile']['method']];tm=arm['endpoints']['1500']['full_TRAIN']
        complete_physical(tm,qualified[names[1]]['plan']['scientific']['TRAIN_keys'])
        points.append(dict(name=name,label=label,TRAIN={k:tm[k] for k in fields},DEV={k:row['full60'][k] for k in fields},
            training_epoch_seconds=arm['training_epoch_seconds'],training_monotonic_seconds=arm['training_monotonic_seconds'],
            inference_epoch_seconds=row['inference_epoch_seconds'],inference_monotonic_seconds=row['inference_seconds'],model_profile=arm['model_profile']))
    original,new,fno=points
    differences={role:{k:new['DEV'][k]-reference['DEV'][k] for k in fields} for role,reference in [('old_TS',original),('same_R16_FNO',fno)]}
    up=qualified[names[1]];train=up['summary'];arm=up['arm']
    out=d.path('results/geobc_structured_TS_v40');assert not out.exists(),'No report replay';out.mkdir(parents=True)
    lines=['# 结构网格Transolver：同5遍物理训练','',
        '**原版DeepJEB，100训练CAD×三载荷／1500步，knownDEV20个新CAD×三载荷，全部原节点。**','',
        '| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |','|---|---:|---:|---:|---:|']
    for point in points:
        v=point['DEV'];lines.append(f"| {point['label']} | {v[fields[0]]*100:.2f}% | {v[fields[1]]*100:.2f}% | {v[fields[2]]*100:.2f}% | {point['training_epoch_seconds']/60:.2f} |")
    for role,label in [('old_TS','原TS'),('same_R16_FNO','相同R16输入FNO')]:
        v=differences[role];lines+=['',f"新TS减{label}：位移{v[fields[0]]*100:+.2f}、应力{v[fields[1]]*100:+.2f}、热点{v[fields[2]]*100:+.2f}个百分点。负数表示改善。"]
    wider=a['candidates']['R64_FNO_W16_1500']['full60']
    lines+=['',f"较细R64网格的width16 FNO同5遍参考：位移{wider[fields[0]]*100:.2f}%／应力{wider[fields[1]]*100:.2f}%。网格表示不同，单列背景，不伪称同表示对照。",'',
        f"新TS TRAIN位移／应力{new['TRAIN'][fields[0]]*100:.2f}%／{new['TRAIN'][fields[1]]*100:.2f}%；整轮训练{up['external']['elapsed_epoch_seconds']/60:.2f}分钟，DEV读出{e['elapsed_epoch_seconds']:.2f}秒。纯训练和缓存／初始化／保存／退出成本分列。",'',
        '官方structured3D的Conv3d邻域投影与hidden128／4.777M参数共同改变；原TS1.882M、FNO16 4.244M。encoder／查询头初始相同，但不能把结果归因于单一邻域因素或宣称架构上限。', '',
        '全部encoder／operator／head从头按原signed物理U／S相对平方损失训练，1500步／5遍，原seed2026和TRAIN40-only scaler；无teacher权重、目标或优化器继承。原CSV按GRID对齐H5 vertices，不抽样节点。', '',
        '新60份finite完整原节点预测提交并重验后才共享knownDEV GT60一次；四个既有对照仅复用标量。knownDEV已用于迭代，不是盲测；FINAL／scalerfit／重新训练0。新候选完整全节点梯度与30.5GB资格来自实际GPU。', '',
        '推进规则同时检查相对原TS改善和相对相同R16 FNO的优先级；只是继续探索的依据，不自动追加40遍／24000步，也不宣布最终求解器。全五行、每载荷20、配对20CAD、最坏／热点、TRAIN300与预算见metrics.json。']
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
    axes[0].set_ylabel('Case mean relative L2 (%)');fig.suptitle('Fresh physical 5 passes / structured3D TS / joint neighborhood and capacity change')
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
