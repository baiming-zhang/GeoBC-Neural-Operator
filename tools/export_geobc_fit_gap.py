"""Plot existing immutable scalar TRAIN/DEV errors; no inference or GT reads."""
import argparse
import time


def export():
    from geobc import available_cad_pilot_data as d
    start=time.monotonic();source='notes/geobc_current_physical5_TRAIN_DEV_gap.json'
    record=d.read(source)
    assert record['status']=='completed_scalar_diagnostic'
    for row in record['rows']:
        for key in ('training_summary_identity','DEV_summary_identity'):
            assert d.identity(row[key]['path'],hashed=True)==row[key]
    names=['R64_FNO_W16_1500','R16_TRANSOLVER_1500']
    rows=[next(r for r in record['rows'] if r['name']==name) for name in names]
    output=d.path('results/geobc_shared_fit_diagnostic')
    assert not (output/'manifest.json').exists(), 'Preserve prior report'
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.2,3.3))
    train=[r['TRAIN_CAD100_all300']['S_percent'] for r in rows]
    dev=[r['knownDEV_CAD20_all60']['S_percent'] for r in rows]
    for offset,values,color,label in [(-.18,train,'#527c9a','100 TRAIN CAD'),(.18,dev,'#dc9248','20 known DEV CAD')]:
        bars=ax.bar([x+offset for x in range(2)],values,width=.34,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3,fontsize=10)
    ax.set_xticks([0,1],['FNO width16','Transolver R16'])
    ax.set_ylabel('Stress relative L2 error (%)');ax.set_ylim(0,112)
    ax.set_title('Shared models still have high TRAIN error',fontsize=12)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,loc='upper left',fontsize=9)
    fig.tight_layout();fig.savefig(output/'stress_train_dev.png',dpi=160);plt.close(fig)
    table='| 方法／5遍 | 训练CAD应力误差 | 新CAD应力误差 |\n|---|---:|---:|\n'
    labels=['宽FNO','原版TS R16']
    table+='\n'.join(f'| {label} | {t:.2f}% | {v:.2f}% |' for label,t,v in zip(labels,train,dev))
    text=('# 当前共享拟合瓶颈\n\n'
        '**高误差不能只归因于新CAD迁移：共享模型在训练场上也没有拟合好。**\n\n'+table+'\n\n'
        '宽FNO的训练与新CAD应力差距为6.82个百分点；TS两边都约94%。这是当前短训练配置的观察，不能证明架构上限，也不能从两组均值孤立推断因果。\n\n'
        '下一步结合GINO原节点验证，区分共享训练拟合／表示瓶颈与迁移落差。未见明显收益的配置不追加长训练；单样本Hash小于1%仍只说明已知场可拟合。\n\n'
        '同100 TRAIN／20 knownDEV CAD、每CAD三载荷、5遍、全原节点physical U(mm)/signed S(MPa)、旧TRAIN40 scaler。仅复用已有指标；模型表示／容量／后端不同，没有新GT、模型推理或FINAL访问。\n\n'
        '![训练与新CAD应力误差](stress_train_dev.png)\n')
    (output/'BRIEF.md').write_text(text,encoding='utf-8')
    d.write(output/'metrics.json',record)
    manifest=dict(status='completed',source_identity=d.identity(source,hashed=True),
        files={name:d.identity(output/name,hashed=True) for name in ('BRIEF.md','metrics.json','stress_train_dev.png')},
        CPU_export_seconds=time.monotonic()-start,scientific_file_arrays_read=0,GT_array_reads=0,
        PT_body_reads=0,GPU_execution=0,inference_replayed=False)
    d.write(output/'manifest.json',manifest)
    return dict(status='completed',report=d.relative(output/'BRIEF.md'),seconds=manifest['CPU_export_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:
        import json
        print(json.dumps(export()))
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
