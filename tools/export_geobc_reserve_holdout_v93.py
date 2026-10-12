"""CPU scalar report after exact successful readout exit; never GT/PT/arrays."""
import argparse
import csv
import time
from geobc import available_cad_pilot_data as d


def export():
    began=time.monotonic();p=d.read('notes/geobc_reserve_holdout_v93_plan.json');out=d.path(p['output'])
    review=d.read('notes/geobc_reserve_holdout_v93_exporter_root_review.json')
    assert review['passed'] and review['conditional_actual_export_released']
    assert d.sha('tools/export_geobc_reserve_holdout_v93.py')==review['source_sha256']
    r=d.read(out/'summary.json');e=d.read(out/'external_terminal.json')
    assert r['status']=='completed' and r['closed'] and not r['active'] and r['protocol_qualified']
    assert e['exit_code']==0 and e['within_external_budget'] and e['closed'] and not e['active']
    assert d.exact_process_exited(e['child_process']) and d.exact_process_exited(e['guard_process'])
    assert e['scientific_digest']==p['scientific_digest']==d.digest(p['scientific'])
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert r['committed_finite_predictions']==18 and r['prediction_barrier_complete']
    assert r['source_GT_parse_attempts']==r['source_GT_parses']==2 and r['physical_case_reads']==6
    assert r['checkpoint_loads']==r['checkpoint_hash_reads']==3 and r['training_updates']==r['scaler_fits']==0
    dest=d.path('results/geobc_reserve_holdout_v93');dest.mkdir(exist_ok=False)
    rows=[];names=['FNO20','FNO_HASH20','TS20']
    for name in names:
        x=r['candidates'][name];assert x['strict_checkpoint_verified'] and x['cases']==6 and x['full6']['all_cases_complete']
        for key,m in x['full6']['per_case'].items():rows.append(dict(candidate=name,key=key,**m))
    with (dest/'per_case.csv').open('w',encoding='utf-8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    d.write(dest/'metrics.json',dict(SCI=p['scientific_digest'],candidates=r['candidates'],whole_epoch_seconds=e['elapsed_epoch_seconds'],
        whole_monotonic_seconds=e['elapsed_monotonic_seconds'],source_GT_parses=2,physical_cases=6,immutable_predictions=18,
        limits=p['scientific']['interpretation_limits'],final_solver_selected=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5));labels=['FNO','FNO+Hash','Transolver'];width=.24
    for i,(metric,label)in enumerate([('u','Displacement'),('stress','Signed stress'),('stress_hotspot','Stress hotspot')]):
        values=[100*r['candidates'][n]['full6']['case_mean_'+metric+'_relative_l2']for n in names]
        bars=ax.bar([x+(i-1)*width for x in range(3)],values,width,label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=8)
    ax.set_xticks(range(3),labels);ax.set_ylabel('Case-mean relative L2 (%)');ax.set_title('Previously unused2 CADs /3 loads /frozen20-pass endpoints')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=170);plt.close(fig)
    text=['# 两个预留 CAD：冻结模型的新几何检查','','固定两CAD×三载荷、同20遍／seed2026。全部18份预测先提交并验哈希和有限性，才共享读取两CSV的原signed位移与应力。无新训练、scaler拟合或参数选择。','','| 模型 | 位移误差 | 应力误差 | 热点误差 |','|---|---:|---:|---:|']
    for name,label in zip(names,labels):
        m=r['candidates'][name]['full6'];text.append('| '+label+' | '+' | '.join(f'{100*m["case_mean_"+k+"_relative_l2"]:.2f}%'for k in ['u','stress','stress_hotspot'])+' |')
    text+=['',f'整阶段实测{e["elapsed_epoch_seconds"]:.2f}秒，包含两CAD输入准备、三个模型加载、预测持久化与读出。纯推理时间和参数量、每载荷两例、完整18例指标在metrics.json／CSV中。','','仅两CAD，且来源按本地可用性筛选；这是新几何检查，不能估计总体泛化，也不能确定最终赢家。三个模型容量、精度及续训历史不同。旧开发集和已揭盲FINAL均未用于本轮；这些预留CAD在读出后已揭盲，不再可作新盲测。','','本轮固定端点关闭，不根据这六场调参或重复推理。']
    (dest/'BRIEF.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for f in ['comparison.png','metrics.json','per_case.csv','BRIEF.md']:assert (dest/f).stat().st_size>0
    d.write(dest/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],CPU_export_seconds=time.monotonic()-began,
        terminal_binding={k:d.identity(out/file,hashed=True)for k,file in [('summary','summary.json'),('external','external_terminal.json')]},
        files=[d.identity(dest/f,hashed=True)for f in ['comparison.png','metrics.json','per_case.csv','BRIEF.md']],new_GT=0,new_PT=0,new_arrays=0,GPU=0,PNG_visual_review_pending=True))
    print(dict(status='completed',CPU_export_seconds=time.monotonic()-began,output=d.relative(dest)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","GT":0,"GPU":0}')
