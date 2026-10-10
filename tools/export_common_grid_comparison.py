"""CPU report from saved scalar endpoints; no arrays/checkpoints/GT/inference."""
import argparse
import csv
import io
import time
from geobc import available_cad_pilot_data as d

OWN='tools/export_common_grid_comparison.py'
REVIEW='notes/geobc_common_grid_exporter_root_review.json'


def run():
    started=time.monotonic();review=d.read(REVIEW)
    plan=d.read('notes/geobc_common_grid_readout_plan.json');parent=d.read(plan['scientific']['parent_plan'])
    assert review['passed'] and review['actual_export_released'] and review['exporter_source_sha256']==d.sha(OWN)
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(plan['scientific'])
    assert all(d.sha(name)==sha for name,sha in plan['source_pins'].items())
    assert all(d.sha(name)==sha for name,sha in parent['source_pins'].items())
    folder=d.path(plan['output']);summary=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json')
    trainfolder=d.path(parent['output']);trained=d.read(trainfolder/'summary.json');trainext=d.read(trainfolder/'external_terminal.json')
    for final,ext in [(summary,external),(trained,trainext)]:
        assert final['closed'] and not final['active'] and ext['closed'] and not ext['active']
        assert all(d.exact_process_exited(ext[key]) for key in ('child_process','guard_process'))
    assert summary['execution']['scientific_digest']==plan['scientific_digest']
    assert trained['execution']['scientific_digest']==parent['scientific_digest']
    good=(summary['status']=='completed' and summary.get('protocol_qualified') is True
          and external['exit_code']==0 and external['within_external_budget']
          and trained['status']=='completed' and trained.get('both4800') is True
          and trainext['exit_code']==0 and trainext['within_external_budget'])
    if good:
        proof=d.read(folder/'prediction_barrier.json')
        assert proof['finite_predictions']==proof['expected']==120 and proof['owned_GT_reads_before_barrier']==0
        assert len(proof['inventory'])==120 and summary['owned_GT_array_reads']==60
        assert summary['raw_GT_reads']==summary['sealed_DEV_reads']==summary['training_updates']==summary['scaler_fits']==0
        assert all(summary['candidates'][arm]['full60']['all_cases_complete'] for arm in ('FNO9','PP256'))
    old=d.read('results/geobc_available_cad_pilot/metrics.json')
    assert old['scientific_qualification'] and old['case_keys']==plan['scientific']['case_keys']
    output='results/geobc_common_grid_comparison';d.path(output).mkdir(parents=True,exist_ok=True)
    assert not d.path(output+'/manifest.json').exists(),'Existing export preserved; no automatic overwrite'
    rows=[]
    for arm in ('FNO9','PP256'):
        train=trained.get('arms',{}).get(arm,{});dev=summary.get('candidates',{}).get(arm,{})
        a=train.get('TRAIN120',{});b=dev.get('full60',{})
        rows.append({'model':arm,'role':'matched macro','qualified':good,
            'TRAIN_U':a.get('case_mean_u_relative_l2'),'TRAIN_S':a.get('case_mean_stress_relative_l2'),
            'DEV_U':b.get('case_mean_u_relative_l2') if good else None,
            'DEV_S':b.get('case_mean_stress_relative_l2') if good else None,
            'DEV_hot':b.get('case_mean_stress_hotspot_relative_l2') if good else None,
            'primary':b.get('selection_score') if good else None,
            'train_epoch_minutes':train.get('training_elapsed_epoch_seconds',0)/60 if train.get('training_elapsed_epoch_seconds') is not None else None,
            'train_monotonic_minutes':train.get('training_elapsed_monotonic_seconds',0)/60 if train.get('training_elapsed_monotonic_seconds') is not None else None,
            'model_inference_seconds':dev.get('inference_seconds'),
            'parameters':(train.get('counts') or {}).get('total')})
    for key,label in [('fno_cad40_fresh4800','OldFNO'),('transolver_pp_cad40_fresh4800','OldPP')]:
        oldrow=old['rows'][key];a=oldrow['full60']
        rows.append({'model':label,'role':'historical development reference, different representation/precision/capacity',
            'qualified':True,'TRAIN_U':None,'TRAIN_S':None,'DEV_U':a['case_mean_u_relative_l2'],
            'DEV_S':a['case_mean_stress_relative_l2'],'DEV_hot':a['case_mean_stress_hotspot_relative_l2'],
            'primary':a['primary_score'],'train_epoch_minutes':None,
            'train_monotonic_minutes':oldrow['inherited_training_seconds']/60,
            'model_inference_seconds':oldrow['actual_inference_seconds'],'parameters':oldrow['total_parameters']})
    leaders=[r['model'] for r in rows[:2] if good and r['primary']==min(x['primary'] for x in rows[:2])] if good else []
    result={'status':'completed','scientific_qualification':good,'rows':rows,'eligible_matched_leaders':leaders,
            'all_ties_retained':True,'paired_CAD20':summary.get('paired_CAD20_FNO_minus_PP') if good else None,
            'perload20':{arm:summary.get('candidates',{}).get(arm,{}).get('per_load20') if good else None for arm in ('FNO9','PP256')},
            'source_evidence':{name:d.identity(path,hashed=True) for name,path in
                [('training_summary',trainfolder/'summary.json'),('training_external',trainfolder/'external_terminal.json'),
                 ('readout_summary',folder/'summary.json'),('readout_external',folder/'external_terminal.json')]},
            'costs':{'TRAIN_whole_epoch_seconds':trainext.get('elapsed_epoch_seconds'),
                     'TRAIN_whole_monotonic_seconds':trainext.get('elapsed_monotonic_seconds'),
                     'readout_epoch_seconds':external.get('elapsed_epoch_seconds'),
                     'readout_monotonic_seconds':external.get('elapsed_monotonic_seconds')},
            'failure_reason':summary.get('error'),'old_baseline_identity':d.identity('results/geobc_available_cad_pilot/metrics.json',hashed=True),
            'limitations':['Single named seed, conditional availability development CAD pool; no unbiased final-test claim',
                           'GT already known from prior complete pilot, no blind label claim',
                           'Controls macro information/heads/capacity/exposure, official architectures/operators/inductive biases differ',
                           'Functional gradient qualification does not measure independent off-node FEM accuracy',
                           'Original S export semantics not a complete physical stress-tensor guarantee'],
            'exporter_numeric_array_validation_performed':False,'GT_model_PT_GPU_reads':0,'final_solver_selected':False}
    def percent(value):return f'{value*100:.1f}%' if value is not None else '缺失'
    text='# 统一表示与容量：FNO／Transolver++\n\n'
    text+=('本次固定4800宏观对照已完整读出。领先的匹配模型：'+ '、'.join(leaders)+'。\n\n') if good else '本次出现技术／协议负结果，取消匹配模型排名，保留完整120预测／60病例分母；旧结果只是背景参照。\n\n'
    text+='| 方法 | 训练U／S | 新CAD U／S | 训练分钟（UTC／mono） | 模型推理秒 |\n|---|---:|---:|---:|---:|\n'
    for r in rows[:2]:
        def num(v):return f'{v:.1f}' if v is not None else '缺失'
        text+=f"| {r['model']} | {percent(r['TRAIN_U'])}／{percent(r['TRAIN_S'])} | {percent(r['DEV_U'])}／{percent(r['DEV_S'])} | {num(r['train_epoch_minutes'])}／{num(r['train_monotonic_minutes'])} | {num(r['model_inference_seconds'])} |\n"
    text+='\n这两行使用共同R64输入、native25查询头、约453万参数、TRAIN120与同样4800次更新。训练误差反映拟合，训练到新CAD的差距反映迁移；不把Hash单几何小于1%的训练拟合与新CAD误差混比。完整旧FNO／旧++参照、热点／逐载荷／逐CAD配对差及成本在metrics.json与CSV。\n\n'
    text+='单seed、可用官方TRAIN池开发病例；标签此前已知，不是最终盲测。模型推理计时不包含初始化、输入读取、写盘与全部阶段成本。离节点独立FEM精度和原signed S完整物理语义仍未解决。\n'
    d.path(output+'/BRIEF.md').write_text(text,encoding='utf-8')
    fields=list(rows[0]);stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    d.path(output+'/metrics.csv').write_text(stream.getvalue(),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,['U','S'],['Displacement relative L2','Signed stress relative L2']):
        for i,r in enumerate(rows):
            for offset,scope,color in [(-.17,'TRAIN','#7f9caf'),(.17,'DEV','#cf7e59')]:
                value=r[scope+'_'+metric]
                if value is not None:ax.bar(i+offset,100*value,width=.32,color=color,label=scope if i==0 else None)
        ax.set_xticks(range(len(rows)),[r['model'] for r in rows]);ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(bottom=0);ax.legend()
    fig.suptitle('Matched macro pair + historical references' if good else 'Nonqualified negative: no new development ranking')
    fig.savefig(d.path(output+'/comparison.png'),dpi=160);plt.close(fig)
    result['CPU_export_seconds']=time.monotonic()-started
    d.write(output+'/metrics.json',result)
    d.write(output+'/manifest.json',{'status':'completed','actual_export':True,'scientific_qualification':good,
        'exporter_source_sha256':d.sha(OWN),'scientific_digest':plan['scientific_digest'],
        'artifacts':{name:d.identity(output+'/'+name,hashed=True) for name in ('BRIEF.md','metrics.json','metrics.csv','comparison.png')},
        'CPU_seconds':result['CPU_export_seconds'],'GT_PT_GPU_reads':0})
    print({'status':'completed','scientific_qualification':good,'CPU_seconds':result['CPU_export_seconds']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:run()
