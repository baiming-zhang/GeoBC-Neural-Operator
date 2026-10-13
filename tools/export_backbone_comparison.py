"""Recover selected historical scalar backbone results from ordinary Git history."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'common_grid_control_4800':'results/geobc_common_grid_comparison/metrics.json',
    'common_grid_control_cases':'results/geobc_common_grid_comparison/source_evidence/common_grid_development60_v1/summary.json',
    'early_family_screen':'results/geobc_fixed4800_pilot/metrics.json',
    'common_input_control':'results/geobc_short_screen/metrics.json',
    'matched_exposure':'results/geobc_matched20_readout_v79/metrics.json',
    'native_context_diagnostic':'results/geobc_nativeTS100_matched10_v80/metrics.json',
}
ORIGINAL_REPORTS={
    'ALL_METHODS_RESULTS.md':'f621a3bc5',
    'results/geobc_exploration/REPORT.md':'f621a3bc5',
    'results/TRANSOLVER_COMPARISON.md':'f621a3bc5',
    'results/geobc_common_grid_comparison/BRIEF.md':'e5a052535',
    'results/RESEARCH_HISTORY.md':'f621a3bc5',
}


def restore_original_reports():
    archive=ROOT/'results/research_history'
    for path,revision in ORIGINAL_REPORTS.items():
        value=subprocess.run(['git','show',revision+':'+path],cwd=ROOT,check=True,capture_output=True).stdout
        target=archive/'original_reports'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
    (archive/'sources.json').write_text(json.dumps({path:dict(revision=revision,original_url='https://github.com/baiming-zhang/GeoBC-Neural-Operator/blob/'+revision+'/'+path,local_path='original_reports/'+path) for path,revision in ORIGINAL_REPORTS.items()},indent=2)+'\n')


def export(revision,output):
    restore_original_reports()
    def historical(name):
        source_revision='e5a052535' if name.startswith('common_grid_control') else revision
        result=subprocess.run(['git','show',source_revision+':'+SOURCES[name]],cwd=ROOT,check=True,capture_output=True)
        return json.loads(result.stdout)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source={key:historical(key) for key in SOURCES}
    experiments={};case_rows=[];aggregate_rows=[]
    def add(name,label,group,means,cases,parameters=None,precision=None,training_seconds=None,added_UTC=None,added_mono=None,inference_UTC=None,inference_mono=None,cost_scope=None):
        U,S,H=[means.get('case_mean_'+field+'_relative_l2') for field in ['u','stress','stress_hotspot']]
        if len(cases)!=group['case_denominator']:raise ValueError('Historical fixed case denominator differs')
        for field,expected in [('u_relative_l2',U),('stress_relative_l2',S),('stress_hotspot_relative_l2',H)]:
            if expected is not None and abs(sum(row[field] for row in cases.values())/len(cases)-expected)>1e-10:raise ValueError('Saved historical mean and case scalars disagree')
        row=dict(experiment=name,model=label,evaluation_role=group['evaluation_role'],TRAIN_CADs=group['TRAIN_CADs'],evaluation_CADs=group['CAD_denominator'],cases=group['case_denominator'],seeds=1,updates=group['updates'],passes=group['passes'],U_percent=U*100,S_percent=S*100,primary_percent=(U+S)*50,hotspot_percent=None if H is None else H*100,parameters=parameters,precision=precision,training_seconds=training_seconds,added_training_UTC_seconds=added_UTC,added_training_monotonic_seconds=added_mono,inference_UTC_seconds=inference_UTC,inference_monotonic_seconds=inference_mono,cost_scope=cost_scope)
        group['models'].append(row);aggregate_rows.append(row)
        for key,value in cases.items():case_rows.append(dict(experiment=name,model=label,evaluation_role=group['evaluation_role'],case=key,U_percent=value['u_relative_l2']*100,S_percent=value['stress_relative_l2']*100,hotspot_percent=None if value.get('stress_hotspot_relative_l2') is None else value['stress_hotspot_relative_l2']*100))
    def group(name,train,CADs,cases,updates,passes,role,scope):
        experiments[name]=dict(TRAIN_CADs=train,CAD_denominator=CADs,case_denominator=cases,updates=updates,passes=passes,seeds=[2026],evaluation_role=role,scope=scope,physical_displacement_gradient_covariates=False,models=[])
        return experiments[name]
    early=source['early_family_screen']
    g=group('early_family_screen',40,5,5,4800,40,'historical_five_CAD_vertical_diagnostic','Five unused training CAD identities; U previously revealed, S read after predictions. Mixed official TRAIN/TEST membership. Unequal capacity/representation; not current DEV20 or FINAL20.')
    for name in ['fno','transolver','transolver_pp','ts3','mlp']:
        v=early['results'][name];u=v['held5']['U_known'];s=v['held5']['S_field_blind']
        means={**u['aggregate'],**s['aggregate']};cases={key:{**u['per_case'][key],**s['per_case'][key]} for key in early['case_keys']}
        add('early_family_screen',v['display_name'],g,means,cases,parameters=v['parameter_counts']['total'],training_seconds=v['training_cost_context']['train_seconds'],cost_scope='Historical measured train_seconds; initialization, checkpoint and readout excluded. Not full campaign cost.')
    control4800=source['common_grid_control_4800'];cases4800=source['common_grid_control_cases']
    if control4800['status']!='completed' or cases4800['status']!='completed':raise ValueError('Historical matched4800 control incomplete')
    g=group('common_grid_control_4800',40,20,60,4800,40,'known_development','Common R64 input, PointNet, native25 query head and approximately matched 4.53M capacity. Original source revision e5a052535. Old unequal-representation references are not mixed into this two-arm control.')
    for key,label in [('FNO9','FNO'),('PP256','Transolver++')]:
        row=next(r for r in control4800['rows'] if r['model']==key)
        full=cases4800['candidates'][key]['full60']
        add('common_grid_control_4800',label,g,full,full['per_case'],parameters=row['parameters'],training_seconds=row['train_epoch_minutes']*60,inference_mono=row['model_inference_seconds'],cost_scope='Matched4800 training UTC '+str(row['train_epoch_minutes']*60)+' s / monotonic '+str(row['train_monotonic_minutes']*60)+' s; training interval from source table, not whole campaign. Inference excludes initialization/input IO/save.')
    control=source['common_input_control']
    if control['status']!='completed':raise ValueError('Historical common-input comparison is incomplete')
    g=group('common_input_control',40,20,60,12000,100,'known_development','Same R64 input, PointNet and native25 query head; approximately 4.53M parameters, complete 12000-update endpoints continued from 4800. Single seed. Interrupted 24000 campaign was not completed.')
    for key,label in [('FNO9_step12000','FNO'),('PP256_step12000','Transolver++')]:
        v=control['candidates'][key]
        add('common_input_control',label,g,v['full60'],v['full60']['per_case'],inference_mono=v['inference_seconds'],cost_scope='Recorded full60 inference-only monotonic duration; cumulative training/interruption cost not reconstructed, null is unknown.')
    matched=source['matched_exposure']['actual_native']
    if matched['status']!='completed' or not matched['prediction_barrier_complete']:raise ValueError('Historical matched-exposure readout incomplete')
    g=group('matched_exposure',100,20,60,6000,20,'known_development','Same 100 CADs/300 loads and 20 full-node passes, physical-unit joint U/S objective and TRAIN-only scaler. Representation, capacity and arithmetic differ. FNO is a frozen scalar reference from an earlier readout, not rerun with the new candidates.')
    labels={'FNO100_PHYSICAL_6000':'FNO','FNO_GLOBAL_HASH20':'FNO + Hash decoder','TS_CONTINUOUS20':'Original Transolver + Hash decoder'}
    for key,label in labels.items():
        v=matched['candidates'][key]
        add('matched_exposure',label,g,v['full60'],v['full60']['per_case'],parameters=v['model_profile']['parameter_count'],precision=v['model_profile']['precision'],added_UTC=v.get('added_training_interval_epoch_seconds'),added_mono=v.get('added_training_interval_monotonic_seconds'),inference_UTC=v.get('inference_epoch_seconds'),inference_mono=v['inference_seconds'],cost_scope='UTC and monotonic inference clocks retained separately. Added training is Hash10-to20 / TS5-to20 including intermediate readouts/saves, not cumulative training; FNO additional interval not comparable and left null.')
    native=source['native_context_diagnostic'];n=native['actual_native']
    g=group('native_context_diagnostic',100,100,300,3000,10,'TRAIN_fitting_only','Native original-node context versus frozen R64 context. No new development or final evaluation; failed continuation criterion closes this representation branch, not the whole Transolver family.')
    full=n['endpoints']['3000']['full_TRAIN']
    add('native_context_diagnostic','Original Transolver native context',g,full,full['per_case'],cost_scope='TRAIN-only diagnostic. Whole stage UTC '+str(native['actual_external']['elapsed_epoch_seconds'])+' s, includes setup and earlier endpoint; no comparable per-model inference cost.')
    full=native['dense_TRAIN_references']['10']
    add('native_context_diagnostic','Original Transolver R64 context',g,full,full['per_case'],cost_scope='Frozen TRAIN-only reference; no new inference and no zero-filled cost.')
    metrics=dict(kind='historical_backbone_selection',source_revision=revision,source_paths=SOURCES,metric_units='Percent, case-equal mean of full-original-node physical U3 and original signed S relative L2; hotspot uses top 1% absolute true S',experiments=experiments,interpretation='FNO is the workhorse package under tested conditions, not a universal backbone ranking or architecture ceiling. Historical PHYSICAL names describe target units/objective, not U3/gradient9 predictor augmentation.',new_training_updates=0,new_model_inference=0,new_GT_reads=0)
    metrics['source_revisions']={key:'e5a052535' if key.startswith('common_grid_control') else revision for key in SOURCES}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for filename,rows in [('summary.csv',aggregate_rows),('per_case.csv',case_rows)]:
        with (output/filename).open('w',encoding='utf-8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    try:
        import matplotlib
    except ModuleNotFoundError:
        sys.path.insert(0,str(ROOT.parents[1]/'.local_deps'))
        import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=experiments['matched_exposure']['models'];x=np.arange(3);labels=['FNO\n4.53M','FNO + Hash\n12.25M','Transolver + Hash\n9.60M']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,fields,ylabel,title in [(axes[0],[('U_percent','U','#247fa9'),('S_percent','Signed S','#ed3a94')],'Full-node error (%)','Historical matched20: accuracy'),(axes[1],[('inference_monotonic_seconds','Monotonic','#247fa9'),('inference_UTC_seconds','UTC','#ed3a94')],'Full60 inference seconds','Recorded inference cost')]:
        for offset,(key,label,color) in zip([-.18,.18],fields):
            bars=ax.bar(x+offset,[r[key] for r in rows],width=.34,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
        ax.set_xticks(x,labels);ax.set_ylabel(ylabel);ax.set_title(title,color='#172638',fontweight='bold');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(r[key] for r in rows for key,_,_ in fields)*1.22)
    fig.suptitle('Backbone/package selection before physical U/gradient inputs',color='#172638',fontweight='bold')
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    controlled=[next(row for row in control4800['rows'] if row['model']==name) for name in ['FNO9','PP256']]
    x=np.arange(2);fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),constrained_layout=True)
    panels=[('TRAIN fitting errors',[('TRAIN_U','U','#247fa9'),('TRAIN_S','Signed S','#ed3a94')],'Full-node error (%)'),('DEV20/60 errors',[('DEV_U','U','#247fa9'),('DEV_S','Signed S','#ed3a94')],'Full-node error (%)'),('Recorded training interval',[('train_monotonic_minutes','Monotonic','#247fa9'),('train_epoch_minutes','UTC','#ed3a94')],'Minutes')]
    for ax,(title,fields,ylabel) in zip(axes,panels):
        factor=100 if ylabel=='Full-node error (%)' else 1
        for offset,(key,label,color) in zip([-.18,.18],fields):
            bars=ax.bar(x+offset,[r[key]*factor for r in controlled],width=.34,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
        ax.set_xticks(x,['FNO9\n4.527M','Transolver++\n4.523M']);ax.set_ylabel(ylabel);ax.set_title(title,color='#172638',fontweight='bold');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(r[key]*factor for r in controlled for key,_,_ in fields)*1.23)
    fig.suptitle('Historical matched4800 control: same R64 input and native25 head',fontweight='bold',color='#172638')
    fig.savefig(output/'common_grid_control.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(experiments=len(experiments),aggregate_rows=len(aggregate_rows),per_case_rows=len(case_rows),new_training_updates=0,new_GT_reads=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='0277577');p.add_argument('--output',default='results/backbone_comparison');args=p.parse_args();export(args.revision,args.output)
