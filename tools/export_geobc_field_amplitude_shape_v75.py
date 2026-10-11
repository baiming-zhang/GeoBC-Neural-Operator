"""Scalar-only closed oracle diagnostic report, not corrected predictions."""
import argparse
import csv
import time

OWN='tools/export_geobc_field_amplitude_shape_v75.py'
REVIEW='notes/geobc_field_amplitude_shape_v75_exporter_root_review.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.field_amplitude_shape_contract_v75 import PLAN,terminal
    r=d.read(REVIEW)
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256']==d.sha(OWN)
    p,a,e,b=terminal(PLAN);s=p['scientific']
    assert r['SCI']==p['scientific_digest'] and a['protocol_qualified']
    assert a['prediction_reads']==a['prediction_read_attempts']==a['prediction_body_hash_checks']==120
    assert a['cached_GT_reads']==a['cached_GT_read_attempts']==a['cached_GT_body_hash_checks']==60
    assert a['raw_GT_reads']==a['PT_body_reads']==a['FINAL_reads']==a['training_updates']==a['model_forward_calls']==0
    assert set(a['cases'])==set(s['candidate_names'])
    for n in s['candidate_names']:
        assert set(a['cases'][n])==set(s['case_keys'])
        assert all(a['aggregate'][n]['full60'][f]['all_cases_complete']for f in ('U','S'))
    out=d.path('results/geobc_field_amplitude_shape_v75');assert not (out/'manifest.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    branches={}
    lines=['# 高应力误差：幅值还是空间分布','',
        '**给每个已保存预测一个由真值计算的最优整体缩放，检查还剩多少误差。这是oracle诊断，不能用于真实预测，也没有修改任何标签或输出。**','',
        '| FNO端点 | 原位移误差 | 缩放后位移残差 | 原应力误差 | 缩放后应力残差 |',
        '|---|---:|---:|---:|---:|']
    for n in s['candidate_names']:
        z=a['aggregate'][n]['full60'];u,stress=z['U'],z['S']
        lines.append(f'| {int(n.rsplit("_",1)[1])//300}遍 | {100*u["mean_raw_relative_L2"]:.2f}% | {100*u["mean_oracle_shape_relative_L2"]:.2f}% | {100*stress["mean_raw_relative_L2"]:.2f}% | {100*stress["mean_oracle_shape_relative_L2"]:.2f}% |')
        branches[n]='amplitude_priority'if stress['mean_oracle_shape_relative_L2']<=.20 and stress['fraction_mean_squared_error_removed']>=.5 else 'pattern_priority'if stress['mean_oracle_shape_relative_L2']>=.35 else 'inconclusive'
    lines+=['','**两端点即使用真值给出的最优整体幅值，应力残差仍约47.8%。整体幅值不对不足以解释主要误差，下一步优先检查跨几何的空间／signed场分布表示。** 这没有证明某一种网络、输入或插值是唯一原因。','',
        '30／40遍的整体缩放分别只解释约1.11%／6.30%的病例等权应力误差平方；位移能解释34.85%／37.35%。这些是oracle解释比例，不是可部署的精度增益。','',
        f'固定knownDEV60、两端点120份完整物理预测，GPU FP64。完整阶段{e["elapsed_epoch_seconds"]:.2f}秒；本诊断新读取120份预测和60份已有GT，全部预测重验后GT才读各一次。不是GT0；无新模型推理、训练、PT、原CSV、FINAL或scaler拟合。','',
        '每病例、每字段只有一个未中心化最优系数alpha=<p,y>/<p,p>；不加偏置、不逐分量调系数、不试符号规则。标签与原signed输出不覆写，零范数不加floor。全部60例／每载荷20例保留在JSON与CSV。开发集已迭代、单seed，不是独立最终盲测。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,branches=branches,
        deployable_correction=False,final_solver_selected=False,scalars_only_export=True))
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='')as file:
        writer=csv.writer(file);writer.writerow(['method','case','field','raw_relative_L2','oracle_shape_relative_L2','alpha','amplitude_ratio','cosine','fraction_squared_error_removed'])
        for n in s['candidate_names']:
            for k in s['case_keys']:
                for field in ('U','S'):
                    z=a['cases'][n][k][field]
                    writer.writerow([n,k,field,z['raw_relative_L2'],z['oracle_shape_relative_L2'],z['alpha'],z['amplitude_ratio'],z['cosine'],z['oracle_fraction_squared_error_removed']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.6),sharey=True)
    for ax,n in zip(axes,s['candidate_names']):
        z=a['aggregate'][n]['full60'];x=[0,1]
        ax.bar([i-.17 for i in x],[100*z[f]['mean_raw_relative_L2']for f in ('U','S')],width=.34,label='Original')
        ax.bar([i+.17 for i in x],[100*z[f]['mean_oracle_shape_relative_L2']for f in ('U','S')],width=.34,label='Oracle best scale')
        ax.set_xticks(x,['Displacement','Signed stress']);ax.set_title(f'FNO{int(n.rsplit("_",1)[1])//300} / knownDEV60');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean physical relative L2 (%)');axes[1].legend()
    fig.suptitle('Oracle diagnostic only: no deployable correction')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    files=['BRIEF.md','metrics.json','per_case.csv','comparison.png'];assert all((out/f).is_file()for f in files)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        files=files,actual_CPU_seconds=time.perf_counter()-began,new_GT_reads=0,PT_reads=0,GPU_calls=0,branches=branches))
    print(dict(status='completed',CPU_seconds=time.perf_counter()-began,branches=branches))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
