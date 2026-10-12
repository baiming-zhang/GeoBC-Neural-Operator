"""Scalar-only accepted GPU tree report; no prediction or source array reads."""
import argparse
import csv
import json
import time

OWN='tools/export_geobc_physical_GBDT_v92.py'
REVIEW='notes/geobc_physical_tensor_GBDT_v92_exporter_root_review.json'


def export():
    began=time.perf_counter()
    from geobc import available_cad_pilot_data as d
    from geobc.physical_gbdt_contract_v92 import PLAN, TRAIN, DEV
    from geobc.transolver_decoder_contract_v64 import terminal
    p,a,e,b=terminal(PLAN);s=p['scientific'];r=d.read(REVIEW)
    assert r['passed'] and r['conditional_actual_export_released'] and r['source_sha256']==d.sha(OWN) and r['SCI']==p['scientific_digest']
    assert a['protocol_qualified'] and a['training_complete'] and a['budget_true'] and a['actual_GPU_training_verified']
    assert a['rounds_attempted']==a['rounds_completed']==400 and not a['update_in_progress'] and not a['possible_partial_update']
    assert len(a['reads'])==15 and all(v['completed'] for v in a['reads'])
    assert a['knownDEV_prediction_barrier_passed'] and a['own_DEV_prediction_array_reads']==a['own_DEV_prediction_file_hash_reads']==2
    assert a['saved_model_reload_verified'] and a['own_saved_model_loads']==1
    assert a['weight_recipe_actual']['TRAIN_nodes']==683842 and set(a['metrics'])==set(TRAIN+DEV)
    assert sum(v['nodes'] for v in a['metrics'].values())==1092529
    assert all(v['all_finite'] and v['full_original_node_count'] for v in a['metrics'].values())
    assert a['raw_CSV_H5_deck_archive_reads']==a['physical_U_GT_U_PT_reads']==a['new_physical_solves']==a['new20DEV_FINAL_reads']==a['scaler_fits']==a['baseline_replays']==0
    old=s['baseline_scalar_reuse']['physical_tensor_MLP']['validation'];new=a['knownDEV_aggregate']
    out=d.path('results/geobc_physical_tensor_GBDT_v92');assert not (out/'manifest.json').exists();out.mkdir(parents=True,exist_ok=True)
    cold={}
    for key in TRAIN+DEV:
        f=d.read(s['records'][key]['feature_manifest']['path'])
        physical=f['feature_source']['physical_GT_free_cold_seconds'];recover=f['costs']['tensor_recovery_seconds'];mask=f['costs']['incidence_mask_seconds']
        cold[key]=dict(inherited_GT_free_physical_seconds=physical,inherited_tensor_recovery_seconds=recover,
            inherited_incidence_mask_seconds=mask,unneeded_neural_feature_preparation_excluded=True,
            new_cache_input_read_hash_normalize_owned_persist_seconds=a['per_case_input_costs'][key]['read_hash_normalize_ownedcache_epoch_seconds'],
            new_live_GPU_model_load_seconds=a['saved_model_load_epoch_seconds'],
            new_single_case_quantile_predict_owned_persist_seconds=a['predictions'][key]['inference_matrix_predict_save_epoch_seconds'],
            assembled_cold_components_seconds=physical+recover+mask+a['per_case_input_costs'][key]['read_hash_normalize_ownedcache_epoch_seconds']+a['saved_model_load_epoch_seconds']+a['predictions'][key]['inference_matrix_predict_save_epoch_seconds'],
            shared_library_metadata_initialization_separate_seconds=a['library_and_metadata_setup_epoch_seconds'],fresh_combined_cold_remeasured=False,
            cache_includes_own_persistence=True,model_load_amortization_not_assumed=True)
    gate=a['selection_gate'];delta=100*(new['mean_signed_stress_relativeL2']-old['stress_relative_l2_mean']);hotdelta=100*(new['mean_hotspot_relativeL2']-old['hotspot_relative_l2_mean'])
    lines=['# 物理张量＋GPU树模型：路线对照','',
        '**旧三个TRAIN CAD／两个knownDEV CAD／ver，全部原节点；GPU树模型应力误差30.92%，旧物理＋MLP31.76%。热点改善明显，但预声明门仍FAIL，关闭固定配置。**','',
        '| 同旧两开发CAD | signed应力误差 | 热点误差 |','|---|---:|---:|']
    for label,v,w in [('物理＋MLP（冻结标量）',old['stress_relative_l2_mean'],old['hotspot_relative_l2_mean']),('物理＋GPU树模型',new['mean_signed_stress_relativeL2'],new['mean_hotspot_relativeL2'])]:
        lines.append(f'| {label} | {100*v:.2f}% | {100*w:.2f}% |')
    lines += ['',f'应力{delta:+.2f}、热点{hotdelta:+.2f}个百分点。两例应力分别{100*a["metrics"][DEV[0]]["stress_relative_l2"]:.2f}%／{100*a["metrics"][DEV[1]]["stress_relative_l2"]:.2f}%，每例须<25%、热点平均<25%的门：{"PASS" if gate["passed"] else "FAIL"}。',
        '',f'完整阶段{e["elapsed_epoch_seconds"]:.2f}秒，GPU训练{a["training_epoch_seconds"]:.2f}秒，固定400轮；683842个TRAIN节点、五例1092529个节点。实际首轮配置CUDA0／GPUhist，编译信息未替代真实运行。保存模型复载后UBJ模型身份一致，再做全节点预测。',
        '',f'物理输入的历史GT-free求解／应力恢复仍需每CAD约{min(v["inherited_GT_free_physical_seconds"] for v in cold.values()):.0f}–{max(v["inherited_GT_free_physical_seconds"] for v in cold.values()):.0f}秒，不是毫秒求解器。本次只复用缓存，没有新FEM求解；缓存增量、模型加载、组装cold成本、共享初始化分别保存，未重新测量联合freshcold。GPU独占不保证，不能把墙钟差当孤立架构加速。',
        '', '输入来自原GT-free有限元解的chosen sigma6恢复proxy，使用旧TRAIN3-only归一化＋validity；原signedS MPa监督，原公式／符号不猜测或改写。输入7维、rawMPa目标、权重／优化器与旧67维asinh MLP不同，不能归因仅树学习器。',
        '', '直接回归原S、每病例S²能量倒数加权并均值归一为1；400轮／depth6／eta.05／bins256／seed2026固定，没有DEV早停、截断、q范围约束、伪tensorGT、材料拟合或符号规则救结果。分位cut仅TRAIN，DEV矩阵ref TRAIN。',
        '', '两个DEV的全部有限不可变预测先保存、文件哈希与数组Cartesian重验，再各读取一次原缓存S；TRAIN_S3、DEV_S2、sigma/valid5各一次。原CSV/H5/deck/archives、物理U／GTU／PT、新20DEV／FINAL、scalerfit、新物理解算及旧基线重跑均0。',
        '', '**这是历史3TRAIN／2knownDEV／ver路线比较，不是主100TRAIN／20knownDEV三载荷的公平FNO排名或最终盲测。** Tree输出分段常数不可微，不能当已实现连续hash算子的替代或独立离节点FEM精度。',
        '', '依赖准备另计182.59秒：首次NCCL下载超时180.02秒，复用95MB缓存offline安装2.57秒，无再次网络下载。二进制／数据／模型主体留本地，只发布标量与源码。']
    (out/'BRIEF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d.write(out/'metrics.json',dict(actual_native=a,actual_external=e,terminal_binding=b,old_physics_MLP_scalar=old,cold_component_accounting=cold,
        source_stress_semantics_identified=False,final_solver_selected=False,differentiable_continuous_field=False))
    with (out/'per_case.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','split','nodes','signed_S_relativeL2','hotspot_relativeL2','RMSE_MPa','MAE_MPa','max_error_MPa','S2_weighted_sign_error'])
        for k in TRAIN+DEV:
            v=a['metrics'][k];writer.writerow([k,'TRAIN' if k in TRAIN else 'knownDEV',v['nodes'],v['stress_relative_l2'],v['hotspot_relative_l2'],v['RMSE_MPa'],v['MAE_MPa'],v['max_abs_error_MPa'],v['S2_weighted_sign_error']])
    loss=[json.loads(v) for v in (d.path(p['output'])/'loss.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(loss)==400 and [v['round'] for v in loss]==list(range(1,401));d.write(out/'loss.json',loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.8))
    axes[0].plot([v['round'] for v in loss],[v['weighted_TRAIN_RMSE_MPa'] for v in loss]);axes[0].set_xlabel('Fixed GPU boosting rounds');axes[0].set_ylabel('Weighted TRAIN RMSE (MPa)')
    for shift,vals,name in [(-.18,[old['stress_relative_l2_mean'],old['hotspot_relative_l2_mean']],'Previous physical MLP'),(.18,[new['mean_signed_stress_relativeL2'],new['mean_hotspot_relativeL2']],'Physical GPU GBDT')]:
        axes[1].bar([shift,1+shift],[100*v for v in vals],.36,label=name)
    axes[1].set_xticks([0,1],['Signed S','Hotspot']);axes[1].set_ylabel('Mean knownDEV relative L2 (%)');axes[1].set_ylim(0,70);axes[1].legend(loc='upper center',fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Legacy3 TRAIN /2 knownDEV/ver; all nodes, not main100 CAD ranking');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    d.write(out/'manifest.json',dict(status='completed',SCI=p['scientific_digest'],source_sha256=d.sha(OWN),terminal_binding=b,
        files=['BRIEF.md','metrics.json','per_case.csv','loss.json','comparison.png'],actual_CPU_seconds=time.perf_counter()-began,
        new_GT_PT_arrays_GPU=0,selection_gate=gate,final_solver_selected=False))
    print(dict(status='completed',CPU_seconds=time.perf_counter()-began,knownDEV_stress=new['mean_signed_stress_relativeL2'],gate=gate['passed']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true')
    if p.parse_args().export:export()
    else:print('{"status":"source_only","new_GT_PT_arrays_GPU":0}')
