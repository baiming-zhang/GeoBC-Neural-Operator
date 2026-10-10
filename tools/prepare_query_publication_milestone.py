"""Organize completed diagnostics and the genuine ongoing macro stage."""
from datetime import datetime,timezone
from geobc import available_cad_pilot_data as d


def run():
    now=datetime.now(timezone.utc).isoformat()
    for name in ['notes/geobc_transolver_comparison_decision.json','notes/geobc_exploration_plan.json']:
        value=d.read(name)
        value['field_query_qualification']={'status':'completed','report':'results/geobc_field_queries/BRIEF.md',
            'native_cases':27,'FP32_cases':9,'native_seconds':25.727333784,'FP32_seconds':9.597258568,
            'independent_offnode_FEM_accuracy_measured':False,
            'conclusion':'Original FNO query stable; new head highest-FP32 fixes TF32 chunk/AD-FD inconsistency; ++ stochastic and context-dependent'}
        value['matched_macro_pair']={'status':'actually_running','plan':'notes/geobc_common_grid_train_plan.json',
            'live_identity_receipt':'notes/geobc_common_grid_live_milestone.json',
            'TRAIN_keys':120,'fixed_updates_each':4800,'final_solver_selected':False,
            'next':'After genuine closed/exited endpoints, all60 existing development cases for both arms, not sealed20DEV'}
        value['updated_at_utc']=now;d.write(name,value)
    readme=d.path('README.md').read_text(encoding='utf-8-sig')
    readme=readme.replace('先做一次真实完整TRAIN病例的前向／反向显存与耗时测量，零优化器更新，再据实际成本确定固定4800训练预算；',
        '真实完整TRAIN病例的前向／反向资源测量已完成，FNO和++显存峰值分别0.80与9.32GiB；共享初始化完全一致，两边各4800次更新的正式对照已在北京时间02:02:35启动；')
    d.path('README.md').write_text(readme,encoding='utf-8')
    plan=d.read('notes/github_publish_plan.json')
    additions=['notes/geobc_continuous_query_followup_plan.json','notes/geobc_continuous_query_followup_root_review.json',
        'notes/geobc_field_query_actual_terminal_receipt.json','notes/geobc_fp32_query_ablation_plan.json',
        'notes/geobc_fp32_query_ablation_root_review.json','notes/geobc_common_grid_probe_plan.json',
        'notes/geobc_common_grid_probe_root_review.json','notes/geobc_common_probe_interpretation_root_review.json',
        'notes/geobc_common_grid_train_plan.json','notes/geobc_common_grid_train_root_review.json',
        'notes/geobc_common_grid_live_milestone.json','notes/geobc_common_probe_windows_launch.json',
        'notes/geobc_common_training_windows_launch.json','notes/github_gitdata_b88_verified_receipt.json',
        'notes/source_revisions/github_gitdata_b88_limit90_source.ps1',
        'runs/geobc_exploration/continuous_query_qualification_v1',
        'runs/geobc_exploration/FP32_query_ablation_v1','runs/geobc_exploration/common_grid_fullcase_probe_v1']
    for name in additions:
        assert d.path(name).exists(),name
        if name not in plan['allowlist']:plan['allowlist'].append(name)
    plan['commit_message']='Qualify GeoBC continuous queries and start matched FNO-Transolver pair'
    plan['ongoing_stage']='Matched R64 4.53M FNO/++ TRAIN120 fixed4800; actual launch and first real fullN gradients, results pending'
    d.write('notes/github_publish_plan.json',plan)
    prior=d.read('notes/github_publish_receipt.json')
    assert prior.get('phase')=='published'
    d.write('notes/github_publish_b88_prior_query_canonical_receipt.json',prior)
    print({'status':'organized','new_allowlist':len(additions),'GT_arrays':0})


if __name__=='__main__':run()
