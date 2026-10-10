"""Index the actually verified private commit, preserving prior receipts."""
from geobc import available_cad_pilot_data as d


def run():
    target='84f4a4898e44c81c2234eb74fc3fe996e556e7fc'
    filename='notes/github_gitdata_publication_'+target+'_query_matched_scalar_20261010t1808_receipt.json'
    receipt=d.read(filename)
    assert receipt['completed'] and receipt['status']=='completed'
    assert receipt['target_commit']==receipt['remote_main_after']==target
    assert receipt['exact_tree_verified'] and receipt['exact_commit_verified'] and not receipt['force_update']
    assert all(receipt[k]['passed'] and receipt[k]['private'] and receipt[k]['user_id']==174442095
               and receipt[k]['repository_id']==1410470321 for k in ('verification_before','verification_after'))
    assert receipt['source_sha256']==d.sha('tools/publish_reviewed_gitdata_snapshot.ps1')
    assert receipt['root_review_sha256']==d.sha('notes/github_gitdata_publication_root_review.json')
    d.write('notes/github_gitdata_84f_verified_receipt.json',receipt)
    value=d.read('notes/github_publish_receipt.json')
    value.update(phase='published',status='published',local_commit=target,last_published_commit=target,
                 published_at_utc=receipt['finished_at_utc'],last_published_at_utc=receipt['finished_at_utc'],
                 immutable_push_receipt=filename,private_identity_confirmed=True,repository_private_verified=True,
                 latest_publication_route='Exact GitData REST normal fast-forward; immutable scalar evidence copied into results',
                 latest_publication_elapsed_seconds=receipt['elapsed_seconds'],
                 prior_failed_receipt='notes/github_gitdata_publication_57bcf1f80635324896f581bc6bfcff7dba05fcec_query_matched_20261010t1806_receipt.json')
    d.write('notes/github_publish_receipt.json',value)
    plan=d.read('notes/github_publish_plan.json')
    plan['latest_verified_publication']={'commit':target,'private':True,'receipt':filename,
        'finished_at_utc':receipt['finished_at_utc'],'elapsed_seconds':receipt['elapsed_seconds'],
        'scope':'Actual native27 + FP32 query9 scalar reports/cloud figures, matched model/fullcase probe and genuine ongoing TRAIN4800 source/launch; training not completed'}
    plan['latest_verified_private_commit']=target
    plan['latest_verified_private_at_utc']=receipt['finished_at_utc']
    plan['latest_verified_private_receipt']=filename
    d.write('notes/github_publish_plan.json',plan)
    qa=d.read('results/geobc_available_cad_pilot/field_visualization_QA.json')
    qa.update(private_commit_contains_these_fields=True,next_result_publication_pending=False,published_commit=target)
    d.write('results/geobc_available_cad_pilot/field_visualization_QA.json',qa)
    readme=d.path('README.md').read_text(encoding='utf-8')
    readme=readme.replace('私人GitHub最新真实验收提交为`b88cee5`，包含FNO配对终态、新20CAD完整结果与查询适配器。之后的原节点云图、真实查询检查与精度对照暂在本地，随下一次有意义结果更新。',
        '私人GitHub最新真实验收提交为`84f4a48`，包含新20CAD完整结果、原节点云图、真实查询检查、FP32精度对照，以及正在运行的统一配置训练代码与启动证据；尚无两臂4800终态结果。')
    d.path('README.md').write_text(readme,encoding='utf-8')
    print({'status':'verified_private_publication','commit':target,'seconds':receipt['elapsed_seconds']})


if __name__=='__main__':run()
