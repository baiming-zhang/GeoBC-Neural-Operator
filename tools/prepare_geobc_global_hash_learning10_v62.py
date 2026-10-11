"""Metadata-only independent new stage after actual shared-field fitting evidence."""
import argparse
import datetime

PLAN='notes/geobc_global_hash_learning10_v62_plan.json'
REVIEW='notes/geobc_global_hash_learning10_v62_root_review.json'
SOURCE_REVIEW='notes/geobc_global_hash_learning10_v62_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.global_hash_learning10_contract_v62 import inputs,SOURCES,MODEL_SOURCES,order,PARENT
    up=inputs();ps=up['parent']['scientific'];a=up['parent_summary'];sr=d.read(SOURCE_REVIEW)
    proposal=d.read('notes/geobc_global_hash_learning10_v62_proposal.json')
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES+MODEL_SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert proposal['parent_SCI']==up['parent']['scientific_digest'] and proposal['upstream_shared_terminal_binding']==up['shared_binding']
    assert proposal['parent_checkpoint_stat']==up['checkpoint_stat']
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:ps[k]for k in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity','scaler','memory_limit_bytes','GPU_lock','loss','optimizer','decoder_chunk_size')}
    s.update(parent_training_plan_identity=d.identity(PARENT,hashed=True),parent_training_SCI=up['parent']['scientific_digest'],
        parent_checkpoint_stat=up['checkpoint_stat'],parent_summary_identity=up['parent_binding']['summary'],parent_external_identity=up['parent_binding']['external'],
        parent_norms_identity=up['norms_identity'],parent_case_exposure_counts=a['case_exposure_counts'],parent_full_order_digest=ps['full_order_digest'],
        parent_TRAIN300=a['endpoints']['1500']['full_TRAIN'],same10_FNO_TRAIN300=up['FNO_summary']['endpoints']['3000']['full_TRAIN'],
        parent_step=1500,additional_updates=1500,fixed_updates=3000,endpoint_passes=[10],full_order_digest=d.digest(order(up['keys'])),
        fresh_initialization=False,cache_setup_budget_seconds=600,training_budget_seconds=600,TRAIN_readout_budget_seconds=300,whole_budget_seconds=1800,
        automatic_continuation=False,DEV_GT_reads_allowed=0,FINAL_reuse_for_selection=False,parameter_count=12245894,Hash_architecture_hyperparameter_changes=0,
        TRAIN_gate=proposal['TRAIN_gate'],proposal_identity=d.identity('notes/geobc_global_hash_learning10_v62_proposal.json',hashed=True),
        precision='FP32 encoder/operator, nativeTCNN FP16 decoder withFP32master; highestTF32off',
        parent_integrity_policy=proposal['parent_integrity_policy'],old_negative5_gate_preserved=True,
        release_scope='New actual shared3 evidence justifies one modest100CAD5to10 check, no automatic extension or legacygate override',
        limits=proposal['limits'],physical_target_policy='Aligned signed U(mm)/S(MPa), all original nodes, physicalTRAIN300once; unused encodedlabels removed before reading')
    manifest=d.read(ps['dataset_identity']['path'])
    s['parent_replay_fixed_key']=max(up['keys'],key=lambda k:manifest['records'][k]['num_nodes'])
    s['parent_replay_policy']='One largest TRAIN full-node physical scalar replay before updates, same prior readout seed, restores RNG; reuse resident target, no new GT array'
    p=dict(kind='GeoBC_globalHash_evidence_gated_physical5_to10_v62',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/GlobalHash100_physical10_v62',prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={k:up[k+'_binding']for k in ('parent','shared','FNO')},source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_released',SCI=p['scientific_digest'],new_GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
