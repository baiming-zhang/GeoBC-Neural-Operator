"""Explicit metadata preparation, no model construction or numerical data."""
import argparse

PLAN='notes/geobc_transolver_decoder_v64_plan.json'
REVIEW='notes/geobc_transolver_decoder_v64_root_review.json'
SOURCE_REVIEW='notes/geobc_transolver_decoder_v64_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_decoder_contract_v64 import inputs,SOURCES,order
    up=inputs();ps=up['shared']['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    names=('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size',
        'parent_TRAIN_keys','selected_TRAIN_keys','whole_budget_seconds','training_budget_seconds','cache_setup_budget_seconds','representative_field','visualization_sample_nodes_max')
    s={k:ps[k]for k in names}
    s.update(families=['TS_ORIGINAL_GLOBAL_HASH'],fresh_models_per_family=1,updates_per_field=1000,total_updates_per_family=3000,total_updates=3000,
        case_pass_endpoints=[0,100,1000],fresh_initialization=True,parent_checkpoint_reads_allowed=0,
        expected_reference_initialization=up['shared_summary']['model_initialization']['FNO_GLOBAL_HASH'],
        FNO_Hash_shared_reference=up['shared_summary']['aggregate_endpoints']['FNO_GLOBAL_HASH'],
        full_order_digest=d.digest(order(up['keys'])),DEV_FINAL_reads_allowed=0,automatic_continuation=False,FINAL_reuse_for_selection=False,
        proposal_identity=d.identity('notes/geobc_transolver_continuous_decoder_v64_proposal.json',hashed=True),
        official_TS_kwargs=dict(n_layers=5,n_hidden=256,n_head=8,slice_num=32,mlp_ratio=1,dropout=0.0),
        runtime_factory='geobc.transolver_conditioned_hash_v64.build_solver',
        precision='FP32 originalTS/PointNet, nativeTCNN FP16 withFP32master; highestTF32off',
        scientific_scope='Samecontinuousdecoder/input/sharedTRAIN3 exposure; officialoriginalTS backbone replacesFNO, unequalbackbonecapacity/cost, not newCAD ranking',
        RNG_policy='Freshreference bytes match actualv61; replace onlyoperator underfork preservingCPU/CUDARNG, allqualification/readout restoresfourRNG; Adam0',
        visualization_policy='Originalnode display sample sharedtruthP99 range explicitclipping; metrics all originalnodes',
        limits=up['proposal']['design']['limits'])
    p=dict(kind='GeoBC_originalTS_same_continuous_decoder_shared3_v64',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/Transolver_continuous_decoder_shared3_v64',prepared_at_utc=d.utc())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={k:up[k+'_binding']for k in ('shared','readout')},source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='prepared_not_released',SCI=p['scientific_digest'],GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
