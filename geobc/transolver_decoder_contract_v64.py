"""Metadata-only release of a shared3 whole-context backbone diagnostic."""
from . import available_cad_pilot_data as d
from .shared_decoder_pair_contract_v61 import terminal,complete_physical,check_identity,order

SOURCES=('geobc/transolver_conditioned_hash_v64.py','geobc/transolver_decoder_contract_v64.py',
    'tools/train_geobc_transolver_decoder_v64.py','tools/run_geobc_transolver_decoder_guard_v64.py',
    'tools/prepare_geobc_transolver_decoder_v64.py')


def inputs():
    sp,sa,se,sb=terminal('notes/geobc_shared_decoder_v61_plan.json')
    rp,ra,re,rb=terminal('notes/geobc_global_hash10_readout_v63_plan.json')
    for path,p,b in [('results/geobc_shared_decoder_v61/manifest.json',sp,sb),('results/geobc_global_hash10_readout_v63/manifest.json',rp,rb)]:
        z=d.read(path);assert z['status']=='completed' and z['SCI']==p['scientific_digest'] and z['terminal_binding']==b
    keys=sp['scientific']['selected_TRAIN_keys'];assert len(keys)==3
    assert sa['training_complete'] and ra['protocol_qualified'] and ra['committed_finite_predictions']==ra['owned_GT_array_reads']==60
    for row in sa['aggregate_endpoints']['FNO_GLOBAL_HASH'].values():complete_physical(row,keys)
    proposal=d.read('notes/geobc_transolver_continuous_decoder_v64_proposal.json')
    assert proposal['actual_GPU_execution']==0 and proposal['design']['fixed_TRAIN_keys']==keys
    return dict(shared=sp,shared_summary=sa,shared_external=se,shared_binding=sb,readout=rp,readout_summary=ra,readout_binding=rb,keys=keys,proposal=proposal)


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);up=inputs();ps=up['shared']['scientific']
    assert p['kind']=='GeoBC_originalTS_same_continuous_decoder_shared3_v64'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    for k in ('shared','readout'):
        assert r['upstream_terminal_bindings'][k]==up[k+'_binding']
        assert all(p['source_pins'].get(a)==h for a,h in up[k]['source_pins'].items())
    for k in ('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size',
        'parent_TRAIN_keys','selected_TRAIN_keys','whole_budget_seconds','training_budget_seconds','cache_setup_budget_seconds','representative_field','visualization_sample_nodes_max'):
        assert s[k]==ps[k]
    assert s['families']==['TS_ORIGINAL_GLOBAL_HASH'] and s['fresh_models_per_family']==1
    assert s['updates_per_field']==1000 and s['total_updates']==s['total_updates_per_family']==3000 and s['case_pass_endpoints']==[0,100,1000]
    assert s['full_order_digest']==d.digest(order(up['keys'])) and s['fresh_initialization'] and s['parent_checkpoint_reads_allowed']==0
    assert s['expected_reference_initialization']==up['shared_summary']['model_initialization']['FNO_GLOBAL_HASH']
    assert s['FNO_Hash_shared_reference']==up['shared_summary']['aggregate_endpoints']['FNO_GLOBAL_HASH']
    assert s['official_TS_kwargs']==dict(n_layers=5,n_hidden=256,n_head=8,slice_num=32,mlp_ratio=1,dropout=0.0)
    assert s['DEV_FINAL_reads_allowed']==0 and not s['automatic_continuation'] and not s['FINAL_reuse_for_selection']
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1200 and s['training_budget_seconds']==900
    assert p['output']=='runs/geobc_exploration/Transolver_continuous_decoder_shared3_v64'
    m=d.read(check_identity(s['dataset_identity']))
    assert all(m['case_roles'][k]=='TRAIN' and k in m['train_keys']['100']for k in up['keys'])
    return p,r,m,up
