"""Own4800-to12000 continuation, same100 CAD and fixed recipe; metadata only."""
from . import available_cad_pilot_data as d

KIND = 'GeoBC_global_operator_hash_matched40_extension_v7'


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']==KIND and r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest']==r['scientific_digest']==d.digest(s) and p['source_pins']==r['source_pins']
    required=('geobc/operator_hash_extension_contract_v7.py','tools/train_geobc_operator_hash_extension_v7.py','tools/run_geobc_operator_hash_extension_guard_v7.py')
    assert all(k in p['source_pins'] for k in required)
    assert all(d.sha(k)==v for k,v in p['source_pins'].items())
    from .operator_hash_readout_contract_v5 import check_identity,complete_physical
    for name,binding in r['upstream_bindings'].items():
        for item in binding.values():check_identity(item)
        summary=d.read(binding['summary']['path']);external=d.read(binding['external']['path'])
        assert summary['status']=='completed' and summary['closed'] and not summary['active']
        assert external['closed'] and not external['active'] and external['exit_code']==0 and external['within_external_budget']
        assert all(d.exact_process_exited(external[k]) for k in ('guard_process','child_process'))
        if name=='reader': assert summary['protocol_qualified'] and summary['committed_finite_predictions']==240 and summary['owned_GT_array_reads']==60
        if name=='native_interface': assert summary['native_TCNN_interface_verified'] and summary['numerical_GT_reads']==summary['training_updates']==0
    assert set(r['upstream_bindings'])=={'training','reader','native_interface'}
    from .operator_hash_recovery_contract_v6 import approved_contract as parent_contract
    own,_,manifest,parent_order=parent_contract(check_identity(s['warm_parent_training_plan_identity']))
    parent=d.read(r['upstream_bindings']['training']['summary']['path']);ts=own['scientific']
    assert d.relative(r['upstream_bindings']['training']['summary']['path'])==d.relative(d.path(own['output'])/'summary.json')
    assert s['family']==ts['family']=='FNO_GLOBAL_HASH' and s['train_CAD_count']==ts['train_CAD_count']==100
    assert s['seed']==ts['seed']==2026 and s['TRAIN_keys']==ts['TRAIN_keys']==manifest['train_keys']['100']
    assert len(s['TRAIN_keys'])==300 and all(manifest['case_roles'][k]=='TRAIN' for k in s['TRAIN_keys'])
    assert s['dataset_identity']==ts['dataset_identity'] and s['scaler']==ts['scaler']==manifest['scaler']
    assert s['optimizer']==ts['optimizer'] and s['loss']==ts['loss']=='full_original_node_encoded4_case_mean_MSE'
    assert not s['fresh_initialization'] and s['endpoint_passes']==[40] and s['decoder_chunk_size']==32768
    assert s['memory_limit_bytes']==30500000000 and s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert s['training_budget_seconds']==600 and s['cache_setup_budget_seconds']==s['TRAIN_readout_budget_seconds']==300
    assert s['whole_budget_seconds']==1200 and s['fixed_updates']==12000 and s['actual_additional_updates']==7200
    assert s['warm_parent_scientific_digest']==own['scientific_digest']
    assert s['warm_checkpoint_identity']==parent['endpoints']['4800']['checkpoint_identity']
    d.stat_matches(s['warm_checkpoint_identity'])
    assert parent['step']==4800 and parent['case_exposure_counts']=={k:16 for k in s['TRAIN_keys']}
    complete_physical(parent['endpoints']['4800']['full_TRAIN'],s['TRAIN_keys'])
    assert s['parent_initial_FNO_backbone_identity']==parent['actual_initial_FNO_backbone_identity']
    assert s['parent_initial_FNO_backbone_identity']['passed']
    assert all(p['source_pins'].get(k)==v for k,v in own['source_pins'].items())
    from .operator_hash_training import balanced_schedule
    order=balanced_schedule(s['TRAIN_keys'],40,2026)
    assert len(order)==12000 and order[:4800]==parent_order
    assert d.digest(order)==s['full_order_digest'] and d.digest(order[:4800])==s['warm_parent_order_digest']==ts['full_order_digest']
    assert d.relative(p['output']).startswith('runs/geobc_exploration/macro_')
    return p,r,manifest,order
