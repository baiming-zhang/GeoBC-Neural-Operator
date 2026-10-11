"""Fixed five-pass100CAD macro pair, with actual native-resource evidence."""
from . import available_cad_pilot_data as d


def approved_contract(plan_path):
    plan=d.read(plan_path);science=plan['scientific'];review=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_macro100_short5_v11'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert plan['scientific_digest']==review['scientific_digest']==d.digest(science)
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(name)==value for name,value in plan['source_pins'].items())
    family=science['family'];assert family in ('PP256_NATIVE','FNO9')
    assert science['seed']==2026 and science['train_CAD_count']==100
    assert science['memory_limit_bytes']==30500000000
    assert science['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert science['loss']=='full_original_node_encoded4_case_mean_MSE'
    assert science['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert science['fresh_initialization'] and science['decoder_chunk_size']==32768
    assert science['endpoint_passes']==[5] and science['fixed_updates']==1500
    assert science['cache_setup_budget_seconds']==science['TRAIN_readout_budget_seconds']==300
    assert science['training_budget_seconds']==(3600 if family=='PP256_NATIVE' else 900)
    assert science['whole_budget_seconds']==(4200 if family=='PP256_NATIVE' else 1500)
    assert not science['FINAL_reuse_for_selection'] and science['DEV_GT_reads_allowed']==0
    item=science['dataset_identity'];assert d.identity(item['path'],hashed=True)==item
    manifest=d.read(item['path']);keys=manifest['train_keys']['100']
    assert science['TRAIN_keys']==keys and len(keys)==300 and science['scaler']==manifest['scaler']
    assert all(manifest['case_roles'][key]=='TRAIN' for key in keys)
    binding=review['PP100_feasibility_terminal_binding']
    for kind in ('summary','external'):
        item=binding[kind];assert d.identity(item['path'],hashed=True)==item
    summary=d.read(binding['summary']['path']);external=d.read(binding['external']['path'])
    assert summary['status']=='completed' and summary['feasibility_qualified'] and summary['actual_largest_fullN_gradient_passed']
    assert summary['closed'] and not summary['active'] and summary['completed_updates']==3
    assert summary['cache_receipt']['loaded_cases']==300
    assert external['exit_code']==0 and external['within_external_budget'] and external['closed'] and not external['active']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    if family=='FNO9':
        # Release the second arm only after the first genuine terminal.
        binding=review['PP100_short5_terminal_binding']
        for kind in ('summary','external'):
            item=binding[kind];assert d.identity(item['path'],hashed=True)==item
        summary=d.read(binding['summary']['path']);external=d.read(binding['external']['path'])
        assert summary['status']=='completed' and summary['training_complete'] and summary['step']==1500
        assert summary['closed'] and not summary['active']
        assert external['exit_code']==0 and external['within_external_budget'] and external['closed'] and not external['active']
        assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,5,2026);assert science['full_order_digest']==d.digest(order)
    assert plan['output']=='runs/geobc_exploration/macro100_'+family+'_5pass_v11'
    return plan,review,manifest,order
