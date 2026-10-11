"""Metadata-only release for one native++100CAD resident-cache witness."""
from . import available_cad_pilot_data as d


def approved_contract(plan_path):
    plan=d.read(plan_path);science=plan['scientific'];review=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_PP100_resident_feasibility_v10'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert plan['scientific_digest']==review['scientific_digest']==d.digest(science)
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(name)==value for name,value in plan['source_pins'].items())
    assert science['family']=='PP256_NATIVE' and science['seed']==2026
    assert science['train_CAD_count']==100 and science['memory_limit_bytes']==30500000000
    assert science['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert science['whole_budget_seconds']==600 and science['diagnostic_updates']==3
    assert science['loss']=='full_original_node_encoded4_case_mean_MSE'
    assert science['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert science['discard_diagnostic_model'] and science['physical_accuracy_claim']==False
    assert science['DEV_FINAL_GT_reads_allowed']==science['raw_CSV_H5_reads_allowed']==0
    item=science['dataset_identity'];assert d.identity(item['path'],hashed=True)==item
    manifest=d.read(item['path']);keys=manifest['train_keys']['100']
    assert science['TRAIN_keys']==keys and len(keys)==300
    assert all(manifest['case_roles'][key]=='TRAIN' for key in keys)
    assert science['scaler']==manifest['scaler']
    witness=max(keys,key=lambda k:manifest['records'][k]['num_nodes'])
    assert witness==science['witness_key']=='107_150/ver'
    assert manifest['records'][witness]['num_nodes']==science['witness_nodes']==325387
    # Genuine previous stage terminal, without loading its arrays or checkpoint.
    binding=review['FINAL_v9_terminal_binding']
    for kind in ('summary','external'):
        item=binding[kind];assert d.identity(item['path'],hashed=True)==item
    summary=d.read(binding['summary']['path']);external=d.read(binding['external']['path'])
    assert summary['status']=='completed' and summary['protocol_qualified']
    assert summary['closed'] and not summary['active']
    assert external['exit_code']==0 and external['within_external_budget'] and external['closed'] and not external['active']
    assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    assert plan['output']=='runs/geobc_exploration/PP100_resident_feasibility_v10'
    return plan,review,manifest,keys
