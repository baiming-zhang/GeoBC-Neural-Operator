"""Shared100 physical-objective release; standard-library-only metadata gate."""
from . import available_cad_pilot_data as d

BASE = 'notes/geobc_macro100_PP256_NATIVE_5pass_v11_plan.json'


def approved_contract(plan_path):
    plan=d.read(plan_path); science=plan['scientific']; review=d.read(plan['root_review'])
    assert plan['kind']=='GeoBC_nativePP100_physical_short5_v13'
    assert review['passed'] and review['whole_source_reviewed'] and review['actual_execution_released']
    assert d.digest(science)==plan['scientific_digest']==review['scientific_digest']
    assert plan['source_pins']==review['source_pins']
    assert all(d.sha(name)==pin for name,pin in plan['source_pins'].items())
    assert science['family']=='PP256_NATIVE' and science['train_CAD_count']==100 and science['seed']==2026
    assert science['memory_limit_bytes']==30500000000 and science['GPU_lock']=='runs/geobc_exploration/.coordinator.lock'
    assert science['loss']=='full_original_node_physical_relative_U_S_squared_equal_case'
    assert science['optimizer']==dict(name='Adam',learning_rate=.001,weight_decay=0.,betas=[.9,.999],eps=1e-8,fused=True)
    assert science['fresh_initialization'] and science['decoder_chunk_size']==32768
    assert science['endpoint_passes']==[5] and science['fixed_updates']==1500
    assert science['cache_setup_budget_seconds']==600 and science['TRAIN_readout_budget_seconds']==300
    assert science['training_budget_seconds']==3600 and science['whole_budget_seconds']==4500
    assert not science['FINAL_reuse_for_selection'] and science['DEV_GT_reads_allowed']==0
    assert science['TRAIN_gate']==dict(stress_absolute_improvement_at_least=.05,U_absolute_deterioration_at_most=.02,fixed_denominator=300)
    item=science['design_reference']; assert d.identity(item['path'],hashed=True)==item
    item=science['dataset_identity']; assert d.identity(item['path'],hashed=True)==item
    manifest=d.read(item['path']); keys=manifest['train_keys']['100']
    assert science['TRAIN_keys']==keys and len(keys)==300 and science['scaler']==manifest['scaler']
    assert all(manifest['case_roles'][key]=='TRAIN' for key in keys)
    # Existing genuinely completed baseline; metadata only, including full300
    # physical readout, exact native/guard exits and immutable file identities.
    from tools.bind_geobc_macro100_v11 import terminal
    assert terminal(BASE)==science['baseline_terminal']
    baseline_plan=d.read(BASE); assert d.identity(BASE,hashed=True)==science['baseline_plan_identity']
    assert baseline_plan['scientific_digest']==science['baseline_scientific_digest']
    baseline=d.read(science['baseline_terminal']['summary']['path'])
    assert baseline['endpoints']['1500']['full_TRAIN']==science['baseline_TRAIN300']
    assert baseline['model_profile']['parameter_count']==4523268
    for key in ('seed','family','train_CAD_count','TRAIN_keys','dataset_identity','scaler','optimizer',
                'fresh_initialization','decoder_chunk_size','endpoint_passes','fixed_updates','full_order_digest'):
        assert science[key]==baseline_plan['scientific'][key]
    binding=science['FNO_physical_DEV_terminal']
    for kind in ('summary','external','status'): assert d.identity(binding[kind]['path'],hashed=True)==binding[kind]
    proof=d.read(binding['summary']['path']); ext=d.read(binding['external']['path'])
    assert proof==d.read(binding['status']['path']) and proof['status']=='completed' and proof['protocol_qualified']
    assert proof['closed'] and not proof['active'] and proof['DEV_gate']['passed']
    assert ext['closed'] and not ext['active'] and ext['exit_code']==0 and ext['within_external_budget']
    assert all(d.exact_process_exited(ext[k]) for k in ('guard_process','child_process'))
    assert proof['execution']['run_process']==ext['child_process'] and proof['execution']['guard']['guard_process']==ext['guard_process']
    assert proof['execution']['scientific_digest']==ext['scientific_digest']==science['FNO_physical_DEV_SCI']
    assert proof['committed_finite_predictions']==proof['owned_GT_array_reads']==60
    assert proof['raw_GT_reads']==proof['FINAL_numeric_reads']==proof['training_updates']==proof['scaler_fits']==0
    from .macro_training import balanced_schedule
    order=balanced_schedule(keys,5,2026); assert d.digest(order)==science['full_order_digest']
    assert plan['output']=='runs/geobc_exploration/macro100_PP256_NATIVE_physical5_v13'
    return plan,review,manifest,order
