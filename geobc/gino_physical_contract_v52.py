"""Fixed current GINO physical5 protocol; metadata-only release validation."""
from . import available_cad_pilot_data as d
from .global_bc_training_contract_v48 import terminal


def approved_contract(plan_path, require_release=True):
    p = d.read(plan_path)
    s, r = p['scientific'], d.read(p['root_review'])
    assert p['kind'] == 'GeoBC_GINO100_fullN_physical5_v52'
    assert r['passed'] and r['whole_source_reviewed']
    if require_release:
        assert r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins']
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert s['family'] == 'GINO_PHYSICAL' and s['seed'] == 2026
    assert s['train_CAD_count'] == 100 and s['passes'] == 5 and s['fixed_updates'] == 1500
    assert s['memory_limit_bytes'] == 30500000000
    assert s['GPU_lock'] == 'runs/geobc_exploration/.coordinator.lock'
    assert [s[k] for k in ('cache_setup_budget_seconds', 'qualification_budget_seconds',
        'training_budget_seconds', 'TRAIN_readout_budget_seconds', 'whole_budget_seconds')] == [600,180,3600,600,5400]
    assert s['loss'] == 'full_original_node_physical_relative_U_S_squared_equal_case'
    assert s['precision'] == 'highest_FP32_TF32_disabled'
    assert s['expected_parameter_count'] == 7478973 and s['expected_real_parameter_DOF'] == 14556861
    assert s['architecture'] == dict(official_commit='b307030d9b7ee530823d6a05bddb8238ba22a743',
        modes=[24,24,22], width=16, layers=4, resolution=64, radius=.033, query_chunk=2048, geometry_dim=32)
    assert s['optimizer'] == dict(name='Adam', learning_rate=.001, weight_decay=0.,
        betas=[.9,.999], eps=1e-8, foreach=True, fused=False)
    assert s['DEV_GT_reads_allowed'] == s['FINAL_reads_allowed'] == s['scaler_fits_allowed'] == 0
    assert not s['automatic40'] and not s['automatic24000'] and not s['automatic_replay']
    predecessor, _, a, _ = terminal('notes/geobc_original_TS_v51_plan.json')
    assert s['predecessor_terminal'] == predecessor and a['protocol_qualified']
    assert d.identity(s['dataset_identity']['path'], hashed=True) == s['dataset_identity']
    m = d.read(s['dataset_identity']['path'])
    keys = m['train_keys']['100']
    assert m['status'] == 'completed' and m['feasibility_pass']
    assert keys == s['TRAIN_keys'] and len(keys) == len(set(keys)) == 300
    assert m['scaler'] == s['scaler'] and all(m['case_roles'][k] == 'TRAIN' for k in keys)
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 5, 2026)
    assert len(order) == 1500 and d.digest(order) == s['full_order_digest']
    for name in ('geometry_inventory_identity', 'old_GINO_summary_identity'):
        assert d.identity(s[name]['path'], hashed=True) == s[name]
    inventory = d.read(s['geometry_inventory_identity']['path'])
    assert inventory['dataset_identity'] == s['dataset_identity']
    assert inventory['TRAIN_CADs'] == 100 and inventory['TRAIN_cases'] == 300
    assert inventory['existing_CSR_reusable_by_metadata'] == 40 and inventory['new_CAD_CSR_needed'] == 60
    old = d.read(s['old_GINO_summary_identity']['path'])
    assert old['status'] == 'completed' and old['endpoint_step'] == 480
    assert old['parameter_counts']['total'] == s['expected_parameter_count']
    assert old['config']['recipe']['fno_n_modes'] == s['architecture']['modes']
    assert old['first_update_gradient_check']['passed']
    oldg = old['config']['contract']['geometries']
    geometries = {}
    for row in inventory['per_CAD']:
        cad, item = row['cad'], row['current_xyz_identity']
        d.stat_matches(item)
        assert all(m['records'][cad+'/'+load]['file_identities']['xyz_path'] == item
                   for load in ('ver','hor','tor'))
        geometries[cad] = oldg[cad] if cad in oldg else dict(cad_id=cad,
            num_nodes=m['records'][cad+'/ver']['num_nodes'], xyz_path=str(d.path(item['path'])),
            xyz_stat={k:item[k] for k in ('bytes','mtime_ns')})
    assert len(geometries) == 100 and geometries == s['geometries']
    assert p['output'] == 'runs/geobc_exploration/GINO100_physical5_v52'
    return p, r, m, order, inventory
