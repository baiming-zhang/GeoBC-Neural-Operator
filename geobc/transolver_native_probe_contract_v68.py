"""Target-free whole-node native-TS qualification; metadata only."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity

UPSTREAMS = ('notes/geobc_transolver_macro100_v66_plan.json',
             'notes/geobc_transolver_macro100_readout_v67_plan.json')
SOURCES = ('geobc/transolver_native_hash_v68.py',
           'geobc/transolver_native_probe_contract_v68.py',
           'tools/probe_geobc_transolver_native_v68.py',
           'tools/run_geobc_transolver_native_probe_guard_v68.py')


def approved_contract(plan_path):
    p = d.read(plan_path)
    s, r = p['scientific'], d.read(p['root_review'])
    assert p['kind'] == 'GeoBC_originalTS_native_fullN_targetfree_qualification_v68'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest'] == r['scientific_digest'] == d.digest(s)
    assert p['source_pins'] == r['source_pins']
    assert set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    for up in UPSTREAMS:
        _, _, _, binding = terminal(up)
        assert r['upstream_terminal_bindings'][up] == binding
    m = d.read(check_identity(s['dataset_identity']))
    keys = m['train_keys']['100']
    expected = [max((k for k in keys if k.endswith('/'+load)),
                    key=lambda k: (m['records'][k]['num_nodes'], k))
                for load in ('ver', 'hor', 'tor')]
    assert s['selected_TRAIN_keys'] == expected and len(expected) == 3
    assert all(m['case_roles'][k] == 'TRAIN' for k in expected)
    assert s['memory_limit_bytes'] == 30500000000 and s['whole_budget_seconds'] == 600
    assert s['GT_reads_allowed'] == s['optimizer_updates_allowed'] == s['PT_reads_allowed'] == 0
    assert s['decoder_chunk_size'] == 32768
    assert s['fresh_initial_model_digest'] == 'e752267992414fba8bf81b22486640d266671626b9957904e6f6a71f61c09f16'
    assert not s['automatic_training_release'] and not s['offnode_runtime_qualified']
    assert p['output'] == 'runs/geobc_exploration/Transolver_native_targetfree_fullN_v68'
    return p, r, m, None
