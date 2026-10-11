"""Shared TRAIN3 fitting release after handled native qualification failure."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal, check_identity

SOURCES = ('geobc/transolver_native_fit_v69.py',
           'geobc/transolver_native_fit_contract_v69.py',
           'tools/train_geobc_transolver_native_v69.py',
           'tools/run_geobc_transolver_native_guard_v69.py')


def inputs():
    p, a, e, b = terminal('notes/geobc_transolver_bf16_v65_plan.json')
    qp = d.read('notes/geobc_transolver_native_probe_v68_plan.json')
    tr = d.read('notes/geobc_transolver_native_v68_technical_review.json')
    assert tr['status'] == 'core_qualification_accepted_with_preserved_terminal_instrumentation_failure'
    assert tr['SCI'] == qp['scientific_digest'] and tr['source_pins'] == qp['source_pins']
    assert all(d.sha(k) == h for k, h in qp['source_pins'].items())
    o = d.path(qp['output'])
    qa, qe = d.read(o/'summary.json'), d.read(o/'external_terminal.json')
    assert qa == d.read(o/'status.json') and qa['status'] == 'failed' and qa['closed'] and not qa['active']
    assert qa['error_type'] == 'AssertionError' and qa['input_array_attempts'] == 12 and qa['input_array_reads'] == 24
    assert set(qa['cases']) == set(qp['scientific']['selected_TRAIN_keys'])
    assert all(x['passed'] and x['unchanged_parameters'] for x in qa['cases'].values())
    assert all(qa[k] == 0 for k in ('GT_array_reads','PT_body_reads','optimizer_updates','grid_input_reads'))
    assert qe['closed'] and not qe['active'] and qe['exit_code'] == 1
    assert qe['guard_error_type'] is None and not qe['external_timeout'] and qe['actual_child_exited']
    assert max(qe['elapsed_epoch_seconds'], qe['elapsed_monotonic_seconds']) < 600
    assert all(d.exact_process_exited(qe[k]) for k in ('child_process','guard_process'))
    for name, filename in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]:
        assert tr['terminal_binding'][name] == d.identity(o/filename, hashed=True)
    # Historical failure/reference metadata only; no checkpoint or truth body.
    old = d.read('notes/geobc_transolver_decoder_v64_plan.json')
    olda = d.read(d.path(old['output'])/'summary.json')
    olde = d.read(d.path(old['output'])/'external_terminal.json')
    return dict(parent=p, parent_summary=a, parent_external=e, parent_binding=b,
        qualification=qp, technical=tr,
        technical_review_identity=d.identity('notes/geobc_transolver_native_v68_technical_review.json', hashed=True),
        fp32_summary=olda, fp32_external=olde,
        fp32_binding=d.read(p['root_review'])['upstream_terminal_bindings']['fp32'],
        shared_binding=d.read(p['root_review'])['upstream_terminal_bindings']['shared'])


def approved_contract(plan_path):
    p = d.read(plan_path)
    s, r, up = p['scientific'], d.read(p['root_review']), inputs()
    old = up['parent']['scientific']
    assert p['kind'] == 'GeoBC_originalTS_BF16_native_same_decoder_shared3_v69'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert p['scientific_digest'] == r['scientific_digest'] == d.digest(s)
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    assert r['upstream_terminal_binding'] == up['parent_binding']
    assert r['qualification_technical_review_identity'] == up['technical_review_identity']
    for k in ('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock',
              'memory_limit_bytes','decoder_chunk_size','selected_TRAIN_keys',
              'whole_budget_seconds','training_budget_seconds','cache_setup_budget_seconds',
              'representative_field','visualization_sample_nodes_max','updates_per_field',
              'total_updates_per_family','total_updates','case_pass_endpoints','full_order_digest',
              'expected_reference_initialization','expected_FP32_TS_initialization','official_TS_kwargs'):
        assert s[k] == old[k]
    assert s['families'] == ['TS_ORIGINAL_BF16_NATIVE_HASH']
    assert s['memory_limit_bytes'] == 30500000000 and s['training_budget_seconds'] == 900
    assert s['whole_budget_seconds'] == 1200 and not s['automatic_continuation']
    assert s['parent_checkpoint_reads_allowed'] == s['DEV_FINAL_reads_allowed'] == 0
    assert not s['old_budget_reset'] and not s['offnode_runtime_qualified']
    m = d.read(check_identity(s['dataset_identity']))
    assert len(s['selected_TRAIN_keys']) == 3
    assert all(m['case_roles'][k] == 'TRAIN' and k in m['train_keys']['100'] for k in s['selected_TRAIN_keys'])
    assert p['output'] == 'runs/geobc_exploration/Transolver_native_continuous_shared3_v69'
    return p, r, m, up
