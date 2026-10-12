"""Independent zero-update dimension-valid recovery, explicit changed skip science."""
import time
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import check_identity, terminal

PLAN = 'notes/geobc_official_UNO_valid_skip_recovery_v91_plan.json'
OUTPUT = 'runs/geobc_exploration/Official_UNO_TRAIN9_valid_skip_v91'
SOURCES = ('geobc/official_uno_valid_skip_v91.py', 'geobc/official_uno_valid_skip_contract_v91.py',
    'tools/train_geobc_official_UNO_valid_skip_v91.py', 'tools/run_geobc_official_UNO_valid_skip_guard_v91.py')


def approved_contract(plan_path):
    p = d.read(plan_path); s = p['scientific']; r = d.read(p['root_review'])
    assert p['output'] == OUTPUT and r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s) == p['scientific_digest'] == r['scientific_digest']
    assert p['source_pins'] == r['source_pins'] and set(SOURCES) <= set(p['source_pins'])
    assert all(d.sha(k) == h for k, h in p['source_pins'].items())
    old = d.read(check_identity(p['original_plan_identity']))
    amended=dict(old['scientific']);amended['UNO_kwargs']=dict(amended['UNO_kwargs'],channel_mlp_skip='linear');amended['candidate']='OFFICIAL_UNO_VALID_SKIP_TRAIN9'
    assert s==amended and p['scientific_digest']!=old['scientific_digest']
    assert p['scientific_amendment']=={'UNO_kwargs.channel_mlp_skip':['soft-gating','linear'],'candidate':['OFFICIAL_UNO_TRAIN9','OFFICIAL_UNO_VALID_SKIP_TRAIN9']}
    assert all(p['source_pins'][k] == h for k, h in old['source_pins'].items())
    a = d.read(check_identity(p['prior_native_identity'])); e = d.read(check_identity(p['prior_external_identity']))
    f = d.read(check_identity(p['failure_receipt_identity']))
    assert a['status'] == 'failed' and a['closed'] and not a['active'] and a['cache_receipt'] is None
    assert a['optimizer_step_attempts'] == a['optimizer_step_completions'] == 0 and not a['arms']
    assert a['error_type'] == 'ModuleNotFoundError' and a['error'] == "No module named 'neuralop'"
    assert e['exit_code'] == 1 and e['closed'] and not e['active'] and e['actual_child_exited']
    assert f['exact_processes_exited'] and f['GT_arrays_reads'] == 0 and not f['model_Adam_RNG_checkpoints_exist']
    assert all(d.exact_process_exited(e[k]) for k in ('guard_process', 'child_process'))
    assert not list(d.path(old['output']).glob('*/step*.pt')) and not list(d.path(old['output']).glob('*/last.pt'))
    second=d.read(check_identity(p['prior_second_native_identity']));second_e=d.read(check_identity(p['prior_second_external_identity']))
    second_f=d.read(check_identity(p['second_failure_receipt_identity']))
    assert second['status']=='failed' and second['closed'] and not second['active'] and second['cache_receipt'] is None
    assert second['optimizer_step_attempts']==second['optimizer_step_completions']==0 and not second['arms']
    assert second['error_type']=='ValueError' and 'same for soft-gating' in second['error']
    assert second_e['exit_code']==1 and second_e['closed'] and not second_e['active']
    assert second_f['GT_arrays_reads']==0 and second_f['exact_processes_exited']
    assert all(d.exact_process_exited(second_e[k]) for k in ('guard_process','child_process'))
    assert time.time() < e['deadline_epoch'], 'Original absolute budget expired; no reset'
    _, _, _, b = terminal('notes/geobc_local_spatial_operator_v88_plan.json')
    assert r['prior_completed_v88_binding'] == b
    m = d.read(check_identity(s['dataset_identity'])); keys = s['selected_TRAIN_keys']
    assert len(keys) == len(set(keys)) == 9 and all(m['case_roles'][k] == 'TRAIN' and k in m['train_keys']['100'] for k in keys)
    assert m['scaler'] == s['scaler']
    from .macro_training import balanced_schedule
    order = balanced_schedule(keys, 100, s['seed']); assert d.digest(order) == p['full_order_digest'] == s['full_order_digest']
    return p, r, m, order, p['failure_receipt_identity']
