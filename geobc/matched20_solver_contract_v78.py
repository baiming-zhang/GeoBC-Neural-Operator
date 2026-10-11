"""Metadata-only release for matched20 parent continuations, not restarts."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import terminal,check_identity,complete_physical

PLAN='notes/geobc_matched20_solver_v78_plan.json'
REVIEW='notes/geobc_matched20_solver_v78_root_review.json'
DESIGN='notes/geobc_matched20_solver_v78_design.json'
SOURCES=('geobc/matched20_solver_contract_v78.py','tools/train_geobc_matched20_solver_v78.py',
    'tools/run_geobc_matched20_solver_guard_v78.py')


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review']);design=d.read(DESIGN)
    assert p['kind']=='GeoBC_two_existing_solvers_matched20_continuation_v78'
    assert s==design['scientific'] and d.digest(s)==p['scientific_digest']==design['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert r['scientific_digest']==p['scientific_digest'] and r['source_pins']==p['source_pins']
    assert set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==h for k,h in p['source_pins'].items())
    sp,sa,se,sb=terminal('notes/geobc_decoder_packages_v77_plan.json')
    assert sa['protocol_qualified'] and sa['TRAIN_gates']['FNO9_TCNN_query']['passed']
    decision=d.read(check_identity(design['basis_actual_v77_decision_identity']))
    assert decision['PNG_visually_verified'] and decision['terminal_binding']==sb
    bindings={'screen77':sb};parents={}
    from .macro_training import balanced_schedule
    keys=s['TRAIN_keys'];order=balanced_schedule(keys,20,s['seed'])
    assert d.digest(order)==p['full_order_digest'] and len(keys)==len(set(keys))==300 and len(order)==6000
    for arm in s['arms']:
        pp,aa,ee,bb=terminal(arm['parent_plan']);ps=pp['scientific']
        assert pp['scientific_digest']==arm['parent_SCI'] and ps['dataset_identity']==s['dataset_identity'] and ps['scaler']==s['scaler'] and ps['TRAIN_keys']==keys
        assert d.digest(order[:arm['parent_step']])==ps['full_order_digest']
        assert ps['optimizer']==arm['parent_optimizer'] and ps['loss']==arm['parent_loss'] and ps['precision']==arm['precision']
        assert arm['parent_case_exposure_counts']=={k:arm['parent_passes'] for k in keys}
        if arm['family']=='FNO_GLOBAL_HASH':
            assert aa['step']==arm['parent_step']==3000 and aa['case_exposure_counts']==arm['parent_case_exposure_counts']
            assert aa['endpoints']['3000']['checkpoint_identity']==arm['parent_checkpoint_identity']
            complete_physical(aa['endpoints']['3000']['full_TRAIN'],keys)
            bindings['Hash10']=bb
        else:
            assert arm['family']=='TS_ORIGINAL_BF16_GLOBAL_HASH' and arm['parent_step']==1500
            state=aa['models'][arm['family']]
            assert state['optimizer_step']==1500 and state['case_exposure_counts']==arm['parent_case_exposure_counts']
            assert state['checkpoint_identity']==arm['parent_checkpoint_identity']
            complete_physical(aa['aggregate_endpoints'][arm['family']]['5'],keys)
            bindings['TS5']=bb
        d.stat_matches(arm['parent_checkpoint_identity'])
        parents[arm['name']]=dict(plan=pp,summary=aa,external=ee,binding=bb)
    fp,fa,fe,fb=terminal(s['baseline_FNO20_plan_identity']['path']);bindings['FNO20']=fb
    assert d.identity(s['baseline_FNO20_plan_identity']['path'],hashed=True)==s['baseline_FNO20_plan_identity']
    assert fb==s['baseline_FNO20_terminal_binding'] and fp['scientific']['TRAIN_keys']==keys
    complete_physical(fa['endpoints']['6000']['full_TRAIN'],keys)
    assert r['upstream_terminal_bindings']==bindings==design['upstream_terminal_bindings']
    assert s['fixed_final_step']==6000 and s['fixed_final_passes']==20 and s['memory_limit_bytes']==30500000000
    assert not s['automatic40'] and not s['automatic_replay'] and not s['FINAL_reuse_for_selection']
    m=d.read(check_identity(s['dataset_identity']));assert m['train_keys']['100']==keys and m['scaler']==s['scaler']
    assert p['output']=='runs/geobc_exploration/Hash_TS_matched20_v78'
    return p,r,m,order,parents
