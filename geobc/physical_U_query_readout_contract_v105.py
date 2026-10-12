"""CPU-only namespace readout recovery; original scientific protocol unchanged."""
from . import available_cad_pilot_data as d
from .transolver_decoder_contract_v64 import check_identity
PLAN='notes/geobc_physical_U_query_readout_v105_plan.json'
REVIEW='notes/geobc_physical_U_query_readout_v105_root_review.json'
SOURCE_REVIEW='notes/geobc_physical_U_query_readout_v105_source_review.json'
OUTPUT='runs/geobc_exploration/physical_U_query_CPU_readout_v105'
SOURCES=('geobc/physical_U_query_readout_contract_v105.py','tools/readout_geobc_physical_U_query_v105.py','tools/guard_geobc_physical_U_query_readout_v105.py')

def approved_contract(path):
    p=d.read(path);r=d.read(p['root_review']);old=d.read(check_identity(p['original_plan_identity']));a=d.read(check_identity(p['original_terminal_binding']['summary']));e=d.read(check_identity(p['original_terminal_binding']['external']))
    assert p['kind']=='GeoBC_physical_U_query_CPU_readout_v105' and p['output']==OUTPUT and p['scientific']==old['scientific'] and p['scientific_digest']==old['scientific_digest']==d.digest(p['scientific'])==r['scientific_digest']
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released'] and r['source_review_identity']==d.identity(SOURCE_REVIEW,hashed=True)
    assert p['source_pins']==r['source_pins'] and set(SOURCES)<=set(p['source_pins']) and all(d.sha(k)==v for k,v in p['source_pins'].items())
    assert a['status']=='failed' and a['closed'] and not a['active'] and a['error_type']=='AssertionError' and a['committed_interior_queries']==2304 and a['gradient_query_points']==192
    assert e['closed'] and not e['active'] and e['exit_code']==1 and e['actual_child_exited'] and d.exact_process_exited(e['child_process']) and d.exact_process_exited(e['guard_process'])
    assert a['input_cache_reads']==3 and a['canonical_basis_array_reads']==9 and a['nodal_witness_positions']==960 and a['nodal_canonical_vectors']==2880
    assert all(v['passed'] and all(q['passed']for q in v['qualification'].values())for v in a['case_records'].values())
    assert p['original_guard_deadline_epoch']==e['deadline_epoch'] and p['original_guard_deadline_monotonic']==e['deadline_monotonic']
    for item in a['prediction_inventory'].values():
        record=d.read(check_identity(item))
        for name in ('physical_U','query_xyz','query_cell_rows'):d.stat_matches(record[name])
    return p,r,a,e
