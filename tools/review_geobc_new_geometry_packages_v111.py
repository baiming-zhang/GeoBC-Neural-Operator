"""Root metadata review of real freshly solved packages; no scientific body loads."""
import argparse,time
from pathlib import Path
from geobc import available_cad_pilot_data as d
from geobc.physical_U_new_geometry_contract_v111 import approved_contract

def review(path):
    p,_=approved_contract(path);s=p['scientific'];o=d.path(p['output']);q=d.read(o/'package_proposal.json');a=d.read(o/'status.json')
    assert q['SCI']==p['scientific_digest'] and q['source_pins']==p['source_pins']
    assert a['status']=='running' and a['active'] and not a['closed'] and a['phase']=='awaiting_independent_package_root_release'
    assert a['field_barrier_complete'] and a['numeric_gate_passed'] and a['committed_finite_endpoints']==18
    assert a['GT_reads']==a['raw_CSV_H5_nodal_target_reads']==a['neural_PT_reads']==a['loader_calls']==0
    execution=a['execution'];outer=execution['guard']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==outer['boot_id']
    assert min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())>0
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    assert d.process_snapshot(execution['run_process']['pid'])==execution['run_process']
    assert q['field_barrier_identity']==d.identity(o/'physics_prediction_barrier.json',hashed=True)
    assert set(q['packages'])==set(s['selection']['CADs']);digests={}
    for cad,pkg in q['packages'].items():
        v=pkg['payload'];case=a['case_records'][cad];n=s['cases'][cad]['expected_geometry']['nodes']
        assert pkg['schema']=='GeoBC_physical_U_response_package_v1' and v['CAD']==cad
        assert pkg['payload_digest']==d.digest(v) and v['basis_order']==s['selection']['basis_names']
        assert v['geometry_identity']==case['input_identity'] and v['geometry_schema']==s['cases'][cad]['expected_geometry']
        d.stat_matches(v['geometry_identity'])
        assert v['reference_load6_columns']==case['reference_load6_columns']
        assert v['scope']['REFC']=='123456' and v['scope']['material_constraints_fixed']
        assert v['scope']['reference_GRID_ID']==case['source_facts']['original']['load_guard']['reference_GRID_ID']
        assert v['scope']['physical_U_units']=='mm' and v['scope']['force_units']=='N' and v['scope']['moment_units']=='Nmm'
        assert not v['new_OptiStruct_GT_accuracy'] and not v['signedS_provided']
        assert case['parallel_K_qualification']['passed'] and case['parallel_M_qualification']['passed'] and case['six_transfer_qualification']['rank']==6
        for name,entry in v['basis_records'].items():
            role='canonical_basis'if name in ('ver','hor','tor')else'added_basis'
            identity=a['prediction_inventory'][cad+'/'+role+'/'+name]
            assert entry['manifest_identity']==identity and d.identity(identity['path'],hashed=True)==identity
            rec=d.read(identity['path']);assert rec['key']==name and rec['initial_guess']=='independent_zero'
            assert rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8 and rec['all_original_nodes']==n
            assert rec['physical_U']==entry['physical_U'] and rec['physical_U']['shape']==[n,3] and rec['physical_U']['dtype']=='torch.float64'
            d.stat_matches(rec['physical_U'])
        digests[cad]=d.digest(v)
    release=o/'package_release.json';assert not release.exists()
    d.write(release,dict(passed=True,reviewer='root',reviewed_at_utc=d.utc(),SCI=p['scientific_digest'],source_pins=p['source_pins'],
        proposal_identity=d.identity(o/'package_proposal.json',hashed=True),field_barrier_identity=q['field_barrier_identity'],
        reviewed_payload_digests=digests,real_process_snapshots=dict(guard=outer['guard_process'],child=execution['run_process']),
        original_absolute_deadlines_unchanged=True,scientific_body_reads=0,actual_GPU_API_execution=0))
    print(dict(status='actual_fresh_packages_metadata_root_released',CADs=list(digests),GT_arrays_GPU_reads=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.review:
        if not args.plan:p.error('--review requires --plan')
        review(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
