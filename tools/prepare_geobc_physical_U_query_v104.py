"""Bind exact previous physics terminal, cached basis and reference frames."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.physical_U_query_contract_v104 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();r=d.read(SOURCE_REVIEW)
    assert r['passed'] and r['whole_source_reviewed'] and not r['actual_execution_released'] and all(d.sha(k)==v for k,v in r['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    old=d.read('runs/geobc_exploration/fixedCAD_sixload_independent_P2_v100/summary.json');new=d.read(b['summary']['path'])
    costs={'120_153':dict(original_GPU_parent_seconds=old['elapsed_epoch_seconds'],source_K_seconds=old['input_operator_seconds'],canonical_basis_group_seconds=old['completed_groups']['0']['wall_seconds'],old_mixed_group_seconds=old['completed_groups']['1']['wall_seconds'],whole_cold_deployment_components_unknown=True,original_failure='dtype readout namespace failure;physics six solves alreadyconverged;CPU recovery kept separately')}
    for cad in ('131_345','131_351'):
        row=new['case_records'][cad];costs[cad]=dict(source_K_seconds=row['source_operator_wall_seconds'],canonical_basis_group_seconds=row['groups']['block_canonical']['wall_seconds'],coarse_alternative_construct_plus_canonical_seconds=row['paired_primary_coarse_seconds'],whole_cold_deployment_components_unknown=True)
    s=dict(design['scientific'],prototype_identity=d.identity('geobc/physical_U_query_v104.py',hashed=True),inherited_actual_basis_costs=costs,reference_frame_policy='API dr=(b0,b1,b2),oldNumPy derivative dr=(b1,b2,b3);compare physicalJacobian gradient after actual frame-specific inverse;N weights use samebarycentric order',true_endpoint_denominators=dict(CADs=3,interior_queries=2304,nodal_witness_positions=960,nodal_canonical_vectors=2880,gradient_query_points=192))
    p=dict(kind='GeoBC_true_physical_U_query_API_v104',scientific=s,scientific_digest=d.digest(s),source_pins=r['source_pins'],root_review=REVIEW,output=OUTPUT)
    rr=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,rr);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
