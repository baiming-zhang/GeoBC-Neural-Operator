"""Freeze actual physics recipe extension; no arrays, GT or CUDA imported."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.load_linear_physics_contract_v100 import inputs,PLAN,REVIEW,SOURCE_REVIEW,DESIGN,OUTPUT


def prepare():
    design,pp,pb=inputs();r=d.read(SOURCE_REVIEW)
    assert r['passed']and r['whole_source_reviewed']and not r['actual_execution_released']and all(d.sha(k)==v for k,v in r['source_pins'].items())
    assert not d.path(PLAN).exists()and not d.path(REVIEW).exists()
    m=d.read('data/geobc_available_cad_pilot/manifest.json');cad=design['scientific']['selection']['CAD'];row=m['CADs'][cad];gid=row['geometry']['grid_node_ids'];d.stat_matches(gid)
    s=dict(design['scientific'],design_identity=d.identity(DESIGN,hashed=True),GRID_map_identity=gid,GRID_cache_scope='One target-free H5row->actualGRID map from existing source-preparation;no neuralfeature/cloud/grid orGT values',expected_geometry=dict(nodes=row['num_nodes'],cells=row['cell_count'],characteristic_length_mm=row['normalization']['characteristic_length_mm']),parallel_RHS_groups=[['ver','hor','tor'],['mix0','mix1','mix2']],parallel_operator_qualification=dict(actual_fullN_lanes=3,old_reference_K_applications=3,new_batched_K_applications=1,relative_tolerance=1e-9,seed=2026,no_extra_repeat_of_original8guards=True),checkpoint_policy='Atomic overwrite last_state.npz every25 per group with all3 independent z/r/p/rz/active/iterations;statecost/pin/identity traced,old completed groups retained',per_group_terminal_reserve_seconds=10,parallel_cost_semantics='Exclusive actual wall for each concurrently solved3 RHS group;each group including final fields limited300;two group walls sum within900. Per-lane time not summed as sequential cost.')
    p=dict(kind='GeoBC_independent_P2_sixload_v100',scientific=s,scientific_digest=d.digest(s),source_pins=r['source_pins'],root_review=REVIEW,output=OUTPUT)
    review=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=pb,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,review);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest'],arrays_GT_GPU=0))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true')
    if a.parse_args().prepare:prepare()
    else:print('{"status":"source_only","GT_arrays_GPU":0}')
