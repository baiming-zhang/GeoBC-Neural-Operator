"""New actual recipe from frozen design and real terminals; metadata only."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.edgeconv_geometry_contract_v94 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT,DESIGN


def prepare():
    design,bp,old,bb,rb=inputs();ds=design['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    from geobc.macro_training import balanced_schedule
    science=dict(ds,design_identity=d.identity(DESIGN,hashed=True),design_scientific_digest=design['scientific_digest'],
        candidate='FNO9_EDGE_CONV',model_package='FNO9_EDGE_CONV',parameter_count_expected=4562122,
        expected_plain_model_initial_digest=old['initialization']['model_digest'],
        expected_operator_initial_digest=old['initialization']['shared_body_digests']['operator'],
        baseline_scalar_endpoints={'100':old['endpoints']['100']},optimizer=bp['scientific']['optimizer'],
        fixed_updates=900,case_exposures=100,endpoints_case_exposures=[100],
        full_order_digest=d.digest(balanced_schedule(ds['selected_TRAIN_keys'],100,ds['seed'])),
        physical_objective='full_original_node_physical_relative_U_S_squared_equal_case',
        memory_limit_bytes=30500000000,GPU_lock=ds['budget']['GPU_lock'],whole_campaign_budget_seconds=1200,
        cache_setup_budget_seconds=120,arm_budget_seconds=600,readout_budget_seconds_per_endpoint=180,
        TRAIN_gate=dict(stress_absolute_improvement_min=.03,primary_absolute_improvement_min=.03,field_deterioration_max=.02,hotspot_deterioration_max=.02),
        automatic100=False,automatic_replay=False)
    p=dict(kind='GeoBC_edgeconv_geometry_TRAIN9_v94',scientific=science,scientific_digest=d.digest(science),
        source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT,prepared_at_utc=d.utc(),full_order_digest=science['full_order_digest'])
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],
        source_pins=p['source_pins'],baseline_binding=bb,latest_holdout_binding=rb,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest'],new_GT_PT_arrays_GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
