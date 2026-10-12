"""Bind accepted physical six-basis terminal and immutable prototype bytes."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.located_all_support_U_contract_v119 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released'] and all(d.sha(k)==v for k,v in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s=dict(design['scientific'],implementation=dict(geometry_fixture_bary=design['scientific']['selection']['barycentric'],
        mixed_valid_plus_outside_extra_batches=2,mixed_batch_point_status_count=18,
        fixture_owner_role='Only geometryXYZ fixture generation;IDs withheld fromlocator/facade untilall48pred barrier/directreferences',
        physical_gradient_units='XYZmm/forceN/momentNmm/supportmm separately compared;no mixedunit norm',
        numpy_dtype_namespace='torch.float64/float32/int64 to float64/float32/int64 exactly;no casts'))
    p=dict(kind='GeoBC_located_all_support_U_response_v119',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
