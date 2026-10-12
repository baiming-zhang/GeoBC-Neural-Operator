"""Metadata-only source bindings; generated packages require later real root review."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.physical_U_new_geometry_contract_v111 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==v for k,v in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    science=dict(design['scientific'],implementation=dict(
        canonical_prepare_inputs_mix3=[[1,.5,0],[.5,0,1],[0,1,-.5]],
        package_release='After all18field finitehashbody barrier and numericalgate:root independently reviews actual descriptor metadata/pins/newmanifests/identity and liveprocess before unchanged110loader. Waiting consumes originalwhole deadline.',
        descriptor_scope='Runtime generated freshowngeometry/sixbasis records;expectedpayload digests supplied by genuine external rootrelease,never descriptor selfclaim',
        original_source_to_first_ready_cost='Includes parsing,newKcoarsequal,sixbasis persistence;independent3mixedvalidation+rootwait+cleanload explicitly measured separately,whole includes all',
        namespace='torch.float64/float32/int64 to float64/float32/int64;strictpathbyteSHAshape/dtype,no cast'))
    p=dict(kind='GeoBC_physical_U_new_geometry_deployment_v111',scientific=science,scientific_digest=d.digest(science),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
