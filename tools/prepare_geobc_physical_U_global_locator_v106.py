"""Bind predecessor and own-cache stat metadata without scientific-array reads."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.physical_U_global_locator_contract_v106 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==v for k,v in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s=dict(design['scientific'],implementation=dict(integer_dtype_namespace='torch.int64 maps exactly to NumPy int64, no cast',
        conservative_boundary_bucket='minimum exactly on bin boundary includes preceding bucket',
        owner_reference_policy='No reference ID/body read until all2304U/384owner/48outside immutable barrier'))
    p=dict(kind='GeoBC_fullCAD_physical_U_locator_v106',scientific=s,scientific_digest=d.digest(s),
           source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],
           source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
