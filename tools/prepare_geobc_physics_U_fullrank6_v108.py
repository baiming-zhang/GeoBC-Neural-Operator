"""Bind strict cached K/coarse state and genuine full20 coverage terminal."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.physics_U_fullrank6_contract_v108 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released'] and all(d.sha(k)==v for k,v in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    p=dict(kind='GeoBC_physical_U_fullrank6_v108',scientific=design['scientific'],scientific_digest=design['scientific_digest'],source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
