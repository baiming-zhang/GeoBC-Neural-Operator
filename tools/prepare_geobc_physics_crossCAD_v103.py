"""Bind actual fixed twoCAD protocol and existing six GT metadata only."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.physics_crossCAD_contract_v103 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();r=d.read(SOURCE_REVIEW)
    assert r['passed'] and r['whole_source_reviewed'] and not r['actual_execution_released'] and all(d.sha(k)==v for k,v in r['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    gt='runs/geobc_exploration/available_CAD20_candidate9/shared_GT/manifest.json';manifest=d.read(gt)
    assert all(cad+'/'+load in manifest['records']for cad in design['scientific']['selection']['CADs']for load in ('ver','hor','tor'))
    science=dict(design['scientific'],owned_GT_manifest_identity=d.identity(gt,hashed=True),design_identity=d.identity('notes/geobc_physics_crossCAD_paired_v103_design.json',hashed=True))
    p=dict(kind='GeoBC_twoCAD_paired_physics_v103',scientific=science,scientific_digest=d.digest(science),source_pins=r['source_pins'],root_review=REVIEW,output=OUTPUT)
    rr=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,rr);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
