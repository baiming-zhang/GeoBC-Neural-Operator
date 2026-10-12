"""Metadata-only preparation, execution remains separately unreleased."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.load_amplitude_contract_v95 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT,DESIGN


def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s=dict(design['scientific'],design_identity=d.identity(DESIGN,hashed=True),raster_absolute_tolerance=.0001,force_absolute_tolerance=.000002,
        scoped_physics_limit='Original ignored DEBUG/PSOLID/numerical controls retained and effects not reconstructed; diagnose neural response against scoped fixed linear statics, not exact originalFEM fidelity or derived GT accuracy.')
    p=dict(kind='GeoBC_frozen3_targetfree_load_amplitude_v95',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest'],scientific_IO=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","scientific_IO":0,"GPU":0}')
