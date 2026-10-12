"""Freeze metadata identities and explicit metric coordinates; no array reads."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.reference_motion_contract_v113 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();s=design['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==v for k,v in sr['source_pins'].items())
    assert not any(d.path(k).exists()for k in (PLAN,REVIEW))
    science=dict(s,implementation=dict(pose_metric_coordinates='[translation_mm,L_fit*rotation_rad]',
        Jacobian_metric='D_fit J D_fit vs D_fit rawC D_fit;same mm/N',
        energy_gradient_metric='D_fit times physical force gradient vs D_fit physical motion',
        reference_point_scope='Explicit RBE derived output DOF,not a claim its XYZ is outside all cells',
        fixed_gradient_points='first two declared mixed force6 requests per CAD',
        numpy_dtype_namespace='torch.float64/float32/int64 to float64/float32/int64;no cast'))
    p=dict(kind='GeoBC_reference_motion_response_v113',scientific=science,scientific_digest=d.digest(science),
        source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],
        source_pins=p['source_pins'],predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
