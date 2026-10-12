"""Metadata-only release for one actual source RBE2 support port."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.support_translation_contract_v117 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW);assert sr['passed']and sr['whole_source_reviewed']and not sr['actual_execution_released']
    assert all(d.sha(k)==v for k,v in sr['source_pins'].items())and not any(d.path(k).exists()for k in (PLAN,REVIEW))
    science=dict(design['scientific'],implementation=dict(physics_basis_body_order='Oldforce6heldclosed until all12 new supportbasis+independentmixed fields barrier;readonce thenactual9directionAPI',
        nonzero_lift='Actualsource region0/fixed rows/optional existingmasterH5 row uniformtranslation;physicalfield keepslift notzero-boundary restore',
        true_residual_normalizer='Normactual b_free-Klift_free,including noexternalforce supportbasis case',
        support_virtualwork='Actual selectedreaction dot supportmotion plusexternal force nodalwork;masterlocalmomentunknownnull',
        dtype_namespace='torch.float64/float32/int64 to float64/float32/int64,no cast',
        old_operator_guards_repeated=0,new_port_qualification_actual=True))
    p=dict(kind='GeoBC_support_translation_port_v117',scientific=science,scientific_digest=d.digest(science),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
