"""Bind immutable caches and force-balance convention without body reads."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.reference_spring_contract_v115 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT

def prepare():
    design,b=inputs();sr=d.read(SOURCE_REVIEW);assert sr['passed']and sr['whole_source_reviewed']and not sr['actual_execution_released']
    assert all(d.sha(k)==v for k,v in sr['source_pins'].items())and not any(d.path(k).exists()for k in (PLAN,REVIEW))
    science=dict(design['scientific'],implementation=dict(force_balance='External applied+fixedreaction-groundspring Hbar.T Sn Hbar u;old residualwithout spring not equilibrium',
        endpoint_equation='op.original is modified scalarK;unmodifiedfineK notused forendpoint residual',
        pose_metric='[t_mm,L_fit theta_rad];translationmm/rotationrad separate',
        input_NPZ_role='Ownedaccepted targetfreefullinput cache including geometryfreefixed source arrays;no rawsource orlabels',
        preconditioner='Unchangedbase fixed32 additivecoarse,no modifiedcoarseassembly',
        readout_order='All6 zero modifiedequilibrium fields barrier beforecompact fullfield/referencepose comparison'))
    p=dict(kind='GeoBC_reference_spring_BC_v115',scientific=science,scientific_digest=d.digest(science),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        predecessor_binding=b,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    if p.parse_args().prepare:prepare()
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
