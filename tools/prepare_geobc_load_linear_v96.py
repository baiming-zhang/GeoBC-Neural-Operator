"""Metadata-only recipe binding; no prototype import or model/data bodies."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.load_linear_contract_v96 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT,DESIGN


def prepare():
    design,pp,pa,pb=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed']and sr['whole_source_reviewed']and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())and not d.path(PLAN).exists()and not d.path(REVIEW).exists()
    s=dict(design['scientific'],design_identity=d.identity(DESIGN,hashed=True),owned_v95_inventory=pa['prediction_inventory'],
        coefficient_gradient_scope='Three actual in-span coefficients; f6=B@c. No claim of independent sixDOF learned/verified response.',
        query_chunk_encoded_tolerance_FP32=.00003,query_chunk_encoded_tolerance_TCNN_BF16=.002,
        qualification_in_span_coefficients=[.5,.25,.75])
    p=dict(kind='GeoBC_reusable_load_linear_API_v96',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=pb,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest'],new_GT_PT_arrays_GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
