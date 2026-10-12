"""Metadata-only freeze; no rawdeck/model/array/GT body reads."""
import argparse
from geobc import available_cad_pilot_data as d
from geobc.load_linear_knownDEV_contract_v97 import inputs,PLAN,REVIEW,SOURCE_REVIEW,OUTPUT,DESIGN


def prepare():
    design,pp,pb,m=inputs();sr=d.read(SOURCE_REVIEW)
    assert sr['passed']and sr['whole_source_reviewed']and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())and not d.path(PLAN).exists()and not d.path(REVIEW).exists()
    source=d.read('notes/geobc_available_cad_pilot_plan.json');cads={k.split('/')[0]for k in design['scientific']['case_keys']};decks={}
    for row in source['scientific']['selected_source_receipt_index']:
        if row['cad_id']in cads:
            x=row['components']['SolverDeck'];item=dict(path=x['path'],bytes=x['recorded_file_stat']['bytes'],mtime_ns=x['recorded_file_stat']['mtime_ns'],sha256=x['observed_sha256']);d.stat_matches(item);decks[row['cad_id']]=item
    assert set(decks)==cads
    prior=d.read('notes/geobc_load_amplitude_homogeneity_v95_plan.json')
    s=dict(design['scientific'],design_identity=d.identity(DESIGN,hashed=True),raw_deck_identities=decks,deck_reader_policy=prior['scientific']['deck_reader_policy'],force_absolute_tolerance=2e-6,raster_absolute_tolerance=1e-4,query_chunk_encoded_tolerance_FP32=3e-5,query_chunk_encoded_tolerance_TCNN_BF16=.002,qualification_in_span_coefficients=[.5,.25,.75],canonical_U_relative_tolerance=2e-5,canonical_encodedS_FP32_tolerance=3e-5,canonical_encodedS_BF16_tolerance=.002)
    p=dict(kind='GeoBC_load_linear_knownDEV_v97',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,output=OUTPUT)
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],predecessor_binding=pb,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='metadata_prepared_actual0',SCI=p['scientific_digest'],PT_GT_arrays_GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
