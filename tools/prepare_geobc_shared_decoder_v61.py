"""Metadata-only preparation of a matched shared three-field diagnostic."""
import argparse
import datetime

PLAN='notes/geobc_shared_decoder_v61_plan.json'
REVIEW='notes/geobc_shared_decoder_v61_root_review.json'
SOURCE_REVIEW='notes/geobc_shared_decoder_v61_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.shared_decoder_pair_contract_v61 import inputs,SOURCES,MODEL_SOURCES,order
    up=inputs();ps=up['plan']['scientific'];sr=d.read(SOURCE_REVIEW)
    proposal=d.read('notes/geobc_shared_decoder_v61_proposal.json')
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES+MODEL_SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert proposal['keys']==up['keys'] and proposal['upstream_terminal_binding']==up['binding']
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    names=('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes',
        'decoder_chunk_size','parent_TRAIN_keys','selected_TRAIN_keys','families','parameter_counts','runtime_factory',
        'whole_budget_seconds','training_budget_seconds','cache_setup_budget_seconds','representative_field','visualization_sample_nodes_max')
    s={k:ps[k]for k in names}
    s.update(fresh_models_per_family=1,updates_per_field=1000,total_updates_per_family=3000,total_updates=6000,
        case_pass_endpoints=[0,100,1000],fresh_initialization=True,parent_checkpoint_reads_allowed=0,
        expected_actual_initialization=up['summary']['model_initialization'],independent_endpoint_reference=up['summary']['aggregate_endpoints'],
        full_order_digest=d.digest(order(up['keys'])),DEV_FINAL_reads_allowed=0,automatic_continuation=False,FINAL_reuse_for_selection=False,
        proposal_identity=d.identity('notes/geobc_shared_decoder_v61_proposal.json',hashed=True),
        limits=proposal['limits'],scientific_scope='One shared model per family on selected TRAIN3; not unseenCAD or finalsolver ranking',
        RNG_policy='Fresh model/Adam0/fourRNG matching ownv60 initial bytes; all3 real gradient checks restore RNG; seeded balanced1000passes; readout restores RNG',
        visualization_policy='Fixed same original node sample and camera; truth99percentile symmetric stress range shared for visibility, explicitly clipped; metrics all original nodes')
    p=dict(kind='GeoBC_fresh_shared_three_decoder_pair_v61',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO_globalHash_shared_three_v61',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],
        source_pins=p['source_pins'],upstream_terminal_binding=up['binding'],source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_released',SCI=p['scientific_digest'],new_GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
