"""Metadata-only fresh diagnostic plan, existing model mathematics unchanged."""
import argparse
import datetime

PLAN='notes/geobc_single_decoder_v60_plan.json'
REVIEW='notes/geobc_single_decoder_v60_root_review.json'
SOURCE_REVIEW='notes/geobc_single_decoder_v60_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.single_decoder_pair_contract_v60 import inputs,SOURCES,MODEL_SOURCES
    up=inputs();ps=up['plan']['scientific'];sr=d.read(SOURCE_REVIEW);proposal=d.read('notes/geobc_single_field_decoder_pair_v60_proposal.json')
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES+MODEL_SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert proposal['selected_TRAIN_keys']==up['keys'] and proposal['upstream_terminal_binding']==up['binding']
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:ps[k]for k in ('seed','train_CAD_count','dataset_identity','scaler','optimizer','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size','parent_TRAIN_keys')}
    s.update(selected_TRAIN_keys=up['keys'],families=['FNO9','FNO_GLOBAL_HASH'],fresh_models_per_family=3,
        updates_per_field=1000,total_updates_per_family=3000,total_updates=6000,endpoints=[0,100,1000],
        fresh_initialization=True,parent_checkpoint_reads_allowed=0,
        parameter_counts=dict(FNO9=4526858,FNO_GLOBAL_HASH=12245894),runtime_factory='geobc.operator_hash_metadata_v6.build_solver',
        factory_choice='Reuse existing metadata-only wrapper fixing source_config serialization; original numerical forward/backward and TCNN config unchanged',
        whole_budget_seconds=1200,training_budget_seconds=900,cache_setup_budget_seconds=180,
        DEV_FINAL_reads_allowed=0,automatic_continuation=False,FINAL_reuse_for_selection=False,
        representative_field='229_200/ver',visualization_sample_nodes_max=20000,
        visualization_policy='Same fixed evenly spaced native node indices and camera; symmetric MPa range shared across truth and both endpoint predictions; metrics use all nodes',
        proposal_identity=d.identity('notes/geobc_single_field_decoder_pair_v60_proposal.json',hashed=True),
        limits=proposal['comparison_scope'],question=proposal['question'],
        scientific_scope='Independent selected TRAIN fitting only; no unseen CAD accuracy, architecture ceiling or final solver claim',
        RNG_policy='Per-family same fresh complete initialization restored per field; fresh Adam0 and four RNG; endpoint readout original TRAIN-index seed restores RNG')
    p=dict(kind='GeoBC_fresh_FNO9_globalHash_three_single_field_decoder_pair_v60',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO_globalHash_fresh_three_single_v60',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],upstream_terminal_binding=up['binding'],
        source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_released',SCI=p['scientific_digest'],selected=up['keys'],new_GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
