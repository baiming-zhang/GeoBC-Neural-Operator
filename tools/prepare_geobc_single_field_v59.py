"""Metadata-only fixed selected-field continuation; no arrays/PT bodies."""
import argparse
import datetime

PLAN='notes/geobc_single_field_v59_plan.json'
REVIEW='notes/geobc_single_field_v59_root_review.json'
SOURCE_REVIEW='notes/geobc_single_field_v59_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.single_field_learning_contract_v59 import inputs,SOURCES
    up=inputs();ps=up['plan']['scientific'];keys=up['keys'];proposal=d.read('notes/geobc_single_field_learning_v59_proposal.json');sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert proposal['selected_TRAIN_keys']==keys and proposal['upstream_terminal_binding']==up['binding']
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:ps[k]for k in ('family','seed','train_CAD_count','dataset_identity','scaler','optimizer','precision','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size','parent_TRAIN_keys')}
    s.update(selected_TRAIN_keys=keys,parent_checkpoints={k:up['summary']['arms']['independent9'][k]['checkpoint_identity']for k in keys},
        parent_training_SCI=up['plan']['scientific_digest'],parent_optimizer_step=6100,parent_case_exposure=120,
        additional_updates_per_field=900,total_additional_updates=2700,
        fixed_total_additional_exposure_endpoints_since_shared20=[300,1000],final_optimizer_step_per_model=7000,final_case_exposure_per_field=1020,
        whole_budget_seconds=900,training_budget_seconds=600,cache_setup_budget_seconds=180,
        DEV_FINAL_reads_allowed=0,automatic_continuation=False,FINAL_reuse_for_selection=False,
        parent_scalar_rows={k:up['summary']['arms']['independent9'][k]['endpoints']['100']['per_case'][k]for k in keys},
        selection_rule=proposal['selection_rule'],proposal_identity=d.identity('notes/geobc_single_field_learning_v59_proposal.json',hashed=True),
        limits=['Selected hard TRAIN fields; no unseen CAD accuracy or deployable shared solver claim',
            'Own100-to1000 learning trajectory, not matched exposure architecture ranking or convergence proof',
            'Each independent model has its own Adam/RNG; all previous cost and original shared history remain recorded'])
    p=dict(kind='GeoBC_FNO_three_single_fields_physical100_to1000_v59',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO_three_single_fields_learning_v59',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],upstream_terminal_binding=up['binding'],
        source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_released',SCI=p['scientific_digest'],selected=keys,new_GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
