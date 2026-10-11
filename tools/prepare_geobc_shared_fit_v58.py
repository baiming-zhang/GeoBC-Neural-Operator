"""Prepare predeclared TRAIN-only diagnostic from immutable scalar metadata."""
import argparse
import datetime

PLAN='notes/geobc_shared_fit_v58_plan.json'
REVIEW='notes/geobc_shared_fit_v58_root_review.json'
SOURCE_REVIEW='notes/geobc_shared_fit_v58_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.shared_fit_contract_v58 import inputs,SOURCES
    up,vp,vb,cads,keys,scores=inputs();ps=up['plan']['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items()) and set(SOURCES)<=set(sr['source_pins'])
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:ps[k]for k in ('family','seed','train_CAD_count','dataset_identity','scaler','optimizer','precision','loss','GPU_lock','memory_limit_bytes','decoder_chunk_size')}
    s.update(selected_CADs=cads,selected_TRAIN_keys=keys,
        selection_rule='Three highest TRAIN20 mean signed stress relative L2 across ver/hor/tor; ties CAD lexicographic',
        selection_scores={c:scores[c]for c in cads},parent_checkpoint=up['checkpoint'],
        parent_training_SCI=up['plan']['scientific_digest'],parent_TRAIN_keys=ps['TRAIN_keys'],
        parent_TRAIN_rows={k:up['summary']['endpoints']['6000']['full_TRAIN']['per_case'][k]for k in keys},
        arms=['shared9','independent9'],additional_exposures_per_case=100,endpoints_additional_exposures=[0,20,100],
        arm_optimizer_updates=900,total_optimizer_updates=1800,cache_setup_budget_seconds=300,
        training_budget_seconds=600,whole_budget_seconds=1200,DEV_FINAL_reads_allowed=0,
        automatic_continuation=False,FINAL_reuse_for_selection=False,
        RNG_policy='Each model restores same parent four RNG; endpoint readout uses original TRAIN index seed and restores RNG',
        optimizer_policy='Each model restores independent copy of parent Adam step6000; shared9 adds900; each independent model adds100',
        limits=['Hard TRAIN-case selection; fitting diagnosis, no heldout or deployable solver claim',
            'Same per-case exposure and total updates, but nine independent models have nine times aggregate model capacity and optimizer trajectories',
            'Subset shared9 adapts away from fullTRAIN300; do not deploy or report as allTRAIN improvement',
            'No architecture upper bound or attribution to gradient interference alone'])
    p=dict(kind='GeoBC_FNO20_selected_TRAIN_shared_vs_independent_v58',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO20_selected_shared_fit_v58',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings=dict(TRAIN20=up['binding'],DEV20=vb),source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_released',SCI=p['scientific_digest'],selected=cads,new_GT_reads=0,PT_body_reads=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
