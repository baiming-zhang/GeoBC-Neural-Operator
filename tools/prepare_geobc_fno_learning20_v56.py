"""Metadata-only fixed20 plan from real qualified10 endpoint and gains."""
import argparse
import datetime

PLAN='notes/geobc_fno_learning20_v56_plan.json'
REVIEW='notes/geobc_fno_learning20_v56_root_review.json'
SOURCE_REVIEW='notes/geobc_fno_learning20_v56_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.fno_learning20_contract_v56 import metadata_inputs,SOURCES
    from geobc.macro_training import balanced_schedule
    up=metadata_inputs();ps=up['parent']['scientific'];a=up['parent_summary'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert set(SOURCES)<=set(sr['source_pins'])
    assert all(sr['source_pins'].get(k)==h for k,h in up['last_recipe_plan']['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:ps[k]for k in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity',
        'scaler','memory_limit_bytes','GPU_lock','optimizer','precision','decoder_chunk_size','loss','parent_norms_identity')}
    s.update(parent_training_plan_identity=d.identity('notes/geobc_macro100_FNO9_physical10_v15_plan.json',hashed=True),
        parent_training_SCI=up['parent']['scientific_digest'],parent_checkpoint=a['endpoints']['3000']['checkpoint_identity'],
        parent_summary_identity=up['parent_binding']['summary'],parent_external_identity=up['parent_binding']['external'],
        parent_case_exposure_counts=a['case_exposure_counts'],parent_full_order_digest=ps['full_order_digest'],
        parent_TRAIN300=a['endpoints']['3000']['full_TRAIN'],parent_step=3000,additional_updates=3000,fixed_updates=6000,
        endpoint_passes=[20],full_order_digest=d.digest(balanced_schedule(s['TRAIN_keys'],20,2026)),
        fresh_initialization=False,cache_setup_budget_seconds=600,training_budget_seconds=600,
        TRAIN_readout_budget_seconds=300,whole_budget_seconds=1800,automatic_continuation=False,
        DEV_GT_reads_allowed=0,FINAL_reuse_for_selection=False,
        TRAIN_gate=dict(primary_absolute_improvement_at_least=.03,
            maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=300),
        previous_5to10_TRAIN_gate=a['TRAIN_gate'],previous_5to10_DEV_gate=up['DEV_summary']['DEV_gate'],
        last_negative_recipe_SCI=up['last_recipe_plan']['scientific_digest'],
        release_scope='One selected fixed20 trend check; no automatic40 or24000 or other solver extension',
        parameter_count=4526858,physical_target_policy='Original signed U(mm)/S(MPa), full native nodes, equal-case joint physical loss unchanged; TRAIN targets loaded once and endpoint reuses resident labels',
        question='Do the previously observed5-to10 TRAIN and DEV gains persist for one selected, cost-bounded10-to20 FNO continuation?',
        reason='Both real5-to10 gates passed; stress-only and casebatch4 did not. Prefer learning-trend evidence over further small recipe variants.',
        evaluation_release='Separate actual terminal review before any new knownDEV60 prediction-beforeGT once; FINAL0',
        limits=['Additional exposure and cost, not equal-budget architecture ranking',
            'KnownDEV iterated/single seed; no architecture upper bound, convergence or unbiased final claim'])
    p=dict(kind='GeoBC_FNO9_evidence_gated_physical10_to20_v56',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO100_physical20_v56',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={k:up[k+'_binding']for k in ('parent','original5','DEV','last_recipe')},
        source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),status='Prepared; selected fixed20 runtime still requires root release')
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_runtime_released',SCI=p['scientific_digest'],new_GT_reads=0,PT_body_reads=0,training_updates=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
