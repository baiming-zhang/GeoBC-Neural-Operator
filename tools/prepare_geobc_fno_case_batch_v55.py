"""Freeze actual parent/control metadata; never release GPU automatically."""
import argparse
import datetime

PLAN='notes/geobc_fno_case_batch_v55_plan.json'
REVIEW='notes/geobc_fno_case_batch_v55_root_review.json'
SOURCE_REVIEW='notes/geobc_fno_case_batch_v55_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.fno_case_batch_contract_v55 import metadata_inputs,SOURCES
    from geobc.macro_training import balanced_schedule
    up=metadata_inputs();cs=up['control']['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert set(SOURCES)<=set(sr['source_pins'])
    assert all(sr['source_pins'].get(k)==h for k,h in up['previous_plan']['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:cs[k]for k in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity',
        'scaler','memory_limit_bytes','GPU_lock','optimizer','precision','decoder_chunk_size',
        'parent_training_plan_identity','parent_training_SCI','parent_checkpoint','parent_summary_identity',
        'parent_external_identity','parent_norms_identity','parent_case_exposure_counts','parent_full_order_digest')}
    assert s['parent_summary_identity']==up['parent_binding']['summary']
    assert s['parent_external_identity']==up['parent_binding']['external']
    s.update(parent_step=1500,parent_exposure_cursor=1500,additional_case_exposures=1500,
        fixed_exposure_cursor=3000,additional_optimizer_updates=375,fixed_optimizer_steps=1875,
        endpoint_passes=[10],cases_per_optimizer_update=4,case_gradient_weights=[.25]*4,
        cursor_semantics='step=actual Adam count; cursor=completed ordered case exposures',
        full_order_digest=d.digest(balanced_schedule(s['TRAIN_keys'],10,2026)),
        fresh_initialization=False,loss='mean4_full_original_node_half_U_squared_plus_half_S_squared',
        cache_setup_budget_seconds=600,training_budget_seconds=600,TRAIN_readout_budget_seconds=300,
        whole_budget_seconds=1800,automatic_continuation=False,DEV_GT_reads_allowed=0,FINAL_reuse_for_selection=False,
        expected_loaded_parent_model_sha256=up['control_summary']['loaded_parent_model_sha256'],
        control_TRAIN300=up['control_summary']['endpoints']['3000']['full_TRAIN'],
        TRAIN_gate=dict(primary_absolute_improvement_at_least=.03,
            maximum_either_field_absolute_deterioration=.02,hotspot_deterioration_max=.02,fixed_denominator=300),
        control_plan_identity=d.identity('notes/geobc_macro100_FNO9_physical10_v15_plan.json',hashed=True),
        control_SCI=up['control']['scientific_digest'],control_knownDEV_plan_identity=d.identity('notes/geobc_physical10_readout_v16_plan.json',hashed=True),
        control_knownDEV_SCI=up['DEV_plan']['scientific_digest'],
        previous_ablation_SCI=up['previous_plan']['scientific_digest'],
        question='Does four-case mean gradient improve shared physical U/S fitting at exactly the same ordered case exposure?',
        physical_target_policy='Original signed U(mm)/S(MPa), complete native nodes, equal full-case weights. TRAIN targets loaded once and reused for endpoint.',
        evaluation_release='Separate true terminal decision; this plan cannot read DEV truth or launch another stage',
        simultaneous_case_graphs=1,parameter_count=4526858,
        limits=['Grouping and Adam update count change together; not sole gradient-conflict causality',
            'KnownDEV iterated/single seed, no architecture upper bound or unbiased final claim',
            'Case exposure10 is not optimizer step3000 in this branch'])
    p=dict(kind='GeoBC_FNO9_four_case_mean_gradient_v55',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO100_case_batch4_v55',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={k:up[k+'_binding']for k in ('parent','control','DEV','previous')},
        source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),status='Prepared; root must release actual bounded experiment')
    d.write(PLAN,p);d.write(REVIEW,r)
    print(dict(status='prepared_not_runtime_released',SCI=p['scientific_digest'],
        new_GT_reads=0,PT_body_reads=0,training_updates=0,GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
