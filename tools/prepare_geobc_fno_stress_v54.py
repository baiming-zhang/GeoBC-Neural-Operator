"""One metadata-only current100 matched-parent plan, never auto-released."""
import argparse
import datetime

PLAN='notes/geobc_fno_stress_v54_plan.json'
REVIEW='notes/geobc_fno_stress_v54_root_review.json'
SOURCE_REVIEW='notes/geobc_fno_stress_v54_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.fno_stress_contract_v54 import metadata_inputs,SOURCES
    from geobc.macro_training import balanced_schedule
    up=metadata_inputs();cp=up['control'];cs=cp['scientific'];pp=up['parent']
    sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert set(SOURCES)<=set(sr['source_pins'])
    assert all(sr['source_pins'].get(k)==h for k,h in cp['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s={k:cs[k] for k in ('family','seed','train_CAD_count','TRAIN_keys','dataset_identity',
        'scaler','memory_limit_bytes','GPU_lock','optimizer','precision','decoder_chunk_size',
        'parent_training_plan_identity','parent_training_SCI','parent_checkpoint','parent_summary_identity',
        'parent_external_identity','parent_norms_identity','parent_case_exposure_counts','parent_full_order_digest')}
    assert s['parent_summary_identity']==up['parent_binding']['summary']
    assert s['parent_external_identity']==up['parent_binding']['external']
    s.update(parent_step=1500,additional_updates=1500,fixed_updates=3000,endpoint_passes=[10],
        full_order_digest=d.digest(balanced_schedule(s['TRAIN_keys'],10,2026)),
        fresh_initialization=False,loss='full_original_node_half_physical_S_squared_no_U_loss',
        stress_loss_coefficient=.5,cache_setup_budget_seconds=600,training_budget_seconds=600,
        TRAIN_readout_budget_seconds=300,whole_budget_seconds=1800,automatic_continuation=False,
        DEV_GT_reads_allowed=0,FINAL_reuse_for_selection=False,
        expected_loaded_parent_model_sha256=up['control_summary']['loaded_parent_model_sha256'],
        control_TRAIN300=up['control_summary']['endpoints']['3000']['full_TRAIN'],
        TRAIN_gate=dict(absolute_S_improvement_min=.03,hotspot_deterioration_max=.02,fixed_denominator=300),
        control_plan_identity=d.identity('notes/geobc_macro100_FNO9_physical10_v15_plan.json',hashed=True),
        control_SCI=cp['scientific_digest'],control_knownDEV_plan_identity=d.identity('notes/geobc_physical10_readout_v16_plan.json',hashed=True),
        control_knownDEV_SCI=up['DEV_plan']['scientific_digest'],
        question='Does removing U supervision improve shared signed stress fitting from exactly the same current100 physical5 state?',
        physical_target_policy='Original signed U(mm)/S(MPa), whole native nodes; only S contributes to updates. TRAIN targets loaded once and reused for endpoint.',
        reported_TRAIN_U_scope='Unsupervised raw specialist output; a future knownDEV composite would explicitly reuse frozen joint10 U predictions',
        evaluation_release='Separate real terminal decision; this plan cannot read DEV truth or launch another training stage',
        limits=['KnownDEV iterated, single seed; no architecture upper bound or unbiased final claim',
            'Different from earlier small3TRAIN2DEV stress-only protocols; no old experiment replay'])
    p=dict(kind='GeoBC_FNO9_matched_stress_specialization_v54',scientific=s,scientific_digest=d.digest(s),
        source_pins=sr['source_pins'],root_review=REVIEW,output='runs/geobc_exploration/FNO100_stress_specialist_v54',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={k:up[k+'_binding'] for k in ('parent','control','DEV')},
        source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),status='Prepared; root must release actual bounded experiment')
    d.write(PLAN,p);d.write(REVIEW,r)
    print({'status':'prepared_not_runtime_released','SCI':p['scientific_digest'],
        'new_GT_reads':0,'PT_body_reads':0,'training_updates':0,'GPU':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
