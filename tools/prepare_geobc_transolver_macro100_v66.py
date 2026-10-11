"""Metadata-only future release, after genuine completed sharedTS physical report."""
import argparse

PLAN='notes/geobc_transolver_macro100_v66_plan.json'
REVIEW='notes/geobc_transolver_macro100_v66_root_review.json'
SOURCE_REVIEW='notes/geobc_transolver_macro100_v66_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_macro100_contract_v66 import inputs,SOURCES
    from geobc.macro_training import balanced_schedule
    up=inputs();sr=d.read(SOURCE_REVIEW);old=up['TS']['scientific']
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s=dict(old);m=d.read(old['dataset_identity']['path']);keys=up['keys']
    s.update(TRAIN_keys=keys,selected_TRAIN_keys=keys,updates_per_field=5,total_updates=1500,total_updates_per_family=1500,
        case_pass_endpoints=[0,1,5],full_order_digest=d.digest(balanced_schedule(keys,5,2026)),cache_setup_budget_seconds=600,training_budget_seconds=900,whole_budget_seconds=1800,
        qualification_TRAIN_keys=[max([k for k in keys if k.endswith('/'+load)],key=lambda k:m['records'][k]['num_nodes'])for load in ('ver','hor','tor')],
        reference_TRAIN5={name:up[name+'_summary']['endpoints']['1500']['full_TRAIN']for name in ('Hash','FNO')},
        tiny_TRAIN_eligibility=dict(U_at_most=.15,S_at_most=.20,hot_at_most=.30,scope='fitting ability and completed cost only, not newCAD selection'),
        scientific_scope='One short fresh100CADx3load fixed5 exposure originalTS+continuous decoder; reuse immutable same5 FNO/Hash physical metrics, no backbone ceiling claim',
        physical_target_policy='TRAIN300 physical only once; remove encoded files before load; fullN0/1/5resident metrics; DEV/FINAL0 until separately committed predictions',
        subsequent_DEV_policy='All valid fixed endpoint candidates are evaluated; TRAIN scalar quality does not drop a finite completed candidate',
        limits=old['limits']+['Different backbone capacity/precision and parameter initialization from prior macro baselines; identical CAD/scaler/physical objective and exposure only',
            'Short five passes can screen capability and transfer but cannot prove convergence or architecture ceiling'])
    p=dict(kind='GeoBC_originalTS_BF16_continuous_decoder_TRAIN100_physical5_v66',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/Transolver_BF16_continuous_decoder_TRAIN100_physical5_v66',prepared_at_utc=d.utc())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={name:up[name+'_binding']for name in ('TS','Hash','FNO')},source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='prepared_not_released',SCI=p['scientific_digest'],new_GT_PT_GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
