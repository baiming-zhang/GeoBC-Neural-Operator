"""Prepare an independent precision experiment after exact FP32 budget terminal."""
import argparse

PLAN='notes/geobc_transolver_bf16_v65_plan.json'
REVIEW='notes/geobc_transolver_bf16_v65_root_review.json'
SOURCE_REVIEW='notes/geobc_transolver_bf16_v65_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.transolver_bf16_contract_v65 import inputs,SOURCES
    up=inputs();old=up['fp32']['scientific'];sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert set(SOURCES)<=set(sr['source_pins']) and all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists()
    s=dict(old)
    s.update(families=['TS_ORIGINAL_BF16_GLOBAL_HASH'],runtime_factory='geobc.transolver_bf16_hash_v65.build_solver',
        precision='OnlyTS operator autocastBF16; FP32 model/Adam/context/returnedlatent, TCNNnativeFP16',
        expected_FP32_TS_initialization=up['fp32_summary']['model_initialization']['TS_ORIGINAL_GLOBAL_HASH'],
        FP32_TS_partial_endpoints=up['fp32_summary']['models']['TS_ORIGINAL_GLOBAL_HASH']['endpoints'],
        old_budget_reset=False,old_training_replayed=False,new_precision_branch_fresh=True,
        scientific_scope='Independent GPUarithmetic branch after actualFP32 compute budget failure; identical freshparameters and same3000 exposure, preserve old failurecost/partial metrics',
        limits=old['limits']+['BF16 arithmetic changes numerical trajectory, compare FP32 only at its actual0/100 endpoints; no invented1000 FP32 accuracy'])
    p=dict(kind='GeoBC_originalTS_BF16_same_decoder_shared3_v65',scientific=s,scientific_digest=d.digest(s),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/Transolver_BF16_continuous_decoder_shared3_v65',prepared_at_utc=d.utc())
    r=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,scientific_digest=p['scientific_digest'],source_pins=p['source_pins'],
        upstream_terminal_bindings={k:up[k+'_binding']for k in ('shared','fp32')},source_review_identity=d.identity(SOURCE_REVIEW,hashed=True))
    d.write(PLAN,p);d.write(REVIEW,r);print(dict(status='prepared_not_released',SCI=p['scientific_digest'],new_GT_PT_GPU=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
