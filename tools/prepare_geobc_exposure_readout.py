"""SOURCE-only preparation and later scalar terminal release for fixed240 predictions."""
import argparse
import ast
import time
from geobc import available_cad_pilot_data as d

PLAN='notes/geobc_exposure_readout_plan.json'


def prepare():
    assert not d.path(PLAN).exists()
    parent=d.read('notes/geobc_exposure_extension_plan.json');old=d.read('notes/geobc_common_grid_readout_plan.json')
    assert all(d.sha(p)==h for p,h in parent['source_pins'].items())
    guard_path='tools/run_geobc_exposure_readout_guard.py'
    source=d.path('tools/run_geobc_common_readout_guard.py').read_text(encoding='utf-8')
    source=source.replace('1800','3600').replace('geobc_common_grid_readout_plan','geobc_exposure_readout_plan')
    source=source.replace('GEOBC_COMMON_READOUT_GUARD','GEOBC_EXPOSURE_READOUT_GUARD')
    source=source.replace('tools.evaluate_geobc_common_grid','tools.evaluate_geobc_exposure_endpoints')
    source=source.replace("    output = d.path(plan['output'])", "    from tools.evaluate_geobc_exposure_endpoints import parent_qualified\n    parent_qualified(plan,review)\n    output = d.path(plan['output'])")
    d.path(guard_path).write_text(source,encoding='utf-8')
    candidates=[dict(name=arm+'_step'+str(step),arm=arm,step=step) for arm in ['FNO9','PP256'] for step in [12000,24000]]
    science=dict(question='Do100/200 shared exposures change U/S/hot on the same known development60?',
        parent_plan='notes/geobc_exposure_extension_plan.json',parent_digest=parent['scientific_digest'],candidates=candidates,
        case_keys=old['scientific']['case_keys'],input_manifest_identity=old['scientific']['input_manifest_identity'],
        GT_manifest_identity=old['scientific']['GT_manifest_identity'],GPU_lock=parent['scientific']['GPU_lock'],
        prediction_denominator=240,case_denominator=60,whole_budget_seconds=3600,memory_cap_GiB=24,
        seed='2026+fixed_case_index, same as completed4800 readout',
        barrier='All4x60 immutable finite full-original-node encoded/physical predictions verified before known ownedGT60 once',
        data_scope='Previously known development only; rawCSV/FINAL/sealedDEV/training/scalerfit0',
        qualification='Genuine both24000 terminal/exact exits and fullTRAIN120 endpoints/resource/cost required',
        failure='Technical/nonfinite/budget failure retains full240/60 denominators, GT gate closed before complete barrier; no subset score or silent replay',
        old4800_reference='results/geobc_common_grid_comparison/metrics.json; scalar comparisons only, not additional predictions or GT',
        final_solver_selected=False)
    pins=dict(parent['source_pins'])
    for p in ['tools/evaluate_geobc_exposure_endpoints.py',guard_path,'tools/prepare_geobc_exposure_readout.py']:
        ast.parse(d.path(p).read_text(encoding='utf-8'));pins[p]=d.sha(p)
    plan=dict(scientific=science,scientific_digest=d.digest(science),source_pins=pins,
        output='runs/geobc_exploration/matched_exposure_development60_v1',root_review='notes/geobc_exposure_readout_root_review.json')
    review=dict(passed=False,whole_SOURCE_reviewed=False,actual_readout_released=False,parent_binding=None,
        scientific_digest=plan['scientific_digest'],source_pins=pins,source_preparation_epoch=time.time(),
        actual_array_GT_PT_model_GPU_reads=0)
    d.write(PLAN,plan);d.write(plan['root_review'],review)
    print({'status':'source_prepared_actual0','science':plan['scientific_digest'],'predictions':240})


def release():
    from tools.evaluate_geobc_exposure_endpoints import parent_qualified
    plan=d.read(PLAN);review=d.read(plan['root_review'])
    assert review['passed'] and review['whole_SOURCE_reviewed'] and not review['actual_readout_released']
    assert plan['scientific_digest']==review['scientific_digest']==d.digest(plan['scientific'])
    assert review['source_pins']==plan['source_pins'] and all(d.sha(p)==h for p,h in plan['source_pins'].items())
    parent=d.read(plan['scientific']['parent_plan']);folder=d.path(parent['output'])
    review['parent_binding']={k:d.identity(folder/f,hashed=True) for k,f in [('summary','summary.json'),('external','external_terminal.json')]}
    parent_qualified(plan,review)
    review.update(actual_readout_released=True,release_epoch=time.time(),release_array_GT_PT_model_GPU_reads=0)
    d.write(plan['root_review'],review)
    print({'released':True,'all240_required_before_GT60':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--release',action='store_true')
    args=parser.parse_args();assert not (args.prepare and args.release)
    if args.prepare:prepare()
    if args.release:release()
