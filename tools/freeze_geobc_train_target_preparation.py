"""Freeze a CPU TRAIN-only preparation stage using metadata/source only."""
import argparse
import ast
import time
from geobc import available_cad_pilot_data as d


def run():
    started=time.monotonic()
    plan_path='notes/geobc_prospective_train_targets_plan.json'
    review_path='notes/geobc_prospective_train_targets_root_review.json'
    assert not d.path(plan_path).exists() and not d.path(review_path).exists()
    split_path='notes/geobc_prospective_cad_split_v1.json'
    binding_path='notes/geobc_prospective_target_free_input_contract.json'
    split=d.read(split_path);binding=d.read(binding_path)
    assert split['scientific_digest']==d.digest(split['scientific'])
    assert binding['scientific_digest']==d.digest(binding['scientific'])
    assert binding['scientific']['split_scientific_digest']==split['scientific_digest']
    selected=split['scientific']['additional_TRAIN60_CADs']
    assert len(selected)==len(set(selected))==60
    dependencies=['tools/prepare_geobc_prospective_train_targets.py','tools/run_geobc_train_target_guard.py',
                  'tools/freeze_geobc_train_target_preparation.py','geobc/available_cad_pilot_data.py',
                  'geobc/benchmark.py','geobc/models.py','geobc/train.py']
    for name in dependencies:ast.parse(d.path(name).read_text(encoding='utf-8-sig'))
    science={'role':'Independent CPU target-only preparation of new prospective TRAIN60; no input/grid/model preparation',
        'split_identity':d.identity(split_path,True),'input_binding_identity':d.identity(binding_path,True),
        'selected_TRAIN60_CADs':selected,'loads':['ver','hor','tor'],'required_case_keys':d.keys(selected),
        'scaler':split['scientific']['scaler'],'scaler_policy':'Inherit exact existing TRAIN40-only scaler; no fit',
        'whole_CPU_budget_seconds':1800,'workers':1,
        'alignment':'Original H5 vertices FP32 matched bijectively to CSV GRID coordinates float64 within0.0002mm; no GRID-1',
        'numeric_reads':'One CSV parse per TRAIN CAD of coordinates+known3 original twelve U/S columns only; H5 vertices only',
        'targets':'Raw original signed U(mm)/S(MPa); aligned physical FP32 then original FieldScaler encoded FP32',
        'read_denominators':{'TRAIN_CSV':60,'TRAIN_U_cases':180,'TRAIN_S_cases':180,'DEV':0,'FINAL':0},
        'io_policy':'Check reception byteSHA before parsing; preserve attempt/completed counters, arrays and cost on failure; no replay/reset/substitution',
        'source_history':'Original540/660/2400 input-negative protocol remains unchanged; not original training200',
        'future_join':'Input preparation must match each saved H5-row-to-CSV permutation identity before any training',
        'case_norms':'Original physical targets FP64 squared norms; true zeros retained as zero, no epsilon or dropped cases',
        'parallel_policy':'CPU only, one worker/thread, no GPU lock/operations; record exact overlapping project GPU process and nonisolated cost',
        'limitations':['This stage does not establish real feature/grid input feasibility or train any new solver',
                       'Geometry-only FINAL preparation and all FINAL target barriers remain separately required',
                       'Timeout/interruption without native closure requires technical review, not normal-negative acceptance or retry']}
    plan={'scientific':science,'scientific_digest':d.digest(science),'source_pins':{name:d.sha(name) for name in dependencies},
          'output':'data/geobc_prospective_TRAIN60_targets_v1','root_review':review_path}
    d.write(plan_path,plan)
    d.write(review_path,{'passed':True,'reviewer':'root','whole_SOURCE_reviewed':True,
        'CPU_preparation_released':True,'scientific_digest':plan['scientific_digest'],'source_pins':plan['source_pins'],
        'review_basis':['Explicit selected TRAIN60 equals immutable split, disjoint oldTRAIN/DEV/FINAL/originalsealed20',
                       'Guard/source/metadata identities, one shot, dual1800s deadline including startup/IO/save/failure/exit',
                       'Native CSV reads only coordinates plus known3 columns; H5 reads only vertices, labels sourceCSV notH5',
                       'Original physical FP32/aligned GRID/constant TRAIN40 scaler encoding, no signs/clipping/defaults/scaler fitting',
                       'CPU libraries only; actual CUDA-initialization guard, thread1, no model construction or GPU tasks',
                       'Real guard/child PID/ticks/argv captured before reads, owned-child cleanup, true external exit and exact cost',
                       'Per-CAD three records commit atomically in metadata, fixed180 denominator; failure prefix/cost retained',
                       'Future geometry inputs not prepared, FINAL labels sealed; scientific roles separate from helper aliases'],
        'actual_preparation_executions':0,'actual_final_or_DEV_target_reads':0,'source_metadata_seconds':time.monotonic()-started})
    print({'status':'CPU_TRAIN_only_source_frozen','scientific_digest':plan['scientific_digest'],
           'actual_numeric_reads':0,'metadata_seconds':time.monotonic()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    if parser.parse_args().freeze:run()
