"""Reserve a future CAD split using only existing manifests and file stats."""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from geobc import available_cad_pilot_data as d

OUTPUT = 'notes/geobc_prospective_cad_split_v1.json'


def ids(value):
    result = set()
    if isinstance(value, str):
        result.update(re.findall(r'(?<![0-9A-Za-z])([0-9]+_[0-9]+)(?![0-9A-Za-z])', value))
    elif isinstance(value, list):
        for item in value: result.update(ids(item))
    elif isinstance(value, dict):
        for key, item in value.items(): result.update(ids(key)); result.update(ids(item))
    return result


def run():
    started = time.monotonic()
    assert not d.path(OUTPUT).exists(), 'Existing reservation is immutable; no re-selection'
    pilot = d.read('notes/geobc_available_cad_pilot_plan.json')
    selection = pilot['scientific']['CAD_selection']
    parent = d.read('notes/geobc_common_grid_train_plan.json')
    files = sorted(d.path('data').glob('prepared*/manifest.json')) + sorted(d.path('data').glob('geobc*/manifest.json'))
    history = set(); evidence = []
    for file in files:
        value = d.read(file); used = set()
        for field in ('selection','cases','train_keys','validation_keys','case_keys','records','CADs','cad_ids'):
            if field in value: used.update(ids(value[field]))
        history.update(used)
        evidence.append({'identity': d.identity(file, hashed=True), 'cached_case_CADs': sorted(used)})
    # These five had U then S service comparisons; explicitly exclude them even
    # though their readout caches are under runs rather than prepared data.
    service5 = ['489_12','616_205','416_282','247_50','142_454']
    history.update(service5)
    old = selection['excluded_old_TRAIN40']; dev = selection['selected_CADs']
    original_sealed = selection['excluded_original_DEV20']
    assert len(old) == 40 and len(dev) == 20 and len(original_sealed) == 20
    candidates = set(selection['eligible_CADs']) - history - set(old) - set(dev) - set(original_sealed)
    assert len(candidates) >= 80, 'Cannot reserve complete fixed100/20/20 from current pool'
    prefix = 'GeoBC prospective CAD split v1|'
    ranked = sorted(candidates, key=lambda cad: (hashlib.sha256((prefix+cad).encode()).hexdigest(), cad))
    final = sorted(ranked[:20]); new_train = sorted(ranked[20:80]); reserve = sorted(ranked[80:])
    train = sorted(old + new_train)
    assert len(set(train)) == 100 and len(new_train) == 60 and len(final) == 20
    assert not (set(train)&set(dev) or set(train)&set(final) or set(dev)&set(final))
    assert not (set(final)&history)
    raw = d.read('data/raw_geobc_coverage/summary.json')
    assert raw['status'] == 'budget_exhausted' and raw['read_receipt']['label_array_reads'] == 0
    assert not raw['read_receipt']['Field_CSV_values_parsed']
    lookup = {(row['cad_id'],row['component']):row for row in raw['files']}
    source_receipts = {}
    for cad in new_train + final:
        source_receipts[cad] = {}
        for component in ('FieldMesh','Field','SolverDeck'):
            receipt = lookup[(cad, component)]
            assert receipt['status'] == 'completed' and receipt['scientific_array_reads'] == 0
            assert not receipt['Field_CSV_values_parsed']
            file = 'data/raw_geobc_coverage/'+receipt['relative_path']
            item = d.identity(file, hashed=False)
            assert {k:item[k] for k in ('bytes','mtime_ns')} == receipt['file_stat']
            assert item['bytes'] == receipt['actual_bytes']
            sha = receipt['observed_sha256']; assert re.fullmatch('[0-9a-f]{64}', sha)
            item.update(received_sha256=sha, checksum_origin=receipt['checksum_origin'])
            source_receipts[cad][component] = item
    science = {
        'role':'Prospective availability-conditional final reservation; distinct new protocol, not completion/replacement of original40vs200 input-negative campaign',
        'selection_rule': 'SHA256 UTF8("'+prefix+'"+CAD) ascending, CAD tie; first20 FINAL,next60 additionalTRAIN,remaining reserve. No label or model-performance selection.',
        'source_pool_identity':d.identity('notes/geobc_available_cad_pilot_plan.json', hashed=True),
        'parent_training_scientific_digest':parent['scientific_digest'],
        'TRAIN_CADs':train,'old_TRAIN40_CADs':old,'additional_TRAIN60_CADs':new_train,
        'DEV_CADs':dev,'FINAL_CADs':final,'reserve_CADs_not_substitutes':reserve,
        'original_sealed_DEV20_preserved':original_sealed,'excluded_tracked_history_CADs':sorted(history),
        'loads':['ver','hor','tor'],'case_counts':{'TRAIN':300,'DEV':60,'FINAL':60},
        'scaler_policy':'For a future40vs100 coverage comparison retain the original TRAIN40-only scaler; no DEV/FINAL fit. Other experiments require a separately frozen TRAIN-only policy.',
        'scaler':parent['scientific']['scaler'],
        'FINAL_policy':'Do not parse numerical FINAL CSV/H5 fields, generate target caches, train or tune on FINAL. Geometry-only preparation needs a separately frozen target-free producer. All selected models must submit complete finite60 predictions or honest failures before one shared FINAL readout.',
        'no_case_substitution':True,'dia_fields_remain_sealed':True,
        'former_protocol':'Original source540/input660/prediction2400 campaign SCI and fixed IDs remain unchanged/negative; do not restart original training200 because its proposed TRAIN includes the newly reservedFINAL.',
        'interpretation_limits':['OfficialTRAIN-pool availability conditional, not officialTEST/unbiased population sample','Identity-disjoint CADs; shape-family separation not demonstrated','Reservation is supported by tracked prepared-manifest/readout history, not a claim about every external human/file read','Existing binary reception SHA is not independent upstream checksum; current stats checked without rereading raw bytes','No numerical labels/models/arrays used to choose IDs; metadata may reference previously known targets that are conservatively excluded']}
    value = {'status':'reserved_metadata_only','observed_UTC':datetime.now(timezone.utc).isoformat(),
        'scientific':science,'scientific_digest':d.digest(science),
        'history_manifest_evidence':evidence,'source_receipts':source_receipts,
        'acquisition_summary_identity':d.identity('data/raw_geobc_coverage/summary.json', hashed=True),
        'source_sha256':d.sha('tools/freeze_geobc_prospective_split.py'),
        'actual_new_TRAIN_target_parses':0,'actual_FINAL_target_parses':0,
        'actual_new_inputs_prepared':0,'actual_training_or_GPU':0,
        'actual_execution_released':False,'source_metadata_seconds':time.monotonic()-started}
    d.write(OUTPUT,value)
    print(json.dumps({'status':value['status'],'TRAIN_CADs':100,'DEV_CADs':20,'FINAL_CADs':20,
        'remaining_candidates_before_split':len(candidates),'reserve':len(reserve),
        'scientific_digest':value['scientific_digest'],'metadata_seconds':value['source_metadata_seconds'],
        'numerical_GT_or_GPU':0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true')
    if parser.parse_args().freeze: run()
