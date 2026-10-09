"""One-time metadata migration for the pre-array null archive SHA failure."""
import copy
import hashlib
import json
import pathlib
import shutil
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = 'ccb9279fa2e21fe24b903da9020f2d8ebfbc514802676741e5f54b916368713f'
NEW = '897114caccffad680559f4cd868183eae5d2594d5b1d2af679defd48f367461f'
MODULE = 'geobc/elastic_crosscad.py'

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))

def write(path, value):
    (ROOT/path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def main():
    stage = 'runs/geobc_exploration/elastic_crosscad5_ver'
    archive = ROOT/'runs/geobc_exploration/elastic_crosscad_attempts/null_inherited_archive_sha'
    assert not archive.exists(), 'Recovery already applied; do not repeat'
    assert hashlib.sha256((ROOT/MODULE).read_bytes()).hexdigest() == NEW
    config, summary, status = (read(stage+'/'+name) for name in ('config.json','summary.json','status.json'))
    manifest = read('data/geobc_elastic_crosscad5/101_428/input/manifest.json')
    assert config['contract']['source_sha256'][MODULE] == OLD
    assert summary['status'] == 'failed' and summary['reason'] == 'ValueError: Inherited cells archive byte identity differs'
    assert not summary['cases'] and all(v == 0 for v in summary['read_receipt'].values())
    assert not manifest['read_receipt']['accessed_members'] and not manifest['read_receipt']['raw_deck_open_attempted']
    prior = max(status['cumulative_stage_wall_seconds'], summary['costs']['cumulative_stage_wall_seconds'])
    assert 0 < prior < 1, 'Unexpected prior cost; no automatic migration'
    saved = []
    for path in [stage+'/config.json', stage+'/summary.json', stage+'/status.json',
                 'data/geobc_elastic_crosscad5/101_428/input/config.json',
                 'data/geobc_elastic_crosscad5/101_428/input/manifest.json',
                 'notes/geobc_elastic_crosscad_plan.json','notes/elastic_crosscad_root_review.json',
                 'runs/geobc_exploration/coordinator_status.json','runs/geobc_exploration/elastic_crosscad5_ver.log']:
        src = ROOT/path; dst = archive/path
        assert src.resolve().is_relative_to(ROOT.resolve()) and dst.resolve().is_relative_to(archive.resolve())
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst); saved.append(path)
    old_config_digest = digest(config)
    config['contract']['source_sha256'][MODULE] = NEW
    assert summary['config']['contract']['source_sha256'][MODULE] == OLD
    summary['config'] = copy.deepcopy(config)
    receipt_path = 'notes/elastic_crosscad_runtime_recovery_root.json'
    summary['runtime_recovery_metadata_migration'] = receipt_path
    status['config_sha256'] = digest(config)
    status['cumulative_stage_wall_seconds'] = prior
    plan = read('notes/geobc_elastic_crosscad_plan.json')
    assert plan['source_sha256'][MODULE] == OLD
    plan['source_sha256'][MODULE] = NEW
    plan['source_ready'] = 'notes/elastic_crosscad_runtime_recovery_source_ready.json'
    plan['runtime_recovery'] = receipt_path
    write('notes/geobc_elastic_crosscad_plan.json', plan)
    write(stage+'/config.json', config); write(stage+'/summary.json', summary); write(stage+'/status.json', status)
    review = read('notes/elastic_crosscad_root_review.json')
    review['source_sha256'][MODULE] = NEW
    review['source_ready'] = plan['source_ready']; review['runtime_recovery'] = receipt_path
    review['plan_sha256'] = hashlib.sha256((ROOT/'notes/geobc_elastic_crosscad_plan.json').read_bytes()).hexdigest()
    write('notes/elastic_crosscad_root_review.json', review)
    receipt = dict(status='pre_array_failed_attempt_preserved_and_source_pin_migration_applied',
                   time_utc=datetime.now(timezone.utc).isoformat(), source_ready=plan['source_ready'],
                   archived_directory=archive.relative_to(ROOT).as_posix(), archived_files=saved,
                   reason='Inherited optional archive SHA is null, not a mismatched pinned digest',
                   previous_source_sha256=OLD, source_sha256=NEW,
                   previous_config_sha256=old_config_digest, config_sha256=digest(config),
                   cumulative_stage_wall_seconds_retained=prior, budget_reset=False,
                   scientific_contract_records_recipe_and_science_digest_unchanged=True,
                   scientific_cache_or_checkpoint_created=False, arrays_read=0, GT_reads=0, GPU_runs=0,
                   coordinator_274_exact_argv_identity_confirmed_gone=True)
    write(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))

def repair_digest():
    stage = 'runs/geobc_exploration/elastic_crosscad5_ver'
    config, summary, status = (read(stage+'/'+name) for name in ('config.json','summary.json','status.json'))
    coordinator = read('runs/geobc_exploration/coordinator_status.json')
    assert coordinator['phase'] == 'failed' and coordinator['pid'] == 361
    assert config['contract']['source_sha256'][MODULE] == NEW
    archive = ROOT/'runs/geobc_exploration/elastic_crosscad_attempts/source_pin_digest_migration_failure'
    assert not archive.exists()
    for path in [stage+'/config.json',stage+'/summary.json',stage+'/status.json',
                 'runs/geobc_exploration/coordinator_status.json','runs/geobc_exploration/elastic_crosscad5_ver.log',
                 'notes/elastic_crosscad_runtime_recovery_root.json']:
        dst=archive/path; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/path,dst)
    contract=config['contract']
    science={k:v for k,v in contract.items() if k not in ('schema','science_sha256','array_access_in_request_contract','raw_deck_read_in_request_contract')}
    old_digest=contract['science_sha256']; contract['science_sha256']=digest(science)
    stopped=datetime.fromisoformat(coordinator['failures'][-1]['failed_at_utc'])
    started=datetime.fromisoformat(coordinator['current_started_at_utc'])
    span=(stopped-started).total_seconds()
    assert 0 <= span < 15
    prior=max(status['cumulative_stage_wall_seconds'],summary['costs']['cumulative_stage_wall_seconds'])
    # Conservative budget charge includes launcher/import and supervisor polling span.
    carried=prior+span
    status.update(config_sha256=digest(config),cumulative_stage_wall_seconds=carried)
    summary['config']=copy.deepcopy(config); summary['costs']['cumulative_stage_wall_seconds']=carried
    write(stage+'/config.json',config); write(stage+'/summary.json',summary); write(stage+'/status.json',status)
    receipt=read('notes/elastic_crosscad_runtime_recovery_root.json')
    receipt['scientific_contract_records_recipe_and_science_digest_unchanged']=False
    receipt['physical_recipe_case_records_inputs_budgets_unchanged']=True
    receipt['correction']='Source pins are part of science digest; recomputed enclosing science/config digests after preserved migration failure.'
    receipt['previous_science_sha256']=old_digest; receipt['science_sha256']=contract['science_sha256']
    receipt['config_sha256']=digest(config)
    receipt['second_failed_attempt_archived']=archive.relative_to(ROOT).as_posix()
    receipt['second_preclock_attempt_conservative_supervisor_span_seconds']=span
    receipt['cumulative_stage_wall_seconds_retained']=carried
    receipt['cost_note']='Second attempt stopped before StageClock or scientific input. Its supervisor span is conservatively charged, including import/poll overhead; not claimed as precisely measured solver work.'
    receipt['coordinator_361_exact_argv_identity_confirmed_gone']=True
    write('notes/elastic_crosscad_runtime_recovery_root.json',receipt)
    print(json.dumps({'status':'source_pin_science_digest_corrected','carried_stage_budget_seconds':carried,'supervisor_span_seconds':span,'config_sha256':digest(config)}))

if __name__ == '__main__':
    import sys
    if '--repair-digest' in sys.argv:
        repair_digest()
    else:
        main()
