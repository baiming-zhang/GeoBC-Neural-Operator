"""Genuine fixed5 endpoint or explicitly ineligible budget-checkpoint diagnosis.

No training continuation, label arrays or checkpoint bodies here. A separately
frozen plan must name the actual terminal endpoint and its complete identities.
"""
from . import available_cad_pilot_data as d
from .global_hash_physical_readout_contract_v18 import check_identity,complete_physical

TRAIN='notes/geobc_gino_physical_v52_device_recovery_plan.json'
METHOD='GINO100_PHYSICAL_ENDPOINT'
REQUIRED_SOURCES=('geobc/gino_knownDEV_radius_v53.py','geobc/gino_readout_contract_v53.py',
    'tools/evaluate_geobc_gino_v53.py','tools/run_geobc_gino_readout_guard_v53.py',
    'tools/prepare_geobc_gino_readout_v53.py')


def classify_endpoint(a,e,s):
    """Classify real closed cost/step metadata without reading numerical state."""
    assert a['closed'] and not a['active'] and e['closed'] and not e['active']
    assert e['actual_child_exited'] and e['guard_error_type'] is None
    assert a['family']=='GINO_PHYSICAL' and a['cache_load_complete']
    assert a['actual_largest_fullN_gradient_check']['passed']
    assert a['actual_fresh_initial_model_sha256']==s['expected_fresh_initial_model_sha256']
    assert a['raw_GT_reads']==a['DEV_FINAL_reads']==a['scaler_fits']==0
    assert a['cache_receipt']['physical_TRAIN_target_reads']==300
    assert a['cache_receipt']['encoded_TRAIN_target_reads']==0
    assert a['prior_failure_cost']==s['prior_failure_cost']
    assert sum(a['case_exposure_counts'].values())==a['optimizer_step_completions']==a['step']
    assert set(a['case_exposure_counts'])==set(s['TRAIN_keys'])
    assert 0<=a['optimizer_step_completions']<=a['optimizer_step_attempts']<=1500
    assert max(a[k] for k in ('peak_allocated_bytes','peak_reserved_bytes',
        'peak_total_device_used_bytes_sampled'))<=30500000000
    if a['status']=='completed':
        assert e['exit_code']==0 and e['within_external_budget'] and not e['external_timeout']
        assert a['training_complete'] and a['step']==a['optimizer_step_attempts']==a['optimizer_step_completions']==1500
        assert not a['optimizer_step_in_progress'] and not a['possible_partial_optimizer_update']
        assert a['case_exposure_counts']=={k:5 for k in s['TRAIN_keys']}
        assert a['physical_TRAIN_total_array_reads']==900 and a['cached_TRAIN_physical_readout_reuses']==300
        assert max(a['cache_setup_epoch_seconds'],a['cache_setup_monotonic_seconds'])<=s['cache_setup_budget_seconds']
        assert max(a['qualification_epoch_seconds'],a['qualification_monotonic_seconds'])<=180
        assert max(a['training_epoch_seconds'],a['training_monotonic_seconds'])<=3600
        assert max(e['original_whole_epoch_span_seconds'],e['original_whole_monotonic_span_seconds'])<=5400
        point=a['endpoints']['1500'];complete_physical(point['full_TRAIN'],s['TRAIN_keys'])
        assert max(point['epoch_seconds'],point['monotonic_seconds'])<=600
        return dict(mode='fixed1500_physical5',step=1500,formal5_eligible=True,
            checkpoint_identity=point['checkpoint_identity'],failed_budget=False)
    # A fixed time cap is a cost negative, not permission to add updates. Only
    # this explicit diagnostic may evaluate the final atomic saved prefix.
    assert a['status']=='failed' and e['exit_code']!=0
    assert a['error_type']=='TimeoutError'
    assert a['error'] in ('Fixed dual-clock phase/whole budget exhausted','External guard termination')
    assert a['phase'] in ('training','TRAIN_endpoint_readout')
    assert a['last_checkpoint'] is not None and a['last_checkpoint']['step']>0
    step=a['last_checkpoint']['step']
    assert 0<step<=a['optimizer_step_completions']<=1500
    assert e['actual_child_exited']
    return dict(mode='budget_limited_last_atomic_checkpoint',step=step,
        formal5_eligible=False,checkpoint_identity=a['last_checkpoint']['stat_identity'],
        failed_budget=True,partial_attempt_may_have_occurred=a['possible_partial_optimizer_update'])


def training_terminal():
    p=d.read(TRAIN);r=d.read(p['root_review']);f=d.path(p['output'])
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(p['scientific'])==p['scientific_digest']==r['scientific_digest']
    assert p['source_pins']==r['source_pins'] and all(d.sha(k)==h for k,h in p['source_pins'].items())
    a=d.read(f/'summary.json');e=d.read(f/'external_terminal.json')
    assert a==d.read(f/'status.json')
    assert a['execution']['run_process']==e['child_process'] and a['execution']['guard']['guard_process']==e['guard_process']
    assert a['execution']['scientific_digest']==e['scientific_digest']==p['scientific_digest']
    assert e['source_pins']==p['source_pins'] and all(d.exact_process_exited(e[k]) for k in ('child_process','guard_process'))
    endpoint=classify_endpoint(a,e,p['scientific'])
    d.stat_matches(endpoint['checkpoint_identity'])
    b=dict(plan=d.identity(TRAIN,hashed=True),terminal={k:d.identity(f/name,hashed=True)
        for k,name in [('summary','summary.json'),('status','status.json'),('external','external_terminal.json')]})
    return b,p,a,e,endpoint


def approved_contract(plan_path):
    p=d.read(plan_path);s=p['scientific'];r=d.read(p['root_review'])
    assert p['kind']=='GeoBC_GINO_physical_knownDEV60_v53'
    assert r['passed'] and r['whole_source_reviewed'] and r['actual_execution_released']
    assert d.digest(s)==p['scientific_digest']==r['scientific_digest'] and p['source_pins']==r['source_pins']
    assert all(d.sha(k)==h for k,h in p['source_pins'].items())
    assert set(REQUIRED_SOURCES)<=set(p['source_pins'])
    assert s['memory_limit_bytes']==30500000000 and s['whole_budget_seconds']==1800
    assert s['GPU_lock']=='runs/geobc_exploration/.coordinator.lock' and s['seed']==2026
    assert s['GT_policy']=='All60 full-node encoded/physical predictions committed and revalidated before shared knownGT60 once'
    assert not s['FINAL_reuse_for_selection'] and not s['automatic_continuation']
    assert s['fixed_prediction_denominator']==60 and s['decoder_chunk_size']==2048
    b,tp,a,e,endpoint=training_terminal()
    assert r['training_binding']==b and s['training_endpoint']==endpoint
    assert s['training_plan_identity']==b['plan'] and s['training_SCI']==tp['scientific_digest']
    assert all(p['source_pins'].get(k)==h for k,h in tp['source_pins'].items())
    m=d.read(check_identity(s['dataset_identity']))
    assert s['dataset_identity']==tp['scientific']['dataset_identity'] and s['scaler']==tp['scientific']['scaler']==m['scaler']
    keys=s['case_keys'];assert keys==m['development_keys'] and len(keys)==len(set(keys))==60
    assert all(m['case_roles'][k]=='DEV' for k in keys)
    known=d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert s['GT_manifest_identity']==known['GT_manifest_identity'] and keys==known['case_keys']
    truths=d.read(check_identity(s['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
    for key in keys:
        record=truths['records'][key]
        assert record['all_original_nodes'] and record['signed_stress_unchanged']
        assert record['physical']['shape']==[m['records'][key]['num_nodes'],4]
    geometries={}
    for key in keys:
        cad=key.split('/')[0];record=m['records'][key];item=record['file_identities']['xyz_path']
        assert record['scientific_group']=='DEV' and not {'physical_path','target_path'}&set(record['file_identities'])
        row=dict(cad_id=cad,num_nodes=record['num_nodes'],xyz_path=str(d.path(item['path'])),xyz_stat={k:item[k] for k in ('bytes','mtime_ns')})
        if cad in geometries:assert geometries[cad]==row
        geometries[cad]=row
    assert len(geometries)==20 and geometries==s['geometries']
    assert s['radius_plan']==dict(radius_cache='data/geobc_gino_radius_DEV20_v53',radius_pilot_summary='runs/geobc_exploration/gino_radius_input_pilot_101_479/summary.json')
    assert s['candidate']==dict(name=METHOD,family='GINO_PHYSICAL',step=endpoint['step'])
    for ref in s['scalar_references']:
        for item in ref['binding'].values():check_identity(item)
        old=d.read(ref['binding']['summary']['path']);oe=d.read(ref['binding']['external']['path'])
        assert old==d.read(ref['binding']['status']['path']) and old['status']=='completed' and old['protocol_qualified']
        assert old['closed'] and not old['active'] and oe['closed'] and not oe['active'] and oe['exit_code']==0 and oe['within_external_budget']
        assert old['execution']['scientific_digest']==oe['scientific_digest']==ref['SCI']
        assert all(d.exact_process_exited(oe[k]) for k in ('child_process','guard_process'))
        complete_physical(old['candidates'][ref['name']]['full60'],keys)
    assert [x['name'] for x in s['scalar_references']]==['FNO100_PHYSICAL_1500','R64_FNO_W16_1500','R16_TRANSOLVER_1500']
    assert p['output']=='runs/geobc_exploration/GINO100_knownDEV60_v53'
    return p,r,m,truths,dict(plan=tp,summary=a,external=e,endpoint=endpoint)
