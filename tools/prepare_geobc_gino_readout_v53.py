"""Freeze real terminal metadata only; root runtime release stays separate."""
import argparse
import datetime

PLAN='notes/geobc_gino_knownDEV_v53_plan.json'
REVIEW='notes/geobc_gino_knownDEV_v53_root_review.json'
SOURCE_REVIEW='notes/geobc_gino_knownDEV_v53_source_review.json'


def prepare():
    from geobc import available_cad_pilot_data as d
    from geobc.gino_readout_contract_v53 import training_terminal,REQUIRED_SOURCES,METHOD,check_identity,complete_physical
    # Must fail before opening arrays, checkpoints, or writing a new plan if
    # the actual training child/guard has not ended with a closed receipt.
    binding,tp,a,e,endpoint=training_terminal()
    sr=d.read(SOURCE_REVIEW)
    assert sr['passed'] and sr['whole_source_reviewed'] and not sr['actual_execution_released']
    assert all(d.sha(k)==h for k,h in sr['source_pins'].items())
    assert all(sr['source_pins'].get(k)==h for k,h in tp['source_pins'].items())
    assert set(REQUIRED_SOURCES)<=set(sr['source_pins'])
    assert not d.path(PLAN).exists() and not d.path(REVIEW).exists(), 'One immutable prepared endpoint'
    s0=tp['scientific'];m=d.read(check_identity(s0['dataset_identity']))
    keys=m['development_keys'];assert len(keys)==len(set(keys))==60
    known=d.read('notes/geobc_exposure_readout_plan.json')['scientific']
    assert keys==known['case_keys']
    truths=d.read(check_identity(known['GT_manifest_identity']))
    assert truths['status']=='completed' and truths['source_GT_parses']==20
    assert set(truths['records'])==set(keys)
    geometries={}
    for key in keys:
        record=m['records'][key];cad=key.split('/')[0]
        assert m['case_roles'][key]==record['scientific_group']=='DEV'
        assert not {'physical_path','target_path'}&set(record['file_identities'])
        xyz=record['file_identities']['xyz_path'];d.stat_matches(xyz)
        row=dict(cad_id=cad,num_nodes=record['num_nodes'],xyz_path=str(d.path(xyz['path'])),
            xyz_stat={k:xyz[k] for k in ('bytes','mtime_ns')})
        if cad in geometries:assert geometries[cad]==row
        geometries[cad]=row
        gt=truths['records'][key]
        assert gt['all_original_nodes'] and gt['signed_stress_unchanged']
        assert gt['physical']['shape']==[record['num_nodes'],4]
    assert len(geometries)==20
    refs=[]
    for file,name in [('notes/geobc_physical100_readout_v12_plan.json','FNO100_PHYSICAL_1500'),
        ('notes/geobc_wide_fno_readout_v25_plan.json','R64_FNO_W16_1500'),
        ('notes/geobc_coarse_three_readout_v21_plan.json','R16_TRANSOLVER_1500')]:
        rp=d.read(file);out=d.path(rp['output'])
        old=d.read(out/'summary.json');oe=d.read(out/'external_terminal.json')
        assert old==d.read(out/'status.json') and old['status']=='completed' and old['protocol_qualified']
        assert old['closed'] and not old['active'] and oe['closed'] and not oe['active']
        assert oe['exit_code']==0 and oe['within_external_budget']
        assert old['execution']['scientific_digest']==oe['scientific_digest']==rp['scientific_digest']
        assert all(d.exact_process_exited(oe[k]) for k in ('child_process','guard_process'))
        complete_physical(old['candidates'][name]['full60'],keys)
        refs.append(dict(name=name,SCI=rp['scientific_digest'],binding={
            k:d.identity(out/f,hashed=True) for k,f in [('summary','summary.json'),
                ('status','status.json'),('external','external_terminal.json')]}))
    science=dict(training_plan_identity=binding['plan'],training_SCI=tp['scientific_digest'],
        training_endpoint=endpoint,dataset_identity=s0['dataset_identity'],scaler=s0['scaler'],
        GT_manifest_identity=known['GT_manifest_identity'],case_keys=keys,geometries=geometries,
        radius_plan=dict(radius_cache='data/geobc_gino_radius_DEV20_v53',
            radius_pilot_summary='runs/geobc_exploration/gino_radius_input_pilot_101_479/summary.json'),
        candidate=dict(name=METHOD,family='GINO_PHYSICAL',step=endpoint['step']),scalar_references=refs,
        seed=2026,decoder_chunk_size=2048,memory_limit_bytes=30500000000,whole_budget_seconds=1800,
        GPU_lock='runs/geobc_exploration/.coordinator.lock',fixed_prediction_denominator=60,
        GT_policy='All60 full-node encoded/physical predictions committed and revalidated before shared knownGT60 once',
        FINAL_reuse_for_selection=False,automatic_continuation=False,
        scope='KnownDEV iterative single-seed physical signed U(mm)/S(MPa), complete native nodes; actual budget prefix is diagnostic only, not equal5 exposure; no final solver or off-node FEM claim')
    plan=dict(kind='GeoBC_GINO_physical_knownDEV60_v53',scientific=science,
        scientific_digest=d.digest(science),source_pins=sr['source_pins'],root_review=REVIEW,
        output='runs/geobc_exploration/GINO100_knownDEV60_v53',
        prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    review=dict(passed=False,whole_source_reviewed=True,actual_execution_released=False,
        scientific_digest=plan['scientific_digest'],source_pins=plan['source_pins'],
        training_binding=binding,source_review_identity=d.identity(SOURCE_REVIEW,hashed=True),
        status='Metadata prepared; exact terminal endpoint requires genuine root release before any runtime')
    d.write(PLAN,plan);d.write(REVIEW,review)
    return dict(status='metadata_prepared_not_runtime_released',plan=PLAN,
        scientific_digest=plan['scientific_digest'],endpoint=endpoint['mode'],step=endpoint['step'],
        formal5_eligible=endpoint['formal5_eligible'],GT_array_reads=0,PT_body_reads=0,
        CUDA_initialized=False,training_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:
        import json
        print(json.dumps(prepare()))
    else:print('{"status":"source_only","job_reads":0,"writes":0,"GPU_execution":0}')
