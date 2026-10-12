"""Single CPU source qualification of complete main TRAIN100/300, no target reads."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal
from pathlib import Path
def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.main_train_source_contract_v122 import approved_contract
    p,r,inventory=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_MAIN_TRAIN_SOURCE_V122_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='source',execution=execution,deck_hash_reads=0,H5_hash_reads=0,H5_input_reads=0,GRID_archive_hash_reads=0,GRID_archive_member_reads=0,permutation_reads=0,CSV_hash_reads=0,CSV_header_reads=0,CSV_GRID_XYZ_reads=0,legacy_cache_reads=0,deck_parses=0,deck_header_reads=0,K_PCG_GPU_reads=0,raw_U_S_target_reads=0,GT_reads=0,DEV_FINAL_reads=0,neural_PT_reads=0,training_updates=0,scaler_fits=0,fixed_CAD_denominator=100,fixed_load_denominator=300,committed_CADs=0,case_records={},source_barrier_complete=False,main300_kinematic_inputs_ready=False)
    lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Whole1800 earlier deadline')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External source guard')))
    try:
        import fcntl,numpy as np
        from geobc.main_train_physics_sources_v122 import prepare
        lock=d.path(s['budgets']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for cad,row in inventory['rows'].items():
            guard();a['phase']=cad+'/targetfree_source';persist();began=(time.time(),time.monotonic())
            arrays,forces,facts=prepare(np,cad,row,s,a,guard,began[1]);guard();folder=o/cad;folder.mkdir(exist_ok=False);cache=folder/'geometry_and_canonical_rhs.npz'
            with cache.open('xb') as f:np.savez(f,**arrays,canonical_forces=forces)
            item=d.identity(cache,hashed=True);schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in dict(arrays,canonical_forces=forces).items()}
            assert forces.shape==(3,row['nodes'],3) and np.isfinite(forces).all()
            rec=dict(status='completed',scientific_group='TRAIN',input_identity=item,array_schema=schema,source_qualification=facts,actual_source_wall_epoch=time.time()-began[0],actual_source_wall_monotonic=time.monotonic()-began[1],K_GPU_PCG_or_target_reads=0)
            assert max(rec['actual_source_wall_epoch'],rec['actual_source_wall_monotonic'])<=300
            d.write(folder/'source_manifest.json',rec);a['case_records'][cad]=dict(manifest_identity=d.identity(folder/'source_manifest.json',hashed=True),input_identity=item,expected_geometry=facts['expected_geometry'],actual_source_wall_epoch=rec['actual_source_wall_epoch'],actual_source_wall_monotonic=rec['actual_source_wall_monotonic'],inherited_geometry=cad in s['legacy_inputs'])
            a['committed_CADs']+=1;del arrays,forces;persist()
        assert a['committed_CADs']==100 and a['legacy_cache_reads']==4 and a['H5_input_reads']==96 and a['CSV_GRID_XYZ_reads']==60 and a['GRID_archive_member_reads']==36
        expected=set(s['selection']['CADs']);assert set(a['case_records'])==expected
        for cad,rec in a['case_records'].items():
            guard();assert d.identity(rec['manifest_identity']['path'],hashed=True)==rec['manifest_identity'];d.stat_matches(rec['input_identity'])
        d.write(o/'source_barrier.json',dict(status='completed',source_CADs=100,canonical_loads=300,all_original_node_source_schemas_qualified=True,source_manifest_inventory=a['case_records'],numerical_K_PCG_feature_qualification=False,GT_reads=0))
        a.update(status='completed',phase='terminal',source_barrier_complete=True,source_qualification_passed=True,main300_kinematic_inputs_ready=False)
    except Exception as error:
        from geobc.elastic_pilot_data import UnsupportedInput
        if isinstance(error,UnsupportedInput):a.update(status='completed',phase='terminal_scientific_input_negative',source_qualification_passed=False,reason=str(error),missing_CADs=100-a['committed_CADs'])
        else:a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"SOURCE_only","scientific_body_GPU_GT":0}')
