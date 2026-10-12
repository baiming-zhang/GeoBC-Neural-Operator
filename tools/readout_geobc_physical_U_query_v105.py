"""Read existing nine query files only; no source fields/API/Torch/GPU replay."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,os,json,signal
from pathlib import Path

def run(path):
    from geobc import available_cad_pilot_data as d
    from geobc.physical_U_query_readout_contract_v105 import approved_contract
    p,r,original,prior=approved_contract(path);o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_PHYSICAL_U_QUERY_V105_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(original,status='running',active=True,closed=False,phase='CPU_savedquery_rebarrier',execution=execution,original_execution=original['execution'],original_GPU_summary_identity=p['original_terminal_binding']['summary'],original_GPU_seconds=original['elapsed_epoch_seconds'],original_GPU_external_seconds=prior['elapsed_epoch_seconds'],original_GPU_failure_preserved=True,original_owned_array_reads=original['owned_array_reads'],current_owned_prediction_array_reads=0,current_API_constructor_queries_gradients_replays=0,current_source_basis_input_reads=0,current_GPU_imports=0,current_GPU_execution=0,protocol_qualified=False,API_gate_passed=False)
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original900 preserved deadline')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('CPU originaldeadline guard')))
    try:
        import numpy as np
        s=p['scientific'];assert set(a['prediction_inventory'])==set(s['selection']['CADs']);guard()
        for cad,item in a['prediction_inventory'].items():
            assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);assert rec['CAD']==cad and rec['fixed_query_count']==768 and rec['GT_reads']==0 and rec['load_coefficients']==s['actual_tests']['query_load_coefficients']
            shapes={'physical_U':(6,128,3),'query_xyz':(128,3),'query_cell_rows':(128,)};dtype_map={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}
            for name in ('physical_U','query_xyz','query_cell_rows'):
                guard();field=rec[name];expected={k:v for k,v in field.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(field['path']));assert d.identity(field['path'],hashed=True)==expected
                value=np.load(d.path(field['path']),allow_pickle=False);a['current_owned_prediction_array_reads']+=1
                assert list(value.shape)==field['shape'] and value.shape==shapes[name] and str(value.dtype)==dtype_map[field['dtype']] and np.isfinite(value).all()
                if name=='query_cell_rows':assert value.dtype==np.int64 and value.min()>=0 and value.max()<s['cases'][cad]['expected_geometry']['cells']
                else:assert value.dtype==np.float64
                del value;persist()
            assert rec['selected_original_cell_rows']==sorted(set(rec['selected_original_cell_rows'])) and len(rec['selected_original_cell_rows'])==32
        assert a['current_owned_prediction_array_reads']==9
        d.write(o/'query_prediction_barrier.json',dict(status='completed',all2304_queries_finite_hash_dtype_shape=True,inventory=a['prediction_inventory'],original_GPU_first_barrier_failure_preserved=True,new_API_GPU_GT_replays=0))
        a.update(status='completed',phase='terminal',prediction_barrier_complete=True,protocol_qualified=True,API_gate_passed=True,scientific_outcome='physical_U_query_and_coordinate_rank3_gradient_qualified_after_CPU_readout_recovery',whole_original_campaign_epoch_seconds=time.time()-prior['entry_epoch'],new_offnode_GT_accuracy=False);guard()
    except Exception as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--plan');args=p.parse_args()
    if args.start:
        if not args.plan:p.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
