"""CPU-only strict dtype-namespace adapter, no K/PCG/input/model replay."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse,json,os,signal
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.load_linear_physics_readout_contract_v101 import approved_contract
    p,r,parent,truths=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output']);outer=json.loads(os.environ['GEOBC_LOAD_LINEAR_PHYSICS_READOUT_V101_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=p['scientific_digest'],boot_id=outer['boot_id']);d.write(o/'execution.json',execution)
    a=dict(parent);a.update(status='running',active=True,closed=False,phase='revalidate6_fields_beforeGT',execution=execution,original_GPU_execution=parent['execution'],original_GPU_summary_identity=p['original_summary_identity'],original_GPU_external_identity=p['original_external_identity'],original_GPU_seconds=parent['elapsed_epoch_seconds'],original_owned_array_reads_before_failure=parent['owned_array_reads'],owned_array_reads=0,current_geometry_input_or_K_or_PCG_replay=False)
    a.pop('error_type',None);a.pop('error',None);lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original1800 UTC and conservatively carriedmono deadline')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External physics CPU readout guard')))
    try:
        import fcntl
        import numpy as np
        lock=d.path(s['budget']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);guard();persist()
        a['phase']='all6_fields_equilibrium_barrier_GTclosed';persist();readout=(time.time(),time.monotonic())
        def read_array(item):
            guard();expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']));assert d.identity(item['path'],hashed=True)==expected
            value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False);a['owned_array_reads']+=1;assert list(value.shape)==item['shape']and str(value.dtype)=={'torch.float64':'float64','torch.float32':'float32'}.get(item['dtype'],item['dtype'])and np.isfinite(value).all();return value
        n=s['expected_geometry']['nodes'];fields={};endpoint_records={}
        for name in ['ver','hor','tor','mix0','mix1','mix2']:
            item=a['prediction_inventory'][name];assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);assert rec['all_original_nodes']==n and rec['initial_guess']=='independent_zero'and rec['key']==name and rec['status']=='completed'
            value=read_array(rec['physical_U']);assert value.shape==(n,3)and value.dtype==np.float64;fields[name]=value;endpoint_records[name]=rec
        d.write(o/'physics_prediction_barrier.json',dict(status='completed',all6_original_nodes=True,GT_reads_before_barrier=0,inventory=a['prediction_inventory'],endpoint_numerical_states={k:v['numerical_state']for k,v in endpoint_records.items()}));a['prediction_barrier_complete']=True;persist()
        def metrics(pred,ref):
            norm=float(np.square(ref).sum());sse=float(np.square(pred-ref).sum());return dict(U_relative_L2=(sse/norm)**.5 if norm>0 else None,U_RMS_mm=(sse/len(ref))**.5,U_max_mm=float(np.linalg.norm(pred-ref,axis=1).max()),fixed_allN=len(ref),zero_reference=norm<=0)
        canonical={};a['phase']='shared_original_GT3_once';persist();cad=s['selection']['CAD']
        for load in ('ver','hor','tor'):
            key=cad+'/'+load;item=truths['records'][key]['physical'];guard();a['GT_attempts']+=1;gt=read_array(item);a['GT_reads']+=1;assert gt.shape==(n,4)and gt.dtype==np.float32
            canonical[key]=metrics(fields[load],gt[:,:3].astype(np.float64));del gt;persist()
        newload={}
        for candidate in ('FNO20','FNO_HASH20','TS20'):
            newload[candidate]={}
            for kind in ('raw','adapter'):
                rows={}
                for i in range(3):
                    key=cad+'/mix'+str(i)+'@'+kind;item=s['neural_prediction_inventory'][candidate+':'+key];assert d.identity(item['path'],hashed=True)==item;rec=d.read(item['path']);pred=read_array(rec['physical']);assert pred.shape==(n,4);rows[key]=metrics(pred[:,:3].astype(np.float64),fields['mix'+str(i)]);del pred
                vals=[v['U_relative_L2']for v in rows.values()];complete=len(vals)==3 and all(v is not None and np.isfinite(v)for v in vals);newload[candidate][kind]=dict(fixed_denominator=3,per_case=rows,all3_complete=complete,mean_U=sum(vals)/3 if complete else None,worst_U=max(vals)if complete else None,reference='independently_solved_approximate_curvedP2_fourQ_not_originalOptiStruct_newloadGT')
        a['endpoint_records']=endpoint_records;a['canonical_original_GT_U']=canonical;a['mixed_neural_U_against_independent_approx_P2']=newload
        allconv=all(v['numerical_state']=='converged'and v['true_relative_residual']<=1e-8 for v in endpoint_records.values());canpass=len(canonical)==3 and all(v['U_relative_L2']is not None and v['U_relative_L2']<=.01 for v in canonical.values())
        a['physics_reference_compatible']=allconv and canpass;a['reference_compatibility_gate']=dict(passed=a['physics_reference_compatible'],all6_true_residual_pass=allconv,canonical3_le1percent=canpass,does_not_claim_OptiStruct_equivalence=True,does_not_select_final_solver=True)
        a['final_readout_seconds']=max(time.time()-readout[0],time.monotonic()-readout[1]);assert a['final_readout_seconds']<=300 and a['GT_attempts']==a['GT_reads']==3
        a['old_and_batch_operator_calls']=None;a['original_K_call_total_not_persisted']=True;a['current_new_K_calls']=0;guard();a.update(status='completed',phase='terminal',protocol_qualified=True,scientific_outcome='compatible_approximate_physics_reference'if a['physics_reference_compatible']else'finite_completed_reference_compatibility_negative')
    except BaseException as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires fixed --plan')
        run(args.plan)
    else:print('{"status":"source_only","arrays_GT_GPU":0}')
