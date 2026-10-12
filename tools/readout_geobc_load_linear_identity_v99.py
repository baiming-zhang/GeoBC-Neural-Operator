"""CPU-only restore of540 prediction barrier/preservation and GT60 readout."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.load_linear_identity_readout_contract_v99 import approved_contract
    p,r,parent,truths=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_LOAD_LINEAR_IDENTITY_V99_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']and d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    d.write(o/'execution.json',execution)
    a=dict(parent);a.update(status='running',active=True,closed=False,phase='revalidate540_GT_still_closed',execution=execution,original_failure_summary_identity=p['original_summary_identity'],original_failure_external_identity=p['original_external_identity'],original_failure_cost=dict(native_seconds=parent['elapsed_epoch_seconds'],old_array_hash_attempts=parent['own_array_hash_attempts'],old_array_reads=parent['own_array_reads']),own_array_hash_attempts=0,own_array_reads=0,owned_GT_attempts=0,GT_reads=0)
    a.pop('error_type',None);a.pop('error',None)
    lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Original absolute1800 deadline retained')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External CPU recovery guard')))
    try:
        import fcntl
        import numpy as np
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.macro_readout_protocol import verify_barrier
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard();inventory=parent['prediction_inventory'];cads=sorted({k.split('/')[0]for k in s['case_keys']});keys=[];counts={};specs=[]
        for cad in cads:
            keys.extend(cad+'/'+load for load in ('ver','hor','tor'));keys.extend(cad+'/mix'+str(i)+'@'+kind for i in range(3)for kind in ('raw','adapter'))
        for c in s['frozen_candidates']:specs.append(dict(name=c['name'],family=c['family'],step=6000,checkpoint=c['checkpoint']))
        def record_reader(item):
            guard();actual=d.identity(item['path'],hashed=True);expected=dict(item);expected['path']=d.relative(d.path(item['path']));assert actual==expected;return d.read(item['path'])
        def array_reader(item):
            guard();a['own_array_hash_attempts']+=1;actual=d.identity(item['path'],hashed=True)
            expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
            assert actual==expected,'Strict normalized path/bytes/mtime/SHA differs'
            value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False);a['own_array_reads']+=1
            assert list(value.shape)==item['shape']and str(value.dtype)==item['dtype']and np.isfinite(value).all();return value
        for key in keys:counts[key]=record_reader(inventory[specs[0]['name']+':'+key])['all_original_nodes']
        barrier=verify_barrier(np,specs,keys,counts,inventory,record_reader,array_reader,guard);assert barrier['finite_predictions']==540
        d.write(o/'prediction_barrier.json',barrier);a.update(prediction_barrier_complete=True,phase='canonical_preservation_GT_still_closed');persist()
        def udefect(value,ref):
            norm=float(np.square(ref).sum());sse=float(np.square(value-ref).sum());return (sse/norm)**.5 if norm>0 else None
        # Preservation and same requested-load rawS checked before any GT read.
        for c in s['frozen_candidates']:
            name=c['name'];row=a['candidates'][name];pres={};stress={}
            old_inventory=s['reference_inventory'][name]
            for key in s['case_keys']:
                rec=record_reader(inventory[name+':'+key]);old_item=next(v for k,v in old_inventory.items()if k.split(':',1)[1]==key);oldrec=record_reader(old_item)
                pred=array_reader(rec['physical']);ref=array_reader(oldrec['physical']);pres[key]=udefect(pred[:,:3].astype(np.float64),ref[:,:3].astype(np.float64));del pred,ref
                newencoded=array_reader(rec['encoded']);oldencoded=array_reader(oldrec['encoded']);stress[key]=float(np.abs(newencoded[:,3]-oldencoded[:,3]).max());del newencoded,oldencoded
            for cad in cads:
                for i in range(3):
                    raw=record_reader(inventory[name+':'+cad+'/mix'+str(i)+'@raw']);new=record_reader(inventory[name+':'+cad+'/mix'+str(i)+'@adapter']);re=array_reader(raw['encoded']);ne=array_reader(new['encoded']);diff=float(np.abs(re[:,3]-ne[:,3]).max());stress[cad+'/mix'+str(i)]=diff;del re,ne
            complete=len(pres)==60 and all(x is not None and np.isfinite(x)for x in pres.values())
            stol=s['canonical_encodedS_FP32_tolerance']if c['family']=='FNO9'else s['canonical_encodedS_BF16_tolerance']
            row['canonical_preservation']=dict(per_case_U_difference=pres,per_case_encodedS_maxabs=stress,all60_complete=complete,worst_U_difference=max(pres.values())if complete else None,worst_encodedS_difference=max(stress.values()),passed=complete and max(pres.values())<=s['canonical_U_relative_tolerance']and max(stress.values())<=stol)
            persist()
        assert all(row['canonical_preservation']['passed']for row in a['candidates'].values()),'Canonical preservation fails; GT stays closed'
        a['phase']='GT60_once_conditional_mixedU';persist()
        canonical={c['name']:{}for c in s['frozen_candidates']};mixed={c['name']:{'raw':{},'adapter':{}}for c in s['frozen_candidates']}
        # Each CAD's3 actual GT arrays read once, retained only for its3 mixtures.
        for cad in cads:
            actual=[]
            for load in ('ver','hor','tor'):
                key=cad+'/'+load;item=truths['records'][key]['physical'];guard();a['owned_GT_attempts']+=1
                expected_GT={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')}
                expected_GT['path']=d.relative(d.path(item['path']))
                assert d.identity(item['path'],hashed=True)==expected_GT
                gt=np.load(d.path(item['path']),allow_pickle=False);a['GT_reads']+=1
                assert gt.shape==(counts[key],4)and gt.dtype==np.float32 and np.isfinite(gt).all();actual.append(gt)
                for c in s['frozen_candidates']:
                    name=c['name'];rec=record_reader(inventory[name+':'+key]);pred=array_reader(rec['physical']);canonical[name][key]=physical_metrics(np,gt,pred);del pred
                a['partial_canonical_metrics']=canonical;persist()
            for i,coeff in enumerate(s['coefficients']):
                conditional_u=sum(coeff[j]*actual[j][:,:3].astype(np.float64)for j in range(3))
                for c in s['frozen_candidates']:
                    name=c['name']
                    for kind in ('raw','adapter'):
                        key=cad+'/mix'+str(i)+'@'+kind;rec=record_reader(inventory[name+':'+key]);pred=array_reader(rec['physical'])[:,:3].astype(np.float64);res=pred-conditional_u
                        mixed[name][kind][cad+'/mix'+str(i)]=dict(u_relative_l2=udefect(pred,conditional_u),u_RMS_mm=float((np.square(res).sum()/len(res))**.5),u_max_mm=float(np.linalg.norm(res,axis=1).max()),reference='conditional_linear_static_derived_U_not_independent_new_FEM')
                        del pred,res
            a['partial_conditional_mixed_U_metrics']=mixed
            del actual,conditional_u;guard();persist()
        assert a['GT_reads']==a['owned_GT_attempts']==60
        for c in s['frozen_candidates']:
            name=c['name'];row=a['candidates'][name];row['canonical_full60']=aggregate(canonical[name],60)
            row['canonical_per_load20']={load:aggregate({k:v for k,v in canonical[name].items()if k.endswith('/'+load)},20)for load in ('ver','hor','tor')}
            row['conditional_mixed_U']={}
            for kind,per in mixed[name].items():
                vals=[x['u_relative_l2']for x in per.values()];complete=len(vals)==60 and all(v is not None and np.isfinite(v)for v in vals)
                row['conditional_mixed_U'][kind]=dict(fixed_denominator=60,per_case=per,all60_complete=complete,mean_U=sum(vals)/60 if complete else None,worst_U=max(vals)if complete else None,per_mix20={str(i):sum(x['u_relative_l2']for k,x in per.items()if k.endswith('/mix'+str(i)))/20 if complete else None for i in range(3)})
            raw=row['conditional_mixed_U']['raw'];adapt=row['conditional_mixed_U']['adapter'];complete=raw['all60_complete']and adapt['all60_complete']and row['canonical_full60']['all_cases_complete'];gain=raw['mean_U']-adapt['mean_U']if complete else None
            row['conditional_macro_gate']=dict(passed=complete and gain>=.03 and row['canonical_preservation']['passed'],U_improvement=gain,threshold=.03,not_independent_new_FEM=True,does_not_select_final_solver=True)
            row['paired_CAD20']={cad:sum(mixed[name]['adapter'][cad+'/mix'+str(i)]['u_relative_l2']-mixed[name]['raw'][cad+'/mix'+str(i)]['u_relative_l2']for i in range(3))/3 if complete else None for cad in cads}
        assert a['committed_finite_predictions']==540 and a['checkpoint_hash_reads']==a['checkpoint_loads']==3
        a.update(status='completed',phase='terminal',protocol_qualified=True,input_tensor_bytes_unchanged=None,resident_final_input_digest_check_reached=False,original_checkpoint_reads=3,current_checkpoint_reads=0,current_input_reads=0,current_inference_replayed=False,original_canonical_GT_accuracy_measured=True,independent_newload_FEM_GT_accuracy_measured=False,conditional_derived_mixed_U_reference_used=True);guard()
    except BaseException as error:
        a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","GT_PT_arrays_GPU":0}')
