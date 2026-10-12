"""Frozen three-model prediction-first comparison on the two unused reserves.

Default is inert; numerical libraries/data are accessed only by --start under
a reviewed plan and independent dual-clock guard. No training or selection.
"""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.reserve_holdout_contract_v93 import approved
    p=approved(plan_path);s=p['scientific'];out=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_RESERVE_HOLDOUT_V93_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest']
    assert outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    assert not (out/'execution.json').exists();d.write(out/'execution.json',execution)
    r=dict(status='running',active=True,closed=False,phase='target_free_preparation',execution=execution,
        candidates={},prediction_inventory={},committed_finite_predictions=0,source_GT_parse_attempts=0,
        source_GT_parses=0,physical_case_reads=0,checkpoint_hash_reads=0,checkpoint_loads=0,
        training_updates=0,scaler_fits=0,old_DEV_FINAL_GT_reads=0,
        peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0,
        fixed_case_denominator=6,fixed_prediction_denominator=18,protocol_qualified=False,
        final_solver_selected=False,automatic_continuation=False)
    torch=None;lock=None;model=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Frozen900-second whole dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            r['peak_allocated_bytes']=max(r['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            r['peak_reserved_bytes']=max(r['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            r['peak_total_device_used_bytes_sampled']=max(r['peak_total_device_used_bytes_sampled'],total-free)
            if max(r[k] for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['memory_limit_bytes']:
                raise MemoryError('30.5GB memory cap')
    def persist():
        r.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'status.json',r)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        import fcntl
        lock=d.path(s['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from geobc import coverage_comparison_data as original
        from geobc.macro_training import encode_case,query_case
        from geobc.benchmark import FieldScaler
        from geobc.macro_readout_protocol import verify_barrier
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device();total=torch.cuda.get_device_properties(device).total_memory
        assert s['memory_limit_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['memory_limit_bytes']/total,device);torch.cuda.reset_peak_memory_stats()
        contract=d.read(s['input_contract_identity']['path']);assert contract['groups']['train_new160']==[]
        ledger=dict(attempted_reads={},completed_reads={},numeric_GT_parses=0)
        class Ledger:
            value=ledger
            def attempt(self,filename,member=''):
                guard();assert member!='known3_U_S'
                k=str(filename)+':'+member;ledger['attempted_reads'][k]=ledger['attempted_reads'].get(k,0)+1
                d.write(out/'input_read_receipt.json',ledger)
            def complete(self,filename,counter=None,member=''):
                guard();assert counter is None or counter in d.INPUT_COUNTERS
                k=str(filename)+':'+member;ledger['completed_reads'][k]=ledger['completed_reads'].get(k,0)+1
                if counter:ledger[counter]=ledger.get(counter,0)+1
                d.write(out/'input_read_receipt.json',ledger)
        records={};geometry={}
        for cad in s['CADs']:
            guard();item=original._prepare_CAD(contract,{'scientific':{'recipe':{'device':'cuda'}}},cad,out/'inputs',Ledger(),lambda x:guard())
            assert item['status']=='completed' and item['feasibility_pass'] and item['DEV_target_reads']==0
            geometry[cad]=item;records.update(item['records'])
        assert set(records)==set(s['case_keys']) and ledger['numeric_GT_parses']==0
        assert all(not {'physical_path','target_path'}&set(x['file_identities']) for x in records.values())
        d.write(out/'input_manifest.json',dict(status='completed',records=records,geometry=geometry,targets_read=0))
        r['phase']='prediction_commit';persist();scaler=FieldScaler(**s['scaler']);inventory={};specs=[]
        def load_input(key):
            record=records[key];case={'key':key};n=record['num_nodes']
            shapes={'xyz':(n,3),'features':(n,25),'cloud':(1024,3),'mask':(1024,),
                'grid':(29,64,64,64),'query_indices':(n,8),'query_weights':(n,8)}
            for name,shape in shapes.items():
                guard();ident=record['file_identities'][name+'_path'];d.stat_matches(ident)
                value=np.load(d.path(ident['path']),allow_pickle=False)
                assert value.shape==shape
                if name=='mask':assert value.dtype==np.bool_
                elif name=='query_indices':assert value.dtype in (np.int32,np.int64) and value.min()>=0 and value.max()<64**3
                else:assert value.dtype==np.float32 and np.isfinite(value).all()
                case[name]=torch.as_tensor(value,device='cuda')
            case['cloud']=case['cloud'][None];case['mask']=case['mask'][None];case['grid']=case['grid'][None]
            return case
        def save_array(file,array):
            guard();ident=original._save(np,file,array);guard();return ident
        for candidate in s['candidates']:
            guard();name=candidate['name'];family=candidate['family'];cp=candidate['checkpoint']
            assert d.identity(cp['path'],hashed=True)==cp;r['checkpoint_hash_reads']+=1
            saved=torch.load(d.path(cp['path']),map_location='cpu',weights_only=False);r['checkpoint_loads']+=1
            tp=d.read(candidate['training_plan_identity']['path']);ts=tp['scientific']
            assert saved['family']==family and saved['step']==saved['cursor']==6000
            assert saved['scientific_digest']==candidate['training_SCI'] and saved['scaler']==s['scaler'] and saved['dataset_identity']==s['dataset_identity']
            assert saved['case_exposure_counts']=={k:20 for k in ts['TRAIN_keys']} and set(saved['rng'])=={'python','numpy','torch','cuda'}
            assert saved['optimizer']['state'] and all(int(x['step'])==6000 for x in saved['optimizer']['state'].values())
            assert saved['full_order_digest']==candidate['full_order_digest']
            checkpoint_boot=saved['native_origin_clocks']['boot_id'] if family=='FNO9' else saved['execution']['boot_id']
            assert checkpoint_boot==candidate['training_boot_id']
            if family=='FNO9':from geobc.macro_training import build_solver
            elif family=='FNO_GLOBAL_HASH':from geobc.operator_hash_metadata_v6 import build_solver
            else:from geobc.transolver_bf16_hash_v65 import build_solver
            model=build_solver(family,seed=2026);model.load_state_dict(saved['model'],strict=True)
            loaded=model.state_dict();assert set(loaded)==set(saved['model'])
            assert all(loaded[k].shape==saved['model'][k].shape and loaded[k].dtype==saved['model'][k].dtype and torch.equal(loaded[k].cpu(),saved['model'][k]) for k in loaded)
            del loaded,saved;model.eval()
            row=dict(family=family,checkpoint=cp,strict_checkpoint_verified=True,cases=0,inference_epoch_seconds=0.,inference_monotonic_seconds=0.,parameter_count=sum(x.numel()for x in model.parameters()))
            r['candidates'][name]=row;spec=dict(name=name,family=family,step=6000,checkpoint=cp);specs.append(spec)
            with torch.no_grad():
                for index,key in enumerate(s['case_keys']):
                    guard();case=load_input(key);n=records[key]['num_nodes'];torch.manual_seed(2026+index);torch.cuda.manual_seed_all(2026+index)
                    torch.cuda.synchronize();began=(time.time(),time.monotonic());context,latent=encode_case(model,case)
                    pred=np.empty((n,4),np.float32)
                    for start in range(0,n,32768):
                        guard();stop=min(start+32768,n);pred[start:stop]=query_case(model,case,start,stop,context,latent).cpu().numpy()
                    torch.cuda.synchronize();row['inference_epoch_seconds']+=time.time()-began[0];row['inference_monotonic_seconds']+=time.monotonic()-began[1]
                    physical=np.asarray(scaler.decode(pred),np.float32)
                    assert np.isfinite(pred).all() and np.isfinite(physical).all()
                    folder=out/'predictions'/name/key
                    record=dict(**spec,candidate=name,key=key,status='completed',all_original_nodes=n,signed_stress_unchanged=True,
                        encoded=save_array(folder/'encoded4.npy',pred),physical=save_array(folder/'physical4.npy',physical))
                    file=folder/'manifest.json';d.write(file,record);inventory[name+':'+key]=d.identity(file,hashed=True)
                    r['committed_finite_predictions']+=1;row['cases']+=1;persist();del case,context,latent,pred,physical
            d.stat_matches(cp);del model;model=None;torch.cuda.empty_cache();guard()
        def record_reader(item):
            guard();assert d.identity(item['path'],hashed=True)==item;return d.read(item['path'])
        def array_reader(item):
            guard();assert d.identity(item['path'],hashed=True)=={k:v for k,v in item.items() if k in ('path','bytes','mtime_ns','sha256')}
            return np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False)
        counts={k:v['num_nodes']for k,v in records.items()}
        barrier=verify_barrier(np,specs,s['case_keys'],counts,inventory,record_reader,array_reader,guard)
        assert barrier['finite_predictions']==18 and r['source_GT_parses']==r['source_GT_parse_attempts']==0
        d.write(out/'prediction_barrier.json',barrier);r.update(prediction_barrier_complete=True,prediction_inventory=inventory,phase='new_reserve_GT_once');persist()
        per={x['name']:{}for x in specs}
        for cad in s['CADs']:
            guard();source=s['raw_sources'][cad]['Field'];d.stat_matches(source)
            permutation=np.load(d.path(geometry[cad]['geometry']['h5_row_to_csv']['path']),allow_pickle=False)
            r['source_GT_parse_attempts']+=1;persist()
            truth12=original._targets(np,source['path'],original._columns(source['path']),permutation,geometry[cad]['num_nodes'])
            r['source_GT_parses']+=1
            for index,load in enumerate(s['loads']):
                key=cad+'/'+load;truth=truth12[:,4*index:4*index+4];r['physical_case_reads']+=1
                for spec in specs:
                    record=record_reader(inventory[spec['name']+':'+key]);pred=array_reader(record['physical'])
                    per[spec['name']][key]=physical_metrics(np,truth,pred)
                persist()
            del truth12;guard()
        assert r['source_GT_parse_attempts']==r['source_GT_parses']==2 and r['physical_case_reads']==6
        for spec in specs:
            name=spec['name'];r['candidates'][name]['full6']=aggregate(per[name],6)
            r['candidates'][name]['per_load2']={load:aggregate({k:per[name][k]for k in s['case_keys']if k.endswith('/'+load)},2)for load in s['loads']}
        r.update(status='completed',phase='terminal',protocol_qualified=True)
    except BaseException as error:
        r.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        r.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(out/'summary.json',r);d.write(out/'status.json',r)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","GT":0,"GPU":0}')
