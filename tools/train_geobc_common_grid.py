"""Fixed4800 common-information/capacity FNO/++ pair on cached TRAIN120."""
import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from geobc import available_cad_pilot_data as d

PLAN='notes/geobc_common_grid_train_plan.json'


def run():
    plan=d.read(PLAN); science=plan['scientific']; review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_training_released']
    assert d.digest(science)==plan['scientific_digest']==review['scientific_digest']
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    outer=json.loads(os.environ['GEOBC_COMMON_TRAIN_GUARD'])
    assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']); output.mkdir(parents=True,exist_ok=True)
    assert not (output/'execution.json').exists(), 'One fresh attempt only; recovery requires separate review'
    execution={'run_process':d.process_snapshot(os.getpid()),'guard':outer,'scientific_digest':plan['scientific_digest']}
    d.write(output/'execution.json',execution)
    result={'status':'running','active':True,'closed':False,'execution':execution,'arms':{},
            'DEV_target_reads':0,'raw_target_reads':0,'scaler_fits':0,'new_data_preparations':0,
            'cached_TRAIN_encoded_target_reads':0,'cached_TRAIN_physical_readout_reads':0,
            'pretrained_checkpoint_reads':0,'optimizer_steps':0,'actual_training_start_epoch':time.time()}
    start_epoch=time.time(); start_mono=time.monotonic(); lock=None
    train_deadline=None; readout_deadline=None
    def guard():
        remaining=min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())
        for deadline in (train_deadline,readout_deadline):
            if deadline: remaining=min(remaining,deadline[0]-time.time(),deadline[1]-time.monotonic())
        if remaining<=0:raise TimeoutError('Predeclared wall deadline reached')
    def persist():
        result.update(elapsed_epoch_seconds=time.time()-start_epoch,elapsed_monotonic_seconds=time.monotonic()-start_mono)
        d.write(output/'status.json',result)
    try:
        acceptance=d.read(science['probe_acceptance'])
        assert acceptance['passed'] and acceptance['actual_resource_and_active_gradient_qualified']
        probe=d.read(science['probe_plan'])
        assert acceptance['scientific_digest']==probe['scientific_digest']
        ext=d.read(d.path(probe['output'])/'external_terminal.json')
        assert all(d.exact_process_exited(ext[k]) for k in ('child_process','guard_process'))
        assert d.identity(d.path(probe['output'])/'summary.json',hashed=True)==acceptance['summary_identity']
        import fcntl
        lock=d.path(science['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch
        from geobc import coverage_comparison_data as data
        from geobc.coverage_comparison_train import case_schedule,physical_metrics,aggregate
        from geobc.common_grid_solver import CommonGridSolver,full_case_backward,parameter_gradients
        from geobc.transformers import close_owned_process_group
        from geobc.benchmark import FieldScaler
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        manifest=data.read_bootstrap_TRAIN40(science['parent_plan'])
        keys=manifest['train_keys']['40']; order=case_schedule(keys)
        assert keys==science['TRAIN_keys'] and manifest['scaler']==science['scaler'] and d.digest(order)==science['case_order_sha256']
        scaler=FieldScaler(**manifest['scaler']); shared=None
        def load(key,physical=False):
            guard(); case=data.load_case(manifest,key,device='cuda',for_fno=True)
            item=manifest['records'][key]['file_identities']['target_path'];data.stat_matches(item)
            case['target']=torch.as_tensor(np.array(np.load(data.path(item['path']),allow_pickle=False),copy=True),device='cuda')
            result['cached_TRAIN_encoded_target_reads']+=1
            if physical:
                item=manifest['records'][key]['file_identities']['physical_path'];data.stat_matches(item)
                case['physical']=np.load(data.path(item['path']),mmap_mode='r',allow_pickle=False)
                result['cached_TRAIN_physical_readout_reads']+=1
            return case
        def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
        for arm in science['arms']:
            guard(); random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
            folder=output/arm;folder.mkdir()
            model=CommonGridSolver(arm,2026).to('cuda').train()
            initial=model.shared_identity()
            if shared is None:shared=initial
            assert initial==shared
            row={'status':'training','step':0,'counts':model.counts(),'shared_init_sha256':initial,
                 'case_exposure_counts':{key:0 for key in keys},'training_entry_epoch':time.time(),
                 'training_entry_monotonic':time.monotonic(),'training_elapsed_epoch_seconds':0,
                 'training_elapsed_monotonic_seconds':0,'optimizer_steps':0,'checkpoint_writes':0}
            result['current_arm']=arm;result['arms'][arm]=row
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,fused=True)
            train_deadline=(row['training_entry_epoch']+18000,row['training_entry_monotonic']+18000)
            def save(step,filename):
                guard(); payload={'step':step,'arm':arm,'scientific_digest':plan['scientific_digest'],
                    'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng(),
                    'case_exposure_counts':dict(row['case_exposure_counts']),'cost_status':dict(row)}
                temporary=filename.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,filename)
                row['checkpoint_writes']+=1
            torch.cuda.reset_peak_memory_stats();persist()
            with (folder/'loss.jsonl').open('x',encoding='utf-8') as log:
                for step,key in enumerate(order,1):
                    guard(); case=load(key);optimizer.zero_grad(set_to_none=True)
                    loss,leaves=full_case_backward(model,case,guard)
                    if step==1:
                        grads=parameter_gradients(model)
                        expected=['backbone.embedding.weight','backbone.embedding.bias'] if arm=='PP256' else []
                        assert grads['operator']['missing']==expected
                        assert all(g['finite'] and g['L2']>0 for g in grads.values())
                        assert not grads['encoder']['missing'] and not grads['head']['missing']
                        assert all(v>0 for v in leaves.values())
                        row['first_actual_fullN_gradient_check']={'passed':True,'gradients':grads,'leaf_gradients':leaves,'expected_inactive_parameters':expected}
                    # Full-case loss finite and every optimized gradient finite.
                    for parameter in model.parameters():
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError('Nonfinite optimized gradient')
                    optimizer.step();torch.cuda.synchronize()
                    row['step']=step;row['optimizer_steps']+=1;row['case_exposure_counts'][key]+=1;result['optimizer_steps']+=1
                    row['training_elapsed_epoch_seconds']=time.time()-row['training_entry_epoch']
                    row['training_elapsed_monotonic_seconds']=time.monotonic()-row['training_entry_monotonic']
                    row['peak_allocated_bytes']=torch.cuda.max_memory_allocated();row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    assert max(row['peak_allocated_bytes'],row['peak_reserved_bytes'])<=24*2**30,'24GiB cap'
                    log.write(json.dumps({'step':step,'key':key,'encoded_MSE':loss,'training_epoch_seconds':row['training_elapsed_epoch_seconds']})+'\n')
                    log.flush();del case
                    if step%120==0:save(step,folder/'last.pt')
                    if step%30==0 or step==1:persist()
            assert row['step']==4800 and set(row['case_exposure_counts'].values())=={40}
            save(4800,folder/'step4800.pt');train_deadline=None
            row['status']='TRAIN120_readout';persist()
            readout_epoch=time.time();readout_mono=time.monotonic();readout_deadline=(readout_epoch+1200,readout_mono+1200)
            model.eval();per={key:None for key in keys}
            with torch.no_grad():
                for index,key in enumerate(keys):
                    guard();torch.manual_seed(2026+index);torch.cuda.manual_seed_all(2026+index)
                    case=load(key,physical=True);context,latent=model.encode_case(case)
                    pred=np.empty((len(case['xyz']),4),dtype=np.float32)
                    for start in range(0,len(pred),32768):
                        guard();stop=min(start+32768,len(pred));pred[start:stop]=model.query(case,start,stop,context,latent).cpu().numpy()
                    per[key]=physical_metrics(np,case['physical'],scaler.decode(pred))
                    del case,context,latent,pred
            row['TRAIN120']=aggregate(per,120);assert row['TRAIN120']['all_cases_complete']
            row['readout_elapsed_epoch_seconds']=time.time()-readout_epoch;row['readout_elapsed_monotonic_seconds']=time.monotonic()-readout_mono
            row.update(status='completed',checkpoint_identity=d.identity(folder/'step4800.pt',hashed=True))
            readout_deadline=None;persist()
            del model,optimizer;close_owned_process_group();torch.cuda.empty_cache()
        result.update(status='completed',current_arm=None,both4800=True,same_shared_init=True,
                      requires_separate_development_readout=True)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_replay=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-start_epoch,elapsed_monotonic_seconds=time.monotonic()-start_mono)
        d.write(output/'summary.json',result);d.write(output/'status.json',result)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
