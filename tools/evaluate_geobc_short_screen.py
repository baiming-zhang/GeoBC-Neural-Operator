"""Independent short-screen readout: both12000 endpoints, all120 predictions before knownDEV60."""
import argparse
import json
import os
import time
from geobc import available_cad_pilot_data as d

PLAN='notes/geobc_short_screen_readout_plan.json'


def parent_qualified(plan,review):
    parent=d.read(plan['scientific']['parent_plan'])
    stop=d.read('notes/geobc_exposure_short_screen_curtailment.json')
    assert d.identity('notes/geobc_exposure_short_screen_curtailment.json',hashed=True)==review['curtailment_identity']
    assert stop['status']=='intentionally_curtailed' and all(stop['exact_processes_exited'].values())
    assert stop['original_scientific_digest']==parent['scientific_digest']==plan['scientific']['parent_digest']
    assert not stop['original_24000_campaign_completed'] and stop['actual_external_exit_code']==-2
    assert all(d.exact_process_exited(p) for p in stop['exact_processes'].values())
    summary=d.read(d.path(parent['output'])/'summary.json')
    assert d.identity(d.path(parent['output'])/'summary.json',hashed=True)==stop['final_native_summary_identity']
    assert summary['closed'] and not summary['active'] and not summary.get('both24000',False)
    assert summary['raw_GT_reads']==summary['DEV_FINAL_reads']==summary['scaler_fits']==0
    assert all(summary['actual_cache_vs_original_loader_tensor_equality'].values())
    for arm in ('FNO9','PP256'):
        row=summary['arms'][arm]; point=row['endpoints']['12000']
        assert row['restored_Adam_RNG_and_case_cursor'] and row['first_actual_cached_fullN_gradient_check']['passed']
        assert point['TRAIN120']['all_cases_complete'] and point['case_exposure_count']==100
        assert max(point['epoch_seconds'],point['monotonic_seconds'])<=1200
        assert point['checkpoint_identity']==stop['immutable_endpoint_identities'][arm]
    return parent,summary


def run():
    plan=d.read(PLAN); science=plan['scientific']; review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_readout_released']
    assert plan['scientific_digest']==review['scientific_digest']==d.digest(science)
    assert plan['source_pins']==review['source_pins'] and all(d.sha(p)==h for p,h in plan['source_pins'].items())
    parent,summary=parent_qualified(plan,review)
    outer=json.loads(os.environ['GEOBC_SHORT_SCREEN_READOUT_GUARD']); assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'execution.json').exists()
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json',execution)
    started=(time.time(),time.monotonic());lock=None
    result=dict(status='running',closed=False,active=True,execution=execution,candidates={},prediction_inventory={},
        fixed_prediction_denominator=120,fixed_case_denominator=60,committed_finite_predictions=0,
        owned_GT_array_reads=0,raw_GT_reads=0,sealed_DEV_FINAL_reads=0,training_updates=0,scaler_fits=0)
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Independent short-screen3600s readout deadline')
    def persist():
        result.update(elapsed_epoch_seconds=time.time()-started[0],elapsed_monotonic_seconds=time.monotonic()-started[1])
        d.write(output/'status.json',result)
    def check(item):
        from geobc.coverage_comparison_data import stat_matches
        stat_matches(item); assert d.sha(item['path'])==item['sha256'];return d.path(item['path'])
    try:
        import fcntl
        lock=d.path(science['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch
        from geobc import coverage_comparison_data as data
        from geobc.common_grid_solver import CommonGridSolver
        from geobc.available_cad_pilot_eval import _save_array
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.benchmark import FieldScaler
        from geobc.transformers import close_owned_process_group
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device();torch.cuda.set_per_process_memory_fraction(30500000000/torch.cuda.get_device_properties(device).total_memory,device)
        manifest=d.read(check(science['input_manifest_identity']));keys=science['case_keys']
        assert manifest['status']=='completed' and manifest['feasibility_pass'] and not manifest['train_keys']
        assert manifest['case_keys']==keys and set(manifest['records'])==set(keys) and len(keys)==60
        assert manifest['scaler']==parent['scientific']['scaler']
        manifest={**manifest,'train_keys':{'40':[]}};scaler=FieldScaler(**manifest['scaler']);inventory={}
        candidates=science['candidates']
        for candidate in candidates:
            guard();arm=candidate['arm'];step=candidate['step'];name=candidate['name']
            endpoint=summary['arms'][arm]['endpoints'][str(step)]['checkpoint_identity']
            saved=torch.load(check(endpoint),map_location='cpu',weights_only=False)
            assert saved['step']==step and saved['arm']==arm and saved['scientific_digest']==parent['scientific_digest']
            assert saved['case_exposure_counts']=={k:step//120 for k in parent['scientific']['TRAIN_keys']}
            model=CommonGridSolver(arm,2026);model.load_state_dict(saved['model'],strict=True);del saved
            model=model.to('cuda').eval();row=dict(arm=arm,step=step,cases=0,inference_seconds=0.,endpoint=endpoint,counts=model.counts())
            result['candidates'][name]=row;torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for i,key in enumerate(keys):
                    guard();case=data.load_case(manifest,key,device='cuda',for_fno=True)
                    assert 'target' not in case and 'physical' not in case
                    n=len(case['xyz']);assert n==manifest['records'][key]['num_nodes']
                    torch.manual_seed(2026+i);torch.cuda.manual_seed_all(2026+i);torch.cuda.synchronize();began=time.monotonic()
                    context,latent=model.encode_case(case);pred=np.empty((n,4),dtype=np.float32)
                    for a in range(0,n,32768):
                        guard();b=min(a+32768,n);pred[a:b]=model.query(case,a,b,context,latent).cpu().numpy()
                    torch.cuda.synchronize();row['inference_seconds']+=time.monotonic()-began
                    assert np.isfinite(pred).all();physical=np.asarray(scaler.decode(pred),dtype=np.float32)
                    assert physical.shape==(n,4) and np.isfinite(physical).all()
                    directory=output/'predictions'/name/key
                    record=dict(candidate=name,arm=arm,step=step,key=key,status='completed',all_original_nodes=n,seed=2026+i,
                        endpoint=endpoint,encoded=_save_array(d,np,directory/'encoded4.npy',pred),
                        physical=_save_array(d,np,directory/'physical4.npy',physical),signed_stress_unchanged=True)
                    file=directory/'manifest.json';d.write(file,record);inventory[name+':'+key]=d.identity(file,hashed=True)
                    row['cases']+=1;result['committed_finite_predictions']+=1
                    row['peak_allocated_bytes']=torch.cuda.max_memory_allocated();row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    assert max(row['peak_allocated_bytes'],row['peak_reserved_bytes'])<=30500000000
                    persist();del case,context,latent,pred,physical
            check(endpoint);del model;close_owned_process_group();torch.cuda.empty_cache()
        expected={c['name']+':'+key for c in candidates for key in keys}
        assert set(inventory)==expected and len(inventory)==120 and result['owned_GT_array_reads']==0
        for token,item in inventory.items():
            guard();record=d.read(check(item));name,key=token.split(':',1)
            candidate=next(c for c in candidates if c['name']==name)
            assert record['candidate']==name and record['key']==key and record['arm']==candidate['arm'] and record['step']==candidate['step']
            assert record['all_original_nodes']==manifest['records'][key]['num_nodes']
            for field in ('encoded','physical'):
                array=np.load(check(record[field]),mmap_mode='r',allow_pickle=False)
                assert array.shape==(record['all_original_nodes'],4) and np.isfinite(array).all();del array
        barrier=dict(status='completed',finite_predictions=120,expected=120,all_cartesian_entries=True,owned_GT_reads_before_barrier=0,inventory=inventory)
        d.write(output/'prediction_barrier.json',barrier);result['prediction_inventory']=inventory
        truths=d.read(check(science['GT_manifest_identity']))
        assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
        per={c['name']:{key:None for key in keys} for c in candidates}
        for key in keys:
            guard();truth=np.load(check(truths['records'][key]['physical']),mmap_mode='r',allow_pickle=False)
            result['owned_GT_array_reads']+=1;assert truth.shape==(manifest['records'][key]['num_nodes'],4)
            assert truths['records'][key]['all_original_nodes'] and truths['records'][key]['signed_stress_unchanged']
            for candidate in candidates:
                name=candidate['name'];record=d.read(check(inventory[name+':'+key]))
                pred=np.load(check(record['physical']),mmap_mode='r',allow_pickle=False)
                per[name][key]=physical_metrics(np,truth,pred);del pred
            del truth;persist()
        for candidate in candidates:
            name=candidate['name'];row=result['candidates'][name];row['full60']=aggregate(per[name],60)
            row['per_load20']={load:aggregate({k:per[name][k] for k in keys if k.endswith('/'+load)},20) for load in ('ver','hor','tor')}
            assert row['full60']['all_cases_complete']
        scores={name:row['full60']['selection_score'] for name,row in result['candidates'].items()}
        result.update(status='completed',protocol_qualified=True,eligible_best=sorted(n for n,s in scores.items() if s==min(scores.values())),
            final_solver_selected=False,independent_offnode_FEM_accuracy_measured=False,parent_curtailment_binding=review['curtailment_identity'],original_24000_campaign_completed=False)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),protocol_qualified=False,eligible_best=None,automatic_replay=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-started[0],elapsed_monotonic_seconds=time.monotonic()-started[1])
        d.write(output/'summary.json',result);d.write(output/'status.json',result)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
