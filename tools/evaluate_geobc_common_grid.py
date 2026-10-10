"""Matched fixed endpoints on all60 known development cases, never sealed DEV.

The data/model/array work is gated by genuine training termination and one GPU
lock. Both120 finite immutable predictions precede all numerical label reads.
"""
import argparse
import json
import os
import time
from geobc import available_cad_pilot_data as d

PLAN='notes/geobc_common_grid_readout_plan.json'


def run():
    plan=d.read(PLAN);science=plan['scientific'];review=d.read(plan['root_review'])
    assert review['passed'] and review['actual_readout_released']
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(science)
    assert review['source_pins']==plan['source_pins']
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    outer=json.loads(os.environ['GEOBC_COMMON_READOUT_GUARD'])
    assert outer['scientific_digest']==plan['scientific_digest']
    output=d.path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'execution.json').exists(),'One readout attempt; no silent replay'
    execution={'run_process':d.process_snapshot(os.getpid()),'guard':outer,'scientific_digest':plan['scientific_digest']}
    d.write(output/'execution.json',execution)
    result={'status':'running','active':True,'closed':False,'execution':execution,
        'candidates':{},'prediction_inventory':{},'owned_GT_array_reads':0,'raw_GT_reads':0,
        'sealed_DEV_reads':0,'training_updates':0,'scaler_fits':0,'committed_finite_predictions':0}
    epoch=time.time();mono=time.monotonic();lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Original1800-second readout deadline reached')
    def persist():
        result.update(elapsed_epoch_seconds=time.time()-epoch,elapsed_monotonic_seconds=time.monotonic()-mono)
        d.write(output/'status.json',result)
    def check(item,hashed=True):
        from geobc.coverage_comparison_data import stat_matches
        stat_matches(item)
        if hashed:assert d.sha(item['path'])==item['sha256']
        return d.path(item['path'])
    try:
        parent=d.read(science['parent_plan']);folder=d.path(parent['output'])
        summary=d.read(folder/'summary.json');ext=d.read(folder/'external_terminal.json')
        assert summary['status']=='completed' and summary['closed'] and not summary['active']
        assert summary['both4800'] and summary['same_shared_init']
        assert ext['exit_code']==0 and ext['closed'] and not ext['active'] and ext['within_external_budget']
        assert all(d.exact_process_exited(ext[k]) for k in ('guard_process','child_process'))
        assert summary['execution']['scientific_digest']==science['parent_scientific_digest']==parent['scientific_digest']
        assert all(d.sha(p)==h for p,h in parent['source_pins'].items())
        result['parent_terminal_bindings']={'summary':d.identity(folder/'summary.json',hashed=True),
            'external':d.identity(folder/'external_terminal.json',hashed=True),'exact_processes_exited':True}
        for arm in science['arms']:
            row=summary['arms'][arm]
            assert row['status']=='completed' and row['step']==4800 and row['optimizer_steps']==4800
            assert row['TRAIN120']['all_cases_complete'] and row['first_actual_fullN_gradient_check']['passed']
            assert row['case_exposure_counts']=={key:40 for key in parent['scientific']['TRAIN_keys']}
            assert max(row['training_elapsed_epoch_seconds'],row['training_elapsed_monotonic_seconds'])<=18000
            assert max(row['readout_elapsed_epoch_seconds'],row['readout_elapsed_monotonic_seconds'])<=1200
        import fcntl
        lock=d.path(science['GPU_lock']).open('a+')
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch
        from geobc import coverage_comparison_data as data
        from geobc.available_cad_pilot_eval import _save_array
        from geobc.common_grid_solver import CommonGridSolver
        from geobc.transformers import close_owned_process_group
        from geobc.coverage_comparison_train import physical_metrics,aggregate
        from geobc.benchmark import FieldScaler
        guard();torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        device=torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(device).total_memory,device)
        check(science['input_manifest_identity'])
        manifest=d.read(science['input_manifest_identity']['path']);keys=science['case_keys']
        assert manifest['status']=='completed' and manifest['feasibility_pass']
        # Evaluation-only producer truthfully has no TRAIN records. The legacy
        # loader eagerly indexes its train40 default even for validation keys.
        assert not manifest['train_keys']
        manifest={**manifest,'train_keys':{'40':[]}}
        assert manifest['case_keys']==keys and set(manifest['records'])==set(keys) and len(keys)==60
        assert manifest['scaler']==parent['scientific']['scaler']
        scaler=FieldScaler(**manifest['scaler'])
        inventory={}
        for arm in science['arms']:
            guard();endpoint=summary['arms'][arm]['checkpoint_identity'];check(endpoint)
            saved=torch.load(d.path(endpoint['path']),map_location='cpu',weights_only=False)
            assert saved['step']==4800 and saved['arm']==arm and saved['scientific_digest']==parent['scientific_digest']
            assert saved['case_exposure_counts']=={key:40 for key in parent['scientific']['TRAIN_keys']}
            model=CommonGridSolver(arm,2026);model.load_state_dict(saved['model'],strict=True)
            del saved;model=model.to('cuda').eval()
            row={'endpoint':endpoint,'optimizer_restored':False,'inference_seconds':0.,'cases':0,
                 'counts':model.counts(),'TRAIN_costs':{k:v for k,v in summary['arms'][arm].items() if k.startswith('training_')}}
            result['candidates'][arm]=row;torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for index,key in enumerate(keys):
                    guard();case=data.load_case(manifest,key,device='cuda',for_fno=True)
                    assert 'target' not in case and 'physical' not in case
                    n=len(case['xyz']);assert n==manifest['records'][key]['num_nodes']
                    torch.manual_seed(2026+index);torch.cuda.manual_seed_all(2026+index)
                    torch.cuda.synchronize();began=time.monotonic()
                    context,latent=model.encode_case(case);prediction=np.empty((n,4),dtype=np.float32)
                    for start in range(0,n,32768):
                        guard();stop=min(n,start+32768)
                        prediction[start:stop]=model.query(case,start,stop,context,latent).float().cpu().numpy()
                    torch.cuda.synchronize();row['inference_seconds']+=time.monotonic()-began
                    assert np.isfinite(prediction).all()
                    physical=np.asarray(scaler.decode(prediction),dtype=np.float32)
                    assert physical.shape==(n,4) and np.isfinite(physical).all()
                    directory=output/'predictions'/arm/key
                    encoded_item=_save_array(d,np,directory/'encoded4.npy',prediction)
                    physical_item=_save_array(d,np,directory/'physical4.npy',physical)
                    token=arm+':'+key
                    record={'arm':arm,'key':key,'status':'completed','all_original_nodes':n,'seed':2026+index,
                            'endpoint':endpoint,'encoded':encoded_item,'physical':physical_item,'signed_stress_unchanged':True}
                    filename=directory/'manifest.json';d.write(filename,record)
                    inventory[token]=d.identity(filename,hashed=True)
                    result['committed_finite_predictions']+=1;row['cases']+=1
                    row['peak_allocated_bytes']=torch.cuda.max_memory_allocated();row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    assert max(row['peak_allocated_bytes'],row['peak_reserved_bytes'])<=24*2**30
                    persist();del case,context,latent,prediction,physical
            check(endpoint);del model;close_owned_process_group();torch.cuda.empty_cache()
        # Reopen every committed record and inspect the actual finite array
        # shape, not only a passed marker, before any numerical GT cache read.
        expected={arm+':'+key for arm in science['arms'] for key in keys}
        assert set(inventory)==expected and len(inventory)==120 and result['owned_GT_array_reads']==0
        for token,item in inventory.items():
            guard();record=d.read(check(item));arm,key=token.split(':',1)
            assert record['key']==key and record['arm']==arm and record['all_original_nodes']==manifest['records'][key]['num_nodes']
            for field in ('encoded','physical'):
                array=np.load(check(record[field]),mmap_mode='r',allow_pickle=False)
                assert array.shape==(record['all_original_nodes'],4) and np.isfinite(array).all()
                del array
        barrier={'status':'completed','finite_predictions':120,'expected':120,
            'all_cartesian_entries':True,'owned_GT_reads_before_barrier':0,'inventory':inventory,
            'GT_scope':'Already observed development cache; not blind or sealed final test'}
        d.write(output/'prediction_barrier.json',barrier);result['prediction_inventory']=inventory
        check(science['GT_manifest_identity']);truths=d.read(science['GT_manifest_identity']['path'])
        assert truths['status']=='completed' and truths['source_GT_parses']==20 and set(truths['records'])==set(keys)
        per={arm:{key:None for key in keys} for arm in science['arms']}
        for key in keys:
            guard();truth=np.load(check(truths['records'][key]['physical']),mmap_mode='r',allow_pickle=False)
            result['owned_GT_array_reads']+=1
            assert truth.shape==(manifest['records'][key]['num_nodes'],4)
            assert truths['records'][key]['all_original_nodes'] and truths['records'][key]['signed_stress_unchanged']
            for arm in science['arms']:
                record=d.read(check(inventory[arm+':'+key]))
                prediction=np.load(check(record['physical']),mmap_mode='r',allow_pickle=False)
                per[arm][key]=physical_metrics(np,truth,prediction);del prediction
            del truth;persist()
        for arm in science['arms']:
            result['candidates'][arm]['full60']=aggregate(per[arm],60)
            result['candidates'][arm]['per_load20']={load:aggregate({key:per[arm][key] for key in keys if key.endswith('/'+load)},20)
                for load in ('ver','hor','tor')}
            assert result['candidates'][arm]['full60']['all_cases_complete']
        cad_ids=sorted({key.split('/')[0] for key in keys})
        paired={}
        for cad in cad_ids:
            paired[cad]={metric:sum(per['FNO9'][cad+'/'+load][metric]-per['PP256'][cad+'/'+load][metric] for load in ('ver','hor','tor'))/3
                        for metric in ('u_relative_l2','stress_relative_l2','stress_hotspot_relative_l2')}
        scores={arm:result['candidates'][arm]['full60']['selection_score'] for arm in science['arms']}
        result.update(status='completed',protocol_qualified=True,paired_CAD20_FNO_minus_PP=paired,
            eligible_best=[arm for arm,score in scores.items() if score==min(scores.values())],
            final_solver_selected=False,independent_offnode_FEM_accuracy_measured=False)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_replay=False,
                      protocol_qualified=False,eligible_best=None,fixed_prediction_denominator=120,fixed_case_denominator=60)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-epoch,elapsed_monotonic_seconds=time.monotonic()-mono)
        d.write(output/'summary.json',result);d.write(output/'status.json',result)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
