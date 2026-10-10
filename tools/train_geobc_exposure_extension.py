"""Continue both completed matched arms to fixed24000 using immutable TRAIN-only GPU cache."""
import argparse
import json
import os
import random
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding

PLAN = 'notes/geobc_exposure_extension_plan.json'


def schedule(keys):
    rng = random.Random(2026); order = []
    for _ in range(200):
        rows = list(keys); rng.shuffle(rows); order.extend(rows)
    return order


def run():
    plan = d.read(PLAN); science = plan['scientific']; review = d.read(plan['root_review'])
    assert review['passed'] and review['actual_execution_released']
    assert d.digest(science) == plan['scientific_digest'] == review['scientific_digest']
    assert review['source_pins'] == plan['source_pins']
    assert all(d.sha(p) == h for p, h in plan['source_pins'].items())
    parent = terminal_binding(review['parent_binding'])
    assert parent['both4800'] and parent['execution']['scientific_digest'] == science['parent_digest']
    outer = json.loads(os.environ['GEOBC_EXPOSURE_GUARD'])
    assert outer['scientific_digest'] == plan['scientific_digest']
    output = d.path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    assert not (output/'execution.json').exists(), 'One attempt; no automatic recovery'
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, scientific_digest=plan['scientific_digest'])
    d.write(output/'execution.json', execution)
    entry = (time.time(), time.monotonic()); lock = None; deadline = None
    result = dict(status='running', active=True, closed=False, execution=execution, arms={},
        raw_GT_reads=0, DEV_FINAL_reads=0, scaler_fits=0, unique_cached_TRAIN_array_reads=0,
        cached_TRAIN_physical_reads=0, own_parent_checkpoint_reads=0, current_arm=None)

    def guard():
        left = min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic())
        if deadline: left = min(left, deadline[0]-time.time(), deadline[1]-time.monotonic())
        if left <= 0: raise TimeoutError('Frozen exposure-extension dual deadline')
    def persist():
        result.update(elapsed_epoch_seconds=time.time()-entry[0], elapsed_monotonic_seconds=time.monotonic()-entry[1])
        d.write(output/'status.json', result)
    try:
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch
        from geobc import coverage_comparison_data as data
        from geobc.coverage_comparison_train import physical_metrics, aggregate
        from geobc.common_grid_solver import CommonGridSolver, full_case_backward, parameter_gradients
        from geobc.benchmark import FieldScaler
        from geobc.transformers import close_owned_process_group
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(device).total_memory, device)
        manifest = data.read_bootstrap_TRAIN40(science['original_TRAIN_plan'])
        keys = manifest['train_keys']['40']; assert keys == science['TRAIN_keys'] and manifest['scaler'] == science['scaler']
        order = schedule(keys); assert len(order) == 24000 and d.digest(order[:4800]) == science['parent_order_digest']
        assert d.digest(order) == science['full_order_digest']
        scaler = FieldScaler(**science['scaler']); cached = {}; arrays = {}; identities = {}
        cache_entry = (time.time(), time.monotonic())
        deadline = (cache_entry[0]+900, cache_entry[1]+900)
        for key in keys:
            guard(); record = manifest['records'][key]; case = {'key':key}
            for name in ('xyz','features','cloud','mask','grid','query_indices','query_weights','target'):
                item = record['file_identities'][name+'_path']; data.stat_matches(item); path = item['path']
                if path not in arrays:
                    value = np.load(data.path(path), allow_pickle=False)
                    assert value.dtype.kind in ('f','i','u','b') and (value.dtype.kind != 'f' or np.isfinite(value).all())
                    if 'shape' in item: assert list(value.shape) == item['shape'] and str(value.dtype) == item['dtype']
                    arrays[path] = torch.as_tensor(np.array(value, copy=True), device='cuda')
                    identities[path] = dict(identity=item, shape=list(value.shape), dtype=str(value.dtype),
                                            GPU_bytes=arrays[path].numel()*arrays[path].element_size())
                    result['unique_cached_TRAIN_array_reads'] += 1
                    del value
                case[name] = arrays[path]
            case['cloud'] = case['cloud'].reshape(1,1024,3); case['mask'] = case['mask'].reshape(1,1024).bool()
            case['grid'] = case['grid'].reshape(1,29,64,64,64)
            assert case['xyz'].shape[1] == 3 and case['features'].shape == (len(case['xyz']),25)
            assert case['target'].shape == (len(case['xyz']),4)
            assert case['query_indices'].shape == case['query_weights'].shape == (len(case['xyz']),8)
            cached[key] = case
            if len(cached)%10 == 0: result['cache_loaded_cases'] = len(cached); persist()
        # Actual one-case equivalence to the original loader; cache never mutates inputs.
        reference = data.load_case(manifest, order[4800], device='cuda', for_fno=True)
        item = manifest['records'][order[4800]]['file_identities']['target_path']; data.stat_matches(item)
        reference['target'] = torch.as_tensor(np.array(np.load(data.path(item['path']), allow_pickle=False),copy=True),device='cuda')
        equal = {name:torch.equal(reference[name],cached[order[4800]][name]) for name in cached[order[4800]] if name != 'key'}
        assert all(equal.values()); result['actual_cache_vs_original_loader_tensor_equality'] = equal
        result['cache_reference_extra_array_reads'] = 8
        del reference; torch.cuda.empty_cache()
        result['GPU_cache'] = dict(unique_files=len(arrays), bytes=sum(v['GPU_bytes'] for v in identities.values()),
            epoch_seconds=time.time()-cache_entry[0], monotonic_seconds=time.monotonic()-cache_entry[1],
            duplicate_files_shared=True, inputs_immutable=True)
        d.write(output/'cache_index.json', identities); deadline = None; persist()
        def rng(): return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
        def restore(saved):
            random.setstate(saved['python']); np.random.set_state(saved['numpy']); torch.set_rng_state(saved['torch']); torch.cuda.set_rng_state_all(saved['cuda'])
        for arm in science['arms']:
            guard(); folder = output/arm; folder.mkdir()
            pin = science['checkpoints'][arm]; assert d.identity(pin['path'], hashed=True) == pin
            payload = torch.load(d.path(pin['path']), map_location='cpu', weights_only=False)
            result['own_parent_checkpoint_reads'] += 1
            assert payload['scientific_digest'] == science['parent_digest'] and payload['arm'] == arm and payload['step'] == 4800
            assert set(payload['case_exposure_counts']) == set(keys) and set(payload['case_exposure_counts'].values()) == {40}
            model = CommonGridSolver(arm,2026).to('cuda').train(); model.load_state_dict(payload['model'],strict=True)
            optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,fused=True)
            optimizer.load_state_dict(payload['optimizer'])
            assert len(optimizer.state) == len(payload['optimizer']['state']) > 0
            assert all(group['lr'] == .001 and group['weight_decay'] == 0 and group['fused'] for group in optimizer.param_groups)
            assert all(int(state['step']) == 4800 for state in optimizer.state.values())
            restore(payload['rng']); row = dict(status='training', step=4800, additional_updates=0,
                counts=model.counts(), parent_checkpoint=pin, case_exposure_counts=payload['case_exposure_counts'],
                restored_Adam_RNG_and_case_cursor=True, endpoints={}, training_entry_epoch=time.time(),
                training_entry_monotonic=time.monotonic(), peak_allocated_bytes=0,peak_reserved_bytes=0)
            del payload
            result['arms'][arm] = row; result['current_arm'] = arm
            deadline = (row['training_entry_epoch']+45000,row['training_entry_monotonic']+45000)
            torch.cuda.reset_peak_memory_stats(); persist()
            def save(step,name):
                guard(); temporary = folder/(name+'.tmp')
                torch.save(dict(step=step,arm=arm,scientific_digest=plan['scientific_digest'],model=model.state_dict(),
                    optimizer=optimizer.state_dict(),rng=rng(),case_exposure_counts=dict(row['case_exposure_counts']),cost_status=dict(row)),temporary)
                os.replace(temporary,folder/name)
            def readout(step):
                saved = rng(); started = (time.time(),time.monotonic()); model.eval(); per={key:None for key in keys}
                def readout_guard():
                    guard()
                    if min(started[0]+1200-time.time(),started[1]+1200-time.monotonic()) <= 0:
                        raise TimeoutError('TRAIN120 fixed readout budget')
                with torch.no_grad():
                    for i,key in enumerate(keys):
                        readout_guard()
                        torch.manual_seed(2026+i); torch.cuda.manual_seed_all(2026+i); case = cached[key]
                        context,latent = model.encode_case(case); prediction=np.empty((len(case['xyz']),4),dtype=np.float32)
                        for a in range(0,len(prediction),32768):
                            readout_guard(); b=min(a+32768,len(prediction)); prediction[a:b]=model.query(case,a,b,context,latent).cpu().numpy()
                        item=manifest['records'][key]['file_identities']['physical_path']; data.stat_matches(item)
                        truth=np.load(data.path(item['path']),mmap_mode='r',allow_pickle=False); result['cached_TRAIN_physical_reads']+=1
                        per[key]=physical_metrics(np,truth,scaler.decode(prediction)); del context,latent,prediction,truth
                readout_guard(); metric=aggregate(per,120); assert metric['all_cases_complete']
                row['endpoints'][str(step)]=dict(TRAIN120=metric,epoch_seconds=time.time()-started[0],monotonic_seconds=time.monotonic()-started[1],
                    case_exposure_count=step//120,training_epoch_seconds_before_readout=started[0]-row['training_entry_epoch'],
                    training_monotonic_seconds_before_readout=started[1]-row['training_entry_monotonic'])
                restore(saved); model.train(); persist()
            with (folder/'loss.jsonl').open('x',encoding='utf-8') as log:
                for step in range(4801,24001):
                    guard(); key=order[step-1]; optimizer.zero_grad(set_to_none=True)
                    loss,leaves=full_case_backward(model,cached[key],guard)
                    if step == 4801:
                        gradients=parameter_gradients(model); unused=['backbone.embedding.weight','backbone.embedding.bias'] if arm=='PP256' else []
                        passed=gradients['operator']['missing']==unused and not gradients['encoder']['missing'] and not gradients['head']['missing']
                        passed &= all(g['finite'] and g['L2']>0 for g in gradients.values()) and all(v>0 for v in leaves.values())
                        row['first_actual_cached_fullN_gradient_check']=dict(passed=passed,gradients=gradients,leaf_gradients=leaves,
                            allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
                        assert passed and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()) <= 24*2**30
                    for parameter in model.parameters():
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()): raise FloatingPointError('Nonfinite gradient')
                    optimizer.step(); torch.cuda.synchronize(); row['step']=step; row['additional_updates']+=1; row['case_exposure_counts'][key]+=1
                    row['training_epoch_seconds']=time.time()-row['training_entry_epoch']; row['training_monotonic_seconds']=time.monotonic()-row['training_entry_monotonic']
                    row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(); row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    assert max(row['peak_allocated_bytes'],row['peak_reserved_bytes']) <= 24*2**30
                    log.write(json.dumps(dict(step=step,key=key,encoded_MSE=loss,training_epoch_seconds=row['training_epoch_seconds']))+'\n'); log.flush()
                    if step%120 == 0: save(step,'last.pt')
                    if step%30 == 0: persist()
                    if step in (12000,24000):
                        assert set(row['case_exposure_counts'].values()) == {step//120}
                        save(step,'step'+str(step)+'.pt')
                        if step == 24000:
                            row['training_epoch_seconds']=time.time()-row['training_entry_epoch']; row['training_monotonic_seconds']=time.monotonic()-row['training_entry_monotonic']; deadline=None
                        readout(step)
                        row['endpoints'][str(step)]['checkpoint_identity']=d.identity(folder/('step'+str(step)+'.pt'),hashed=True)
            row['status']='completed'; persist(); del model,optimizer; close_owned_process_group(); torch.cuda.empty_cache()
        result.update(status='completed',both24000=True,current_arm=None,
                      requires_separate_development_readout=True,training_cost_includes_midpoint_readout=True)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),automatic_replay=False)
        raise
    finally:
        result.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-entry[0],elapsed_monotonic_seconds=time.monotonic()-entry[1])
        d.write(output/'summary.json',result); d.write(output/'status.json',result)
        if lock is not None:lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true')
    if parser.parse_args().start:run()
