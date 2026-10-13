"""Predict every fixed final case before comparing shared original signed fields."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(args):
    config=json.loads(Path(args.config).read_text());catalog=json.loads(Path(args.source).read_text())
    clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK']);output=Path(args.output)
    state=dict(status='running',active=True,closed=False,phase='imports',predictions={},arms={},committed_predictions=0,
               final_GT_reads=0,final_CSV_parses=0,final_CSV_parse_attempts=0,checkpoint_reads=0,
               input_reads=0,feature_reads=0,GRID_cache_reads=0,permutation_reads=0,training_updates=0,scaler_fits=0,
               peak_allocated_bytes=0,peak_reserved_bytes=0)
    torch=None;cache=None
    def persist():
        if cache is not None:state.update(input_reads=cache.input_reads,feature_reads=cache.feature_reads)
        state['native_seconds']=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Final evaluation deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes']=max(state['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            if max(total-free,state['peak_allocated_bytes'],state['peak_reserved_bytes'])>config['memory_bytes']:raise MemoryError('Final inference memory budget')
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.npy.tmp')
        with temporary.open('wb') as file:np.save(file,value,allow_pickle=False)
        os.replace(temporary,path)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External evaluation deadline')))
    try:
        import gc
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.runtime import GPUQueue
        from geobc.physics_inputs import local_path
        from geobc.learning_data import LearningCache
        from geobc.kinematic_neural_field import KinematicCovariateFNO
        from geobc.fieldwise_neural_field import FieldwiseCovariateFNO
        from geobc.physical_loss import decode
        from geobc.final_fields import read_final_fields
        scale=json.loads(Path('configs/output_scaler.json').read_text());split=json.loads(Path('configs/split.json').read_text())
        keys=[cad+'/'+load for cad in split['final'] for load in config['loads']]
        features=json.loads(local_path(catalog['physical_input_index']).read_text())
        if features['scientific_group']!='FINAL' or features['complete_CADs']!=20 or features['complete_loads']!=60 or set(catalog['records'])!=set(keys) or catalog['scaler']!=scale:
            raise ValueError('Complete fixed final inputs and scaler required')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
            state['phase']='target_free_cache';persist();tick=(time.time(),time.monotonic())
            cache=LearningCache({},guard);cache.load_predictors(catalog['records'],features['features'],'FINAL')
            state['cache_seconds']=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]);persist()
            for seed in config['seeds']:
                for mode in config['modes']:
                    guard();name=mode.lower()+'_seed'+str(seed);state['phase']=name+'/predictions';persist();tick=(time.time(),time.monotonic())
                    checkpoint=torch.load(local_path(catalog['checkpoints'][name]),map_location='cpu',weights_only=False);state['checkpoint_reads']+=1
                    if checkpoint['arm']!=name or checkpoint['cursor']!=config['checkpoint_cursor'] or checkpoint['config']['seeds']!=config['seeds']:
                        raise ValueError('Registered trained checkpoint differs')
                    model=(FieldwiseCovariateFNO(seed) if mode=='FIELDWISE' else KinematicCovariateFNO(mode,seed)).cuda()
                    model.load_state_dict(checkpoint['model']);model.eval();del checkpoint
                    parameters=sum(p.numel() for p in model.parameters())
                    if parameters!=config['parameters'][mode]:raise ValueError('Selected architecture differs')
                    state['predictions'][name]={}
                    with torch.no_grad():
                        for key in keys:
                            guard();case=cache.cases[key];context,latent=model.encode_case(case);chunks=[]
                            for start in range(0,len(case['xyz']),config['decoder_chunk_nodes']):
                                guard();stop=min(start+config['decoder_chunk_nodes'],len(case['xyz']))
                                chunks.append(decode(model.query(case,start,stop,context,latent),scale).cpu().numpy())
                            value=np.concatenate(chunks)
                            if value.shape!=(len(case['xyz']),4) or value.dtype!=np.float64 or not np.isfinite(value).all():raise FloatingPointError('Incomplete final physical predictions')
                            path=output/name/'predictions'/(key.replace('/','_')+'.npy');save(path,value)
                            state['predictions'][name][key]=str(path.resolve());state['committed_predictions']+=1
                            del context,latent,chunks,value
                        state['arms'][name]=dict(mode=mode,seed=seed,parameters=parameters,seconds=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]),predictions=60)
                    persist();del model;gc.collect();torch.cuda.empty_cache()
            barrier=state['committed_predictions']==config['prediction_denominator'] and len(state['predictions'])==6 and all(set(paths)==set(keys) and all(Path(p).is_file() for p in paths.values()) for paths in state['predictions'].values())
            if not barrier:raise ValueError('All selected model predictions must precede final readout')
            state['prediction_barrier_completed']=True;state['phase']='shared_final_truth';persist()
            state['metrics']={name:[] for name in state['predictions']}
            for cad in split['final']:
                guard();truth_record=catalog['truth_sources'][cad]
                with np.load(local_path(truth_record['GRID_cache']),allow_pickle=False) as source:ids=source['grid_node_ids']
                state['GRID_cache_reads']+=1
                permutation=np.load(local_path(truth_record['permutation']),allow_pickle=False);state['permutation_reads']+=1
                state['final_CSV_parse_attempts']+=1;persist()
                truths=read_final_fields(local_path(truth_record['CSV']),permutation,ids,barrier)
                state['final_CSV_parses']+=1;state['final_GT_reads']+=3
                for load,truth in truths.items():
                    key=cad+'/'+load;unorm=np.linalg.norm(truth[:,:3]);snorm=np.linalg.norm(truth[:,3])
                    if unorm==0 or snorm==0:raise ValueError('Undefined original final field norm')
                    for name,paths in state['predictions'].items():
                        prediction=np.load(paths[key],allow_pickle=False)
                        if prediction.shape!=truth.shape or not np.isfinite(prediction).all():raise ValueError('Saved final prediction differs')
                        U=float(np.linalg.norm(prediction[:,:3]-truth[:,:3])/unorm)*100;S=float(np.linalg.norm(prediction[:,3]-truth[:,3])/snorm)*100
                        state['metrics'][name].append(dict(key=key,U=U,S=S,primary=.5*(U+S)))
                persist()
            state['aggregate']={name:{field:sum(r[field] for r in rows)/60 for field in ['U','S','primary']} for name,rows in state['metrics'].items()}
            differences=[{field:state['aggregate']['zero_seed'+str(seed)][field]-state['aggregate']['fieldwise_seed'+str(seed)][field] for field in ['U','S','primary']} for seed in config['seeds']]
            means={f:sum(r[f] for r in differences)/3 for f in ['U','S','primary']};wins=sum(r['primary']>0 for r in differences);gate=config['practical_benefit']
            state.update(paired_improvement_pp=means,paired_primary_seed_wins=wins,practical_benefit_passed=means['primary']>=gate['mean_primary_reduction_pp'] and means['S']>=gate['mean_stress_reduction_pp'] and wins>=gate['primary_seed_wins'],status='completed',phase='terminal')
    except Exception as error:
        state.update(status='failed',phase='technical_failure',error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(active=False,closed=True);persist();(output/'summary.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','source','output','lock']:p.add_argument('--'+name,required=True)
    run(p.parse_args())
