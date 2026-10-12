"""One released81-prediction/54-displacement-property run; labels stay closed."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import os
import json
from pathlib import Path
import signal


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.load_amplitude_contract_v95 import approved_contract
    p,review,m=approved_contract(plan_path);s=p['scientific'];o=d.path(p['output'])
    outer=json.loads(os.environ['GEOBC_LOAD_AMPLITUDE_V95_GUARD'])
    assert outer['scientific_digest']==p['scientific_digest'] and outer['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid'])==outer['guard_process']
    execution=dict(run_process=d.process_snapshot(os.getpid()),guard=outer,boot_id=outer['boot_id'],scientific_digest=p['scientific_digest'])
    assert not (o/'execution.json').exists();d.write(o/'execution.json',execution)
    a=dict(status='running',active=True,closed=False,phase='deck_physics_gate',execution=execution,
        source_deck_hash_reads=0,source_deck_parse_attempts=0,source_deck_parses=0,source_deck_header_reads=0,
        input_array_attempts=0,input_array_reads=0,GT_reads=0,CSV_H5_reads=0,physical_encoded_target_reads=0,
        training_updates=0,scaler_fits=0,checkpoint_hash_reads=0,checkpoint_loads=0,
        committed_finite_predictions=0,fixed_prediction_denominator=81,fixed_property_denominator=54,
        property_rows=0,candidates={},physics_gate={},input_witness={},protocol_qualified=False,
        final_solver_selected=False,automatic_replay=False,peak_allocated_bytes=0,peak_reserved_bytes=0,peak_total_device_used_bytes_sampled=0)
    torch=None;lock=None
    def guard():
        if min(outer['deadline_epoch']-time.time(),outer['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Frozen900 whole dual-clock deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes']=max(a['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes']=max(a['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info();a['peak_total_device_used_bytes_sampled']=max(a['peak_total_device_used_bytes_sampled'],total-free)
            if max(a[k]for k in ('peak_allocated_bytes','peak_reserved_bytes','peak_total_device_used_bytes_sampled'))>s['budget']['memory_bytes']:raise MemoryError('30.5GB cap')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'status.json',a)
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        import fcntl
        lock=d.path(s['budget']['GPU_lock']).open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        import numpy as np
        import torch as tm
        torch=tm
        from geobc import elastic_pilot_data as parser,coverage_comparison_data as original
        from geobc.prospective_dataset import resolve_case
        from geobc.solver_selection_data import _splat
        from geobc.macro_training import encode_case,query_case
        from geobc.fno_stress_specialist_v54 import tree_digest
        from geobc.macro_readout_protocol import verify_barrier
        from geobc.benchmark import FieldScaler
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory;assert s['budget']['memory_bytes']<total
        torch.cuda.set_per_process_memory_fraction(s['budget']['memory_bytes']/total,0);torch.cuda.reset_peak_memory_stats()
        force_vectors={};material_vectors={}
        for cad,item in s['raw_deck_identities'].items():
            guard();assert d.identity(item['path'],hashed=True)==item;a['source_deck_hash_reads']+=1;a['source_deck_parse_attempts']+=1;persist()
            deck=parser._parse_deck(d.path(item['path']),s['deck_reader_policy'],time.monotonic(),900);a['source_deck_parses']+=1
            gen=parser._cards(d.path(item['path']),s['deck_reader_policy']['reader_scope'],time.monotonic(),900)
            try:controls=next(gen)[3]
            finally:gen.close()
            a['source_deck_header_reads']+=1
            mids={deck['psolid'][e['PID']]['MID']for e in deck['ctetra'].values()}
            if len(mids)!=1 or len(deck['materials'])!=1:raise parser.UnsupportedInput('Single explicit MAT1 required')
            material=deck['materials'][next(iter(mids))]
            if material['E']is None or material['E']<=0 or material['nu']is None or not -1<material['nu']<.5:raise parser.UnsupportedInput('Linear elastic MAT1 E/nu unsupported')
            rho=parser._number(material['remaining_fields'][0])if material['remaining_fields']else None
            if rho is None or rho<=0:raise parser.UnsupportedInput('Explicit MAT1 density required for input mapping')
            material_vectors[cad]=[material['E']/113800,material['nu'],rho/4.47e-9]
            rows={};constraints=[]
            for load in s['loads']:
                selected=original._deck_case(deck,controls,load)
                if not selected['active_constraints'] or any(x[2]!=0. for x in selected['active_constraints']):raise parser.UnsupportedInput('Nonzero/missing prescribed support displacement')
                if any(x['kind']not in ('FORCE','MOMENT')for x in selected['active_actions']):raise parser.UnsupportedInput('Unsupported load kind')
                f=np.zeros(3);mom=np.zeros(3)
                for action in selected['active_actions']:(f if action['kind']=='FORCE'else mom)[:]+=np.asarray(action['vector'])
                v=np.concatenate((f/42300,mom/565000));assert np.isfinite(v).all() and np.linalg.norm(v)>0
                force_vectors[cad+'/'+load]=v
                cons=sorted((int(x[0]),''.join(sorted(x[1])),float(x[2]))for x in selected['active_constraints']);constraints.append(cons)
                rows[load]=dict(selected_controls=selected['selected_controls'],force_N=f.tolist(),moment_Nmm=mom.tolist(),constraints=cons)
            a['physics_gate'][cad]=dict(explicit_material=material,zero_constraints=True,STATICS_subcases=rows,same_constraints_across3=all(x==constraints[0]for x in constraints),ignored_original_controls=deck['ignored_controls'],exact_original_solver_fidelity_claimed=False)
            persist();del deck;guard()
        a['phase']='target_free_input_cache';persist();arrays={};cases={};array_specs={}
        for key in s['selected_TRAIN_input_keys']:
            record,items=resolve_case(m,key,False,100,True);assert set(items)=={'xyz','features','cloud','mask','grid','query_indices','query_weights'}
            case={'key':key,'scientific_group':'TRAIN'}
            for name,item in items.items():
                guard();file=item['path']
                if file not in arrays:
                    d.stat_matches(item);a['input_array_attempts']+=1;value=np.load(d.path(file),allow_pickle=False);a['input_array_reads']+=1
                    n=record['num_nodes'];shapes={'xyz':(n,3),'features':(n,25),'cloud':(1024,3),'mask':(1024,),'grid':(29,64,64,64),'query_indices':(n,8),'query_weights':(n,8)}
                    assert value.shape==shapes[name] and str(value.dtype)in (('bool',)if name=='mask'else('int32','int64')if name=='query_indices'else('float32',))
                    if value.dtype.kind=='f':assert np.isfinite(value).all()
                    if name=='query_indices':assert value.min()>=0 and value.max()<64**3
                    arrays[file]=torch.as_tensor(value,device='cuda');array_specs[file]=dict(identity=item,name=name)
                else:assert array_specs[file]['identity']==item and array_specs[file]['name']==name
                case[name]=arrays[file]
            case['cloud']=case['cloud'][None];case['mask']=case['mask'][None];case['grid']=case['grid'][None];cases[key]=case;persist()
        original_digests={file:tree_digest(value)for file,value in arrays.items()}
        assert a['input_array_reads']==a['input_array_attempts']==len(arrays)
        for key,base in cases.items():
            guard();v=torch.as_tensor(force_vectors[key],device='cuda',dtype=torch.float32)
            if not torch.allclose(base['features'][:,7:13],v[None].expand(len(base['xyz']),-1),atol=s['force_absolute_tolerance'],rtol=0):raise parser.UnsupportedInput('Raw deck and native force/moment features disagree')
            mat=torch.as_tensor(material_vectors[key.split('/')[0]],device='cuda',dtype=torch.float32)
            if not torch.allclose(base['features'][:,13:16],mat[None].expand(len(base['xyz']),-1),atol=s['force_absolute_tolerance'],rtol=0):raise parser.UnsupportedInput('Raw MAT1 and native material features disagree')
            comparisons={}
            for alpha in s['load_amplitudes']:
                features=base['features'].clone();features[:,7:13]*=alpha
                grid=base['grid'].clone();grid[:,11:17]*=alpha
                raster,active=_splat(features,base['query_indices'],base['query_weights'],64)
                error=float((raster.T.reshape(25,64,64,64)-grid[0,4:]).abs().max())
                if error>s['raster_absolute_tolerance']or not torch.equal(active.reshape(64,64,64).float(),grid[0,0]):raise parser.UnsupportedInput('Scaled raster/occupancy input witness mismatch')
                assert torch.equal(features[:,:7],base['features'][:,:7]) and torch.equal(features[:,13:],base['features'][:,13:])
                assert torch.equal(grid[:,:11],base['grid'][:,:11]) and torch.equal(grid[:,17:],base['grid'][:,17:])
                if alpha==1.:assert torch.equal(features,base['features'])and torch.equal(grid,base['grid'])
                comparisons[str(alpha)]=dict(raster_all25_max_abs_error=error,native_force_matches_deck=True,unchanged_channels_verified=True,alpha1_byte_exact=alpha==1.)
                del features,grid,raster,active;guard()
            a['input_witness'][key]=comparisons;persist()
        a['phase']='prediction_commit';persist();inventory={};specs=[];virtual_keys=[key+'@a'+str(alpha)for key in cases for alpha in s['load_amplitudes']]
        counts={key+'@a'+str(alpha):len(cases[key]['xyz'])for key in cases for alpha in s['load_amplitudes']};scaler=FieldScaler(**s['scaler'])
        for c in s['frozen_candidates']:
            guard();name=c['name'];family=c['family'];cp=c['checkpoint'];assert d.identity(cp['path'],hashed=True)==cp;a['checkpoint_hash_reads']+=1
            saved=torch.load(d.path(cp['path']),map_location='cpu',weights_only=False);a['checkpoint_loads']+=1;tp=d.read(c['training_plan_identity']['path'])
            assert saved['family']==family and saved['step']==saved['cursor']==6000 and saved['scientific_digest']==c['training_SCI']
            assert saved['case_exposure_counts']=={k:20 for k in tp['scientific']['TRAIN_keys']} and saved['full_order_digest']==c['full_order_digest']
            assert saved['scaler']==s['scaler'] and saved['dataset_identity']==s['dataset_identity'] and set(saved['rng'])=={'python','numpy','torch','cuda'}
            assert saved['optimizer']['state']and all(int(x['step'])==6000 for x in saved['optimizer']['state'].values())
            checkpoint_boot=saved['native_origin_clocks']['boot_id']if family=='FNO9'else saved['execution']['boot_id'];assert checkpoint_boot==c['training_boot_id']
            if family=='FNO9':from geobc.macro_training import build_solver
            elif family=='FNO_GLOBAL_HASH':from geobc.operator_hash_metadata_v6 import build_solver
            else:from geobc.transolver_bf16_hash_v65 import build_solver
            model=build_solver(family,seed=2026);model.load_state_dict(saved['model'],strict=True)
            state=model.state_dict();assert set(state)==set(saved['model'])and all(state[k].dtype==saved['model'][k].dtype and state[k].shape==saved['model'][k].shape and torch.equal(state[k].cpu(),saved['model'][k])for k in state)
            del saved,state;model.eval();digest=tree_digest(model.state_dict());spec=dict(name=name,family=family,step=6000,checkpoint=cp);specs.append(spec)
            row=dict(strict_checkpoint_verified=True,parameter_count=sum(x.numel()for x in model.parameters()),predictions=0,inference_epoch_seconds=0.,inference_monotonic_seconds=0.);a['candidates'][name]=row
            with torch.no_grad():
                for index,(key,base)in enumerate(cases.items()):
                    for alpha in s['load_amplitudes']:
                        guard();case=dict(base);case['features']=base['features'].clone();case['features'][:,7:13]*=alpha;case['grid']=base['grid'].clone();case['grid'][:,11:17]*=alpha
                        torch.manual_seed(2026+index);torch.cuda.manual_seed_all(2026+index);torch.cuda.synchronize();began=(time.time(),time.monotonic())
                        context,latent=encode_case(model,case);n=len(case['xyz']);pred=np.empty((n,4),np.float32)
                        for start in range(0,n,32768):guard();pred[start:min(start+32768,n)]=query_case(model,case,start,min(start+32768,n),context,latent).cpu().numpy()
                        torch.cuda.synchronize();row['inference_epoch_seconds']+=time.time()-began[0];row['inference_monotonic_seconds']+=time.monotonic()-began[1]
                        physical=np.asarray(scaler.decode(pred),np.float32);assert np.isfinite(pred).all()and np.isfinite(physical).all()
                        virtual=key+'@a'+str(alpha);folder=o/'predictions'/name/virtual
                        record=dict(candidate=name,key=virtual,original_key=key,load_amplitude=alpha,family=family,step=6000,checkpoint=cp,all_original_nodes=n,status='completed',signed_stress_unchanged=True,
                            encoded=original._save(np,folder/'encoded4.npy',pred),physical=original._save(np,folder/'physical4.npy',physical))
                        file=folder/'manifest.json';d.write(file,record);inventory[name+':'+virtual]=d.identity(file,hashed=True);row['predictions']+=1;a['committed_finite_predictions']+=1;persist()
                        del case,context,latent,pred,physical;guard()
            assert tree_digest(model.state_dict())==digest;row['model_bytes_unchanged']=True;d.stat_matches(cp);del model;torch.cuda.empty_cache();guard()
        def record_reader(item):
            guard();assert d.identity(item['path'],hashed=True)==item;return d.read(item['path'])
        def array_reader(item):
            guard();assert d.identity(item['path'],hashed=True)=={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};return np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False)
        barrier=verify_barrier(np,specs,virtual_keys,counts,inventory,record_reader,array_reader,guard);assert barrier['finite_predictions']==81
        d.write(o/'prediction_barrier.json',barrier);a.update(prediction_barrier_complete=True,prediction_inventory=inventory,phase='U_property_readout');persist()
        for c in s['frozen_candidates']:
            rows={};name=c['name']
            for key in cases:
                base=array_reader(record_reader(inventory[name+':'+key+'@a1.0'])['physical'])[:,:3].astype(np.float64)
                norm=float(np.square(base).sum())
                for alpha in [.5,1.5]:
                    value=array_reader(record_reader(inventory[name+':'+key+'@a'+str(alpha)])['physical'])[:,:3].astype(np.float64)
                    error=value-alpha*base;sse=float(np.square(error).sum());defect=(sse/(alpha*alpha*norm))**.5 if norm>0 else None
                    rows[key+'@a'+str(alpha)]=dict(key=key,alpha=alpha,nodes=len(base),relative_U_defect=defect,baseline_U_energy_mm2=norm,RMS_vector_defect_mm=(sse/len(base))**.5,max_vector_defect_mm=float(np.linalg.norm(error,axis=1).max()),reference='Own predicted physicalU(alpha1); not GT',passed=(defect is not None and defect<=.01));a['property_rows']+=1
            values=[x['relative_U_defect']for x in rows.values()];complete=len(values)==18 and all(x is not None and np.isfinite(x)for x in values)
            a['candidates'][name]['properties']=dict(fixed_denominator=18,all_cases_complete=complete,per_case=rows,mean_relative_U_defect=sum(values)/18 if complete else None,worst_relative_U_defect=max(values)if complete else None,all_case_1percent_pass=complete and all(x['passed']for x in rows.values()))
        assert a['committed_finite_predictions']==81 and a['property_rows']==54 and a['checkpoint_hash_reads']==a['checkpoint_loads']==3
        assert original_digests=={file:tree_digest(value)for file,value in arrays.items()}
        a.update(status='completed',phase='terminal',protocol_qualified=True,input_tensor_bytes_unchanged=True,scientific_outcome='property_diagnostic_complete',GT_accuracy_measured=False);guard()
    except Exception as error:
        # A scoped source/input failure is not a model accuracy negative.
        if type(error).__name__=='UnsupportedInput':
            a.update(status='completed',phase='terminal_input_negative',scientific_outcome='input_negative',input_negative=True,protocol_qualified=False,error=str(error),eligible_best=None)
        else:
            a.update(status='failed',phase='terminal_failure',error_type=type(error).__name__,error=str(error));raise
    finally:
        a.update(active=False,closed=True,elapsed_epoch_seconds=time.time()-ENTRY_EPOCH,elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC);d.write(o/'summary.json',a);d.write(o/'status.json',a)
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--plan');args=parser.parse_args()
    if args.start:
        if not args.plan:parser.error('--start requires frozen --plan')
        run(args.plan)
    else:print('{"status":"source_only","GT":0,"GPU":0}')
