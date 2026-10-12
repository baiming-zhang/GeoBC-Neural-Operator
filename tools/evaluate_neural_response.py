"""Qualify the trained continuous U/S interface on three complete CAD geometries."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


class ResponseNegative(ValueError):
    pass


def run(args):
    config=json.loads(Path(args.config).read_text()); catalog=json.loads(Path(args.source).read_text())
    clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK']); output=Path(args.output)
    state=dict(status='running',active=True,closed=False,phase='imports',case_records={},prediction_inventory={},
               predicted_rows=0,outside_rejections=0,reference_reads=0,GT_reads=0,training_updates=0,
               checkpoint_reads=0,source_reads=0,feature_reads=0,peak_allocated_bytes=0,peak_reserved_bytes=0)
    torch=None
    def persist():
        state['native_seconds']=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');os.replace(temporary,output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=0:
            raise TimeoutError('Continuous neural response deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes']=max(state['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            if max(total-free,state['peak_allocated_bytes'],state['peak_reserved_bytes'])>config['memory_bytes']:
                raise MemoryError('Continuous response memory budget')
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.npy.tmp')
        with temporary.open('wb') as file: np.save(file,value.detach().cpu().numpy(),allow_pickle=False)
        os.replace(temporary,path)
    def compare(a,b):
        norm=float(torch.linalg.vector_norm(b)); error=float(torch.linalg.vector_norm(a-b))
        good=error<=config['same_model_relative_tolerance']*norm if norm>0 else float((a-b).abs().max())<=config['zero_absolute_tolerance']
        if not good: raise ResponseNegative('Continuous same-model comparison failed')
        return error/norm if norm>0 else None
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External deadline')))
    try:
        import gc
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.runtime import GPUQueue
        from geobc.physics_inputs import local_path
        from geobc.learning_data import LearningCache
        from geobc.kinematic_neural_field import KinematicCovariateFNO
        from geobc.neural_response import LocatedNeuralResponse,LocatedFieldwiseNeuralResponse
        from geobc.query import shape
        scale=json.loads(Path('configs/output_scaler.json').read_text())
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        references=[]
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            tick=(time.time(),time.monotonic())
            routes=config.get('field_routes',{'joint':'ACTUAL'})
            models={}
            for field,mode in routes.items():
                path=catalog['checkpoints'][field] if 'field_routes' in config else catalog['checkpoint']
                checkpoint=torch.load(local_path(path),map_location='cpu',weights_only=False);state['checkpoint_reads']+=1
                if checkpoint['arm']!=mode.lower()+'_seed2026' or checkpoint['cursor']!=6000:
                    raise ValueError('Expected first-seed trained endpoint')
                model=KinematicCovariateFNO(mode,2026).cuda();model.load_state_dict(checkpoint['model']);model.eval()
                models[field]=model;del checkpoint
            state['model_parameters']=sum(sum(p.numel() for p in model.parameters()) for model in models.values())
            state['field_routes']=routes
            state['model_load_seconds']=max(time.time()-tick[0],time.monotonic()-tick[1])
            for cad in config['CADs']:
                guard();state['phase']=cad+'/load';persist();tick=(time.time(),time.monotonic())
                record=catalog['cases'][cad]; n,c=record['expected_geometry']['nodes'],record['expected_geometry']['cells']
                with np.load(local_path(record['geometry']),allow_pickle=False) as archive:
                    vertices=archive['vertices_mm'];cells=archive['cells10']
                    if vertices.dtype!=np.float64 or vertices.shape!=(n,3) or cells.dtype!=np.int64 or cells.shape!=(c,10) or not np.isfinite(vertices).all():
                        raise ValueError('Geometry schema differs')
                    vertices=torch.as_tensor(vertices,device='cuda');cells=torch.as_tensor(cells,device='cuda')
                state['source_reads']+=1
                cache=LearningCache({},guard)
                cache.load_predictors(record['predictors'],{cad:record['physical_inputs']},'DEV',keep_physical_covariates=True)
                state['feature_reads']+=cache.feature_reads
                cases={load:cache.cases[cad+'/'+load] for load in config['loads']}
                B=torch.tensor([record['reference_load6'][load] for load in config['loads']],device='cuda',dtype=torch.float64).T
                common=(vertices,cells,cache.physical_covariates[cad],B,cases,scale,guard)
                api=LocatedFieldwiseNeuralResponse(models['U'],models['S'],*common) if 'field_routes' in config else LocatedNeuralResponse(models['joint'],*common)
                cr=dict(nodes=n,cells=c,load_and_index_seconds=max(time.time()-tick[0],time.monotonic()-tick[1]),index=api.locator.metadata)
                rows=torch.as_tensor(np.linspace(0,c-1,config['geometry_cell_rows'],dtype=np.int64),device='cuda')
                bary=torch.tensor(config['interior_barycentric'],device='cuda',dtype=torch.float64)
                N,_,_=shape(torch,bary[:,:3]);interior=torch.einsum('pi,cia->cpa',N,vertices[cells[rows]]).reshape(-1,3)
                nodal_ids=torch.as_tensor(np.linspace(0,n-1,config['nodal_witnesses_per_CAD'],dtype=np.int64),device='cuda')
                xyz=torch.cat((interior,vertices[nodal_ids]),0)
                outside=(api.locator.domain_high+api.locator.extent)[None]
                before=cache.physical_covariates[cad].clone()
                owned={};tick=(time.time(),time.monotonic());state['phase']=cad+'/predict';persist()
                for load in config['loads']:
                    with torch.no_grad():
                        result=api.query(xyz,load)
                        if result['status']!='completed': raise ValueError('Known physical query rejected')
                        value=result['physical']; owned[load]=value
                        save(output/cad/(load+'_physical4.npy'),value)
                        save(output/cad/(load+'_owners.npy'),result['cell_rows'])
                        rejected=api.query(outside,load)
                        if rejected['status']!='rejected' or rejected['physical'] is not None: raise ValueError('Outside box not rejected')
                        state['outside_rejections']+=1;state['predicted_rows']+=len(xyz)
                        state['prediction_inventory'][cad+'/'+load]=str((output/cad/(load+'_physical4.npy')).resolve())
                cr['query_seconds']=max(time.time()-tick[0],time.monotonic()-tick[1])
                state['case_records'][cad]=cr;persist()
                # Prediction arrays exist before any held endpoint field is opened.
                references.append(dict(CAD=cad,nodal_ids=nodal_ids.cpu().tolist(),paths=record['held_predictions']))
                cr['continuous_checks']={}
                explicit_rows=rows[:,None].expand(-1,len(bary)).reshape(-1)
                for load in config['loads']:
                    with torch.no_grad():
                        owners=np.load(output/cad/(load+'_owners.npy'),allow_pickle=False)
                        if not np.array_equal(owners[:len(interior)],explicit_rows.cpu().numpy()):
                            raise ResponseNegative('Known interior owner differs')
                        direct=api.query_in_cells(interior,explicit_rows,load)
                        expected=owned[load][:len(interior)]
                        U=compare(direct[:,:3],expected[:,:3]); S=compare(direct[:,3],expected[:,3])
                        chunk=torch.cat([api.query(xyz[start:start+11],load)['physical'] for start in range(0,len(xyz),11)])
                        chunkU=compare(chunk[:,:3],owned[load][:,:3]);chunkS=compare(chunk[:,3],owned[load][:,3])
                    point=interior[:2].detach().clone().requires_grad_(True)
                    located=api.query(point,load)['physical'];direct=api.query_in_cells(point,explicit_rows[:2],load)
                    gradients=[]
                    for channel in range(4):
                        actual_grad,=torch.autograd.grad(located[:,channel].sum(),point,retain_graph=True)
                        reference_grad,=torch.autograd.grad(direct[:,channel].sum(),point,retain_graph=True)
                        if not bool(torch.isfinite(actual_grad).all()) or float(actual_grad.norm())<=0:
                            raise ArithmeticError('Actual coupled coordinate gradient missing')
                        gradients.append(compare(actual_grad,reference_grad))
                    cr['continuous_checks'][load]=dict(same_model_U=U,same_model_S=S,chunk_U=chunkU,chunk_S=chunkS,coordinate_gradient_relative_by_channel=gradients)
                if not torch.equal(before,cache.physical_covariates[cad]): raise ArithmeticError('Covariates changed')
                guard();persist()
                del api,cache,cases,vertices,cells,xyz,interior,nodal_ids,owned,before,point,located,direct,chunk,value,actual_grad,reference_grad
                gc.collect();torch.cuda.empty_cache()
            if state['predicted_rows']!=config['interior_query_rows']+config['nodal_query_rows'] or state['outside_rejections']!=config['outside_decisions']:
                raise ValueError('Complete continuous-query denominator differs')
            state['phase']='held_nodal_predictions';persist()
            for item in references:
                cad=item['CAD'];cr=state['case_records'][cad];cr['nodal_checks']={}
                for load in config['loads']:
                    guard()
                    held_paths=item['paths'][load]
                    held_paths=held_paths if isinstance(held_paths,dict) else {'joint':held_paths}
                    reference_parts={}
                    for field,path in held_paths.items():
                        held=np.load(local_path(path),allow_pickle=False);state['reference_reads']+=1
                        if held.dtype!=np.float64 or held.shape!=(cr['nodes'],4) or not np.isfinite(held).all(): raise ValueError('Held prediction schema differs')
                        selected=held[item['nodal_ids']]
                        reference_parts[field]=torch.as_tensor(selected,device='cuda')
                    reference=torch.cat((reference_parts['U'][:,:3],reference_parts['S'][:,3:4]),1) if 'field_routes' in config else reference_parts['joint']
                    own=np.load(output/cad/(load+'_physical4.npy'),allow_pickle=False)
                    actual=torch.as_tensor(own[-config['nodal_witnesses_per_CAD']:],device='cuda')
                    cr['nodal_checks'][load]=dict(U=compare(actual[:,:3],reference[:,:3]),S=compare(actual[:,3],reference[:,3]))
            state.update(status='completed',phase='terminal',continuous_interface_passed=True)
    except Exception as error:
        if isinstance(error,ResponseNegative) or type(error).__name__=='LocatorNegative':
            state.update(status='completed',phase='numerical_negative',continuous_interface_passed=False,reason=str(error))
        else:
            state.update(status='failed',phase='failure',error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(active=False,closed=True);persist()
        (output/'summary.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['config','source','output','lock']:parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
