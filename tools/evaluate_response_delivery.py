"""Exercise the public JSON request handler on the complete prepared final catalog."""
import time
ENTRY_EPOCH,ENTRY_MONOTONIC=time.time(),time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(args):
    config=json.loads(Path(args.config).read_text());catalog=json.loads(Path(args.source).read_text())
    output=Path(args.output);clock=json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK'])
    state=dict(status='running',active=True,closed=False,phase='imports',case_records={},predicted_rows=0,outside_rejections=0,
               completed_CADs=0,GT_reads=0,training_updates=0,scaler_fits=0,peak_allocated_bytes=0,peak_reserved_bytes=0)
    torch=None;session=None
    def persist():
        if session is not None:state['counts']=session.counts.copy()
        state['native_seconds']=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=0:raise TimeoutError('Delivery-stage deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes']=max(state['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes']=max(state['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
            free,total=torch.cuda.mem_get_info()
            if max(total-free,state['peak_allocated_bytes'],state['peak_reserved_bytes'])>config['memory_bytes']:raise MemoryError('Delivery-stage memory budget')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External delivery deadline')))
    try:
        import numpy as np
        import torch as tm
        torch=tm
        from geobc.deployment import PreparedResponseSession,handle_line
        from geobc.runtime import GPUQueue
        from geobc.query import shape
        split=json.loads(Path('configs/split.json').read_text())
        if list(catalog['cases'])!=split['final'] or catalog['seed']!=config['seed'] or catalog['scaler']!=json.loads(Path('configs/output_scaler.json').read_text()):raise ValueError('Frozen deployment catalog differs')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
            state['phase']='model_load';persist();tick=(time.time(),time.monotonic())
            session=PreparedResponseSession(catalog,guard)
            state['model_load_seconds']=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]);persist()
            with (output/'requests.jsonl').open('x') as requests,(output/'responses.jsonl').open('x') as responses:
                for cad in split['final']:
                    guard();state['phase']=cad+'/prepare';persist();tick=(time.time(),time.monotonic())
                    api=session.open_CAD(cad)
                    prepare=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1])
                    cells=api.covariates.cells;vertices=api.covariates.reference.vertices
                    rows=torch.as_tensor(np.linspace(0,len(cells)-1,config['geometry_cell_rows'],dtype=np.int64),device='cuda')
                    bary=torch.tensor([config['interior_barycentric'][:3]],device='cuda',dtype=torch.float64)
                    N,_,_=shape(torch,bary)
                    xyz=torch.einsum('pi,cia->cpa',N,vertices[cells[rows]]).reshape(-1,3).cpu().tolist()
                    outside=(api.locator.domain_high+api.locator.extent)[None].cpu().tolist()
                    state['phase']=cad+'/JSON_query';persist();tick=(time.time(),time.monotonic())
                    for load in config['loads']:
                        for points,kind in [(xyz,'interior'),(outside,'outside')]:
                            line=json.dumps(dict(cad=cad,load=load,xyz_mm=points),allow_nan=False)
                            requests.write(line+'\n');requests.flush()
                            answer=handle_line(session,line)
                            responses.write(json.dumps(answer,allow_nan=False)+'\n');responses.flush()
                            if kind=='interior':
                                U=np.asarray(answer['U_mm']);S=np.asarray(answer['signed_S_MPa'])
                                if answer['status']!='completed' or U.shape!=(len(xyz),3) or S.shape!=(len(xyz),) or not np.isfinite(U).all() or not np.isfinite(S).all():raise ValueError('Complete interior response required')
                                state['predicted_rows']+=len(xyz)
                            else:
                                if answer['status']!='rejected' or answer['U_mm'] is not None or answer['signed_S_MPa'] is not None:raise ValueError('Explicit outside rejection required')
                                state['outside_rejections']+=1
                    state['case_records'][cad]=dict(nodes=len(vertices),cells=len(cells),prepare_seconds=prepare,query_seconds=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]),interior_rows=3*len(xyz),outside_rejections=3)
                    state['completed_CADs']+=1;persist()
                    del api,vertices,cells,rows,N,xyz
            if state['completed_CADs']!=20 or state['predicted_rows']!=config['interior_response_rows'] or state['outside_rejections']!=config['outside_decisions'] or session.counts['checkpoint_reads']!=1 or session.counts['geometry_reads']!=20 or session.counts['feature_reads']!=20:raise ValueError('Complete deployment denominator required')
            guard();state.update(status='completed',phase='terminal',prepared_response_delivery_passed=True)
    except Exception as error:
        state.update(status='failed',phase='failure',error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(active=False,closed=True);persist();(output/'summary.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','source','output','lock']:p.add_argument('--'+name,required=True)
    run(p.parse_args())
