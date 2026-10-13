"""CPU-only source qualification for the fixed final CAD set, without labels."""
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
    state=dict(status='running',active=True,closed=False,phase='imports',scientific_group='FINAL',CAD_denominator=20,load_denominator=60,
               case_records={},sources={},complete_sources=False,GT_reads=0,GPU_calls=0,K_PCG_runs=0,
               counts=dict(H5_attempts=0,H5_inputs=0,permutation_reads=0,coordinate_attempts=0,GRID_XYZ_inputs=0,deck_parses=0))
    def persist():
        state['native_seconds']=dict(UTC=time.time()-ENTRY_EPOCH,monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(),clock['deadline_monotonic']-time.monotonic())<=10:
            raise TimeoutError('Whole source stage deadline')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('External deadline')))
    try:
        import numpy as np
        from geobc.runtime import GPUQueue
        from geobc.source_inputs import prepare_source,UnsupportedInput
        recipe=json.loads(Path('configs/physics.json').read_text())
        split=json.loads(Path('configs/split.json').read_text())
        if catalog['CADs']!=split['final'] or set(catalog['cases'])!=set(split['final']) or len(split['final'])!=20:
            raise ValueError('Fixed final set differs')
        with GPUQueue(args.lock):
            for cad in catalog['CADs']:
                guard();state['phase']=cad+'/target_free_source';persist();tick=(time.time(),time.monotonic())
                record=catalog['cases'][cad]
                if record['scientific_group']!='FINAL':raise ValueError('Source role differs')
                def source_guard():
                    guard()
                    if max(time.time()-tick[0],time.monotonic()-tick[1])>config['source_seconds']:
                        raise TimeoutError('Per-CAD source allowance')
                try:
                    arrays,facts=prepare_source(record,recipe,source_guard,state['counts'],config['source_seconds'])
                    folder=output/cad;folder.mkdir()
                    path=folder/'geometry_and_canonical_rhs.npz';temporary=path.with_suffix('.npz.tmp')
                    with temporary.open('wb') as file:np.savez(file,**arrays)
                    os.replace(temporary,path)
                    state['sources'][cad]=dict(path=str(path.resolve()),expected_geometry=facts['expected_geometry'])
                    state['case_records'][cad]=dict(status='completed',**facts,seconds=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]))
                    (folder/'source.json').write_text(json.dumps(state['case_records'][cad],indent=2)+'\n')
                    source_guard()
                    del arrays
                except UnsupportedInput as error:
                    state['case_records'][cad]=dict(status='input_negative',reason=str(error),seconds=dict(UTC=time.time()-tick[0],monotonic=time.monotonic()-tick[1]))
                persist()
            state['complete_sources']=len(state['sources'])==20
            state.update(status='completed',phase='terminal')
            (output/'source_index.json').write_text(json.dumps(dict(CADs=catalog['CADs'],scientific_group='FINAL',sources=state['sources'],complete_sources=state['complete_sources'],GT_reads=0),indent=2)+'\n')
    except Exception as error:
        state.update(status='failed',phase='failure',error=type(error).__name__+': '+str(error));raise
    finally:
        state.update(active=False,closed=True);persist()
        (output/'summary.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','source','output','lock']:p.add_argument('--'+name,required=True)
    run(p.parse_args())
