"""Recover development covariates from existing qualified displacement fields."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(args):
    config = json.loads(Path(args.config).read_text())
    catalog = json.loads(Path(args.source).read_text())
    clock = json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK'])
    output = Path(args.output)
    state = dict(status='running', active=True, closed=False, phase='imports',
                 scientific_group='knownDEV', CAD_denominator=20, load_denominator=60,
                 complete_inputs=False, case_records={}, features={}, source_reads=0,
                 displacement_reads=0, weight_reads=0, gradient_seconds=0., GT_reads=0,
                 K_PCG_runs=0, peak_allocated_bytes=0, peak_reserved_bytes=0)
    torch = None
    def persist():
        state['native_seconds'] = dict(UTC=time.time()-ENTRY_EPOCH, monotonic=time.monotonic()-ENTRY_MONOTONIC)
        temporary = output/'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        os.replace(temporary, output/'status.json')
    def guard():
        if min(clock['deadline_epoch']-time.time(), clock['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Development input deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes'] = max(state['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes'] = max(state['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            if max(total-free, state['peak_allocated_bytes'], state['peak_reserved_bytes']) > config['memory_bytes']:
                raise MemoryError('Development input memory budget')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('External deadline')))
    try:
        import gc
        import numpy as np
        import torch as tm
        torch = tm
        from geobc.runtime import GPUQueue
        from geobc.physics_inputs import local_path, load_array
        from geobc.kinematics import recover_kinematic_inputs, qualify_derivative
        train = json.loads(local_path('publication/GeoBC-Neural-Operator/runs/physical_inputs_completion/complete_inputs.json').read_text())
        if train['complete_CADs'] != 100 or train['complete_loads'] != 300:
            raise ValueError('Training input production is incomplete')
        split = json.loads(Path('configs/split.json').read_text())
        if set(catalog['CADs']) != set(split['development']) or len(catalog['CADs']) != 20:
            raise ValueError('Fixed development set differs')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            for cad in catalog['CADs']:
                guard(); state['phase'] = cad+'/load'; persist()
                tick = (time.time(), time.monotonic())
                case = catalog['case_records'][cad]
                if case['scientific_group'] != 'knownDEV':
                    raise ValueError('Incorrect predictor role')
                expected = case['expected_geometry']; n, c = expected['nodes'], expected['cells']
                inputs = {}
                with np.load(local_path(case['input_identity']['path']), allow_pickle=False) as archive:
                    for key, dtype, shape in [('vertices_mm', np.float64, (n, 3)), ('cells10', np.int64, (c, 10))]:
                        guard(); value = archive[key]
                        if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
                            raise ValueError('Development geometry schema differs')
                        inputs[key] = torch.as_tensor(value, device='cuda')
                state['source_reads'] += 1
                weights = load_array(case['positive_fourQ_weights'], guard); state['weight_reads'] += 1
                basis = []
                for load in ['ver', 'hor', 'tor']:
                    record = case['canonical_basis'][load]
                    if record['true_relative_residual'] > 1e-8:
                        raise ValueError('Inherited displacement is not converged')
                    field = load_array(record['physical_U'], guard); state['displacement_reads'] += 1
                    if field.dtype != torch.float64 or field.shape != (n, 3):
                        raise ValueError('Displacement schema differs')
                    basis.append(field)
                U = torch.stack(basis)
                if weights.dtype != torch.float64 or weights.shape != (c, 4) or not bool((weights>0).all()):
                    raise ValueError('Positive curved-cell volume weights differ')
                torch.cuda.synchronize()
                load_seconds = max(time.time()-tick[0], time.monotonic()-tick[1])
                state['phase'] = cad+'/full_gradient'; persist()
                tick = (time.time(), time.monotonic()); before = U.clone()
                derivative = qualify_derivative(np, torch, inputs, U, guard)
                features, qualification = recover_kinematic_inputs(torch, inputs['vertices_mm'], inputs['cells10'], U, weights.sum(1), guard)
                if not torch.equal(before, U) or not torch.equal(features[..., :3], U):
                    raise ArithmeticError('Original displacement changed')
                folder = output/cad; folder.mkdir()
                destination = folder/'kinematic_inputs.npy'
                temporary = destination.with_suffix('.npy.tmp')
                with temporary.open('wb') as file:
                    np.save(file, features.detach().cpu().numpy(), allow_pickle=False)
                os.replace(temporary, destination)
                guard(); seconds = max(time.time()-tick[0], time.monotonic()-tick[1])
                state['gradient_seconds'] += seconds
                if state['gradient_seconds'] > config['gradient_seconds']:
                    raise TimeoutError('Cumulative development gradient budget')
                state['features'][cad] = dict(path=str(destination.resolve()), shape=list(features.shape), dtype='float64')
                state['case_records'][cad] = dict(nodes=n, cells=c, load_seconds=load_seconds, gradient_seconds=seconds,
                                                  derivative=derivative, qualification=qualification,
                                                  inherited_actual_cost=case['inherited_actual_cost'])
                persist()
                del inputs, weights, U, basis, field, before, features
                gc.collect(); torch.cuda.empty_cache()
            if set(state['features']) != set(catalog['CADs']) or state['displacement_reads'] != 60:
                raise ValueError('Development Cartesian input set incomplete')
            index = dict(CADs=catalog['CADs'], scientific_group='knownDEV', canonical_loads=['ver','hor','tor'],
                         features=state['features'], complete_CADs=20, complete_loads=60, GT_reads=0)
            (output/'complete_inputs.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
            state.update(status='completed', phase='terminal', complete_inputs=True)
    except Exception as error:
        state.update(status='failed', phase='failure', error=type(error).__name__+': '+str(error))
        raise
    finally:
        state.update(active=False, closed=True); persist()
        (output/'summary.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['config', 'source', 'output', 'lock']:
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
