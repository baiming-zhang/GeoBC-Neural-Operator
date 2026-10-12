"""Complete fixed TRAIN inputs using the validated Chebyshev candidate and saved fields."""
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
    a = {'status': 'running', 'active': True, 'closed': False, 'phase': 'imports',
         'fixed_CAD_denominator': 100, 'fixed_load_denominator': 300, 'reused_feature_CADs': 95,
         'new_zero_fields': 0, 'reused_comparison_fields': 3, 'completed_new_feature_CADs': 0,
         'feature_inventory': dict(catalog['existing_features']), 'case_records': {},
         'source_cache_reads': 0, 'owned_array_reads': 0, 'GT_reads': 0, 'raw_source_reads': 0,
         'neural_training_released': False, 'combined_solve_seconds': 0., 'gradient_seconds': 0.,
         'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0, 'complete_inputs': False}
    torch = None
    def elapsed(tick):
        return max(time.time()-tick[0], time.monotonic()-tick[1])
    def persist():
        a['native_seconds'] = {'UTC': time.time()-ENTRY_EPOCH, 'monotonic': time.monotonic()-ENTRY_MONOTONIC}
        (output/'status.json').write_text(json.dumps(a, indent=2)+'\n', encoding='utf-8')
    def guard():
        if min(clock['deadline_epoch']-time.time(), clock['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Whole completion deadline')
        if torch is not None and torch.cuda.is_initialized():
            a['peak_allocated_bytes'] = max(a['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            a['peak_reserved_bytes'] = max(a['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            if max(total-free, a['peak_allocated_bytes'], a['peak_reserved_bytes']) > config['memory_bytes']:
                raise MemoryError('Memory budget')
    def save_array(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as file:
            np.save(file, value.detach().cpu().numpy(), allow_pickle=False)
        os.replace(temporary, path)
        return {'path': str(path.resolve()), 'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('External completion deadline')))
    try:
        import gc
        import numpy as np
        import torch as tm
        torch = tm
        from geobc.runtime import GPUQueue, CoarseLedger
        from geobc.physics_inputs import load_source, load_array, load_coarse, local_path
        from geobc.elasticity import MatrixFreeElastic, ScientificNegative, _build_geometry, _algebra_guards, _restore_full
        from geobc.coarse import make_mapping, assemble_coarse, AdditiveCoarse, algebra_guards
        from geobc.parallel_elasticity import CanonicalLanesElastic
        from geobc.chebyshev import ChebyshevCoarse
        from geobc.pcg import solve
        from geobc.kinematics import recover_kinematic_inputs, qualify_derivative
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if len(catalog['existing_features']) != 95 or len(catalog['CADs']) != 100:
            raise ValueError('The fixed inherited input prefix differs')
        with GPUQueue(args.lock):
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            recipe = json.loads(Path('configs/physics.json').read_text())
            order = [config['existing_comparison_CAD']]+config['new_solver_CADs']
            for cad in order:
                guard(); a['phase'] = cad+'/source'; persist()
                case = catalog['cases'][cad]; tick = (time.time(), time.monotonic())
                folder = output/cad; folder.mkdir()
                inputs, forces = load_source(case['source'], case['expected_geometry'], guard)
                a['source_cache_reads'] += 1
                record = {'recipe': recipe, 'expected_geometry': case['expected_geometry']}
                if cad == config['existing_comparison_CAD']:
                    weights = load_array(case['weights'], guard)
                    U = torch.stack([load_array(catalog['comparison_fields'][label], guard) for label in ['ver', 'hor', 'tor']])
                    a['owned_array_reads'] += 4
                    cr = {'source_seconds': elapsed(tick), 'new_K_PCG': False, 'existing_three_comparison_fields': True}
                else:
                    a['phase'] = cad+'/operator'; persist()
                    ledger = CoarseLedger(torch, guard)
                    if 'operator_arrays' in case:
                        arrays = {name: load_array(item, guard) for name, item in case['operator_arrays'].items()}
                        a['owned_array_reads'] += 3
                        chol = arrays['free_cholesky']; lam, mu = (float(x) for x in inputs['lame'])
                        operator = MatrixFreeElastic(torch, inputs['cells10'], arrays['gradients'], arrays['weights'], len(inputs['vertices_mm']), lam, mu)
                        scalar = load_coarse(case['coarse_cache'], chol, ledger, guard)
                        a['owned_array_reads'] += 10
                        cr = {'fine_K_rebuilds': 0, 'fine_eight_checks_repeated': 0, 'coarse_assembly_repeated': 0}
                    else:
                        start = time.monotonic(); prior = elapsed(tick)
                        operator, chol, diagonal, geometry = _build_geometry(torch, inputs, record, start, prior)
                        fine = _algebra_guards(torch, operator, diagonal, inputs, record, start, prior)
                        arrays = {'gradients': operator.gradients, 'weights': operator.weights, 'free_cholesky': chol}
                        for name, value in arrays.items(): save_array(folder/'cache'/(name+'.npy'), value)
                        P, PT, corners, cmap, fmap, mapping = make_mapping(torch, inputs, ledger)
                        Ac, cd, gamma, assembly = assemble_coarse(torch, inputs, arrays, cmap, fmap, ledger)
                        scalar = AdditiveCoarse(torch, P, PT, Ac, cd, gamma, chol, ledger)
                        checks = algebra_guards(torch, scalar, operator, inputs, ledger)
                        cr = {'fine_K_rebuilds': 1, 'fine': fine, 'geometry': geometry, 'coarse': checks, 'mapping': mapping, 'assembly': assembly}
                        for name, matrix in [('P', P), ('PT', PT), ('Ac', Ac)]:
                            for component, value in [('rowptr', matrix.crow_indices()), ('columns', matrix.col_indices()), ('values', matrix.values())]:
                                save_array(folder/'cache'/(name+'_'+component+'.npy'), value)
                        save_array(folder/'cache/coarse_diagonal.npy', cd)
                        del diagonal, corners, cmap, fmap
                    weights = arrays['weights']; free = inputs['free_node_ids']
                    parallel = CanonicalLanesElastic(torch, operator, guard)
                    candidate = ChebyshevCoarse(torch, scalar.P, scalar.PT, scalar.Ac, scalar.diagonal, 1/scalar.omega, chol, ledger)
                    gen = torch.Generator(device='cuda').manual_seed(2026)
                    w = torch.zeros_like(forces); w[:, free] = torch.randn((3, len(free), 3), dtype=torch.float64, device='cuda', generator=gen)
                    K = parallel.apply(w); Kref = torch.stack([operator.apply(v) for v in w])
                    kp = torch.linalg.vector_norm(K-Kref, dim=(1, 2))/torch.linalg.vector_norm(Kref, dim=(1, 2))
                    M = candidate.apply_lanes(w[:, free]); Mref = torch.stack([candidate.apply(v) for v in w[:, free]])
                    mp = torch.linalg.vector_norm(M-Mref, dim=(1, 2))/torch.linalg.vector_norm(Mref, dim=(1, 2))
                    symmetry = abs(float((w[0, free]*M[1]).sum()-(w[1, free]*M[0]).sum()))/float(torch.linalg.vector_norm(w[0, free])*torch.linalg.vector_norm(M[1])+torch.linalg.vector_norm(w[1, free])*torch.linalg.vector_norm(M[0]))
                    if not bool((kp<=1e-9).all() and (mp<=1e-9).all() and ((w[:, free]*M).sum((1, 2))>0).all()) or symmetry>1e-9:
                        raise ScientificNegative('Actual K/M lane or polynomial energy qualification failed')
                    cr.update(setup_seconds=elapsed(tick), K_relative=kp.cpu().tolist(), M_relative=mp.cpu().tolist(), M_symmetry=symmetry)
                    if cr['setup_seconds'] > 420:
                        raise TimeoutError('Case source/operator/coarse setup budget')
                    del w, K, Kref, M, Mref
                    a['case_records'][cad] = cr; a['phase'] = cad+'/Chebyshev_zero_PCG'; persist()
                    def apply(value):
                        full = torch.zeros_like(forces); full[:, free] = value
                        return parallel.apply(full)[:, free]
                    def checkpoint(values):
                        temporary = folder/'last_state.npz.tmp'
                        with temporary.open('wb') as file:
                            np.savez(file, **{key: values[key].detach().cpu().numpy() for key in ['x', 'r', 'p', 'rz', 'active', 'iterations']}, rhs=forces[:, free].cpu().numpy())
                        os.replace(temporary, folder/'last_state.npz')
                        a['running_group'] = {'CAD': cad, 'step': values['step'], 'seconds': values['seconds'], 'trace': values['trace']}; persist()
                    solve_tick = (time.time(), time.monotonic())
                    seconds = min(300, config['combined_solve_seconds']-a['combined_solve_seconds'])
                    z, result = solve(apply, forces[:, free], candidate.apply_lanes, guard, checkpoint, seconds=seconds)
                    torch.cuda.synchronize(); wall = elapsed(solve_tick); a['combined_solve_seconds'] += wall
                    U = torch.stack([_restore_full(torch, value, inputs) for value in z])
                    endpoints = []
                    for i, load in enumerate(['ver', 'hor', 'tor']):
                        residual = operator.apply(U[i])-forces[i]
                        relative = float(torch.linalg.vector_norm(residual[free])/torch.linalg.vector_norm(forces[i, free]))
                        reaction = torch.zeros_like(residual); reaction[inputs['fixed_node_ids']] = residual[inputs['fixed_node_ids']]
                        external = forces[i]+reaction
                        field = save_array(folder/load/'displacement_mm.npy', U[i]); a['new_zero_fields'] += 1
                        endpoints.append({'load': load, 'field': field, 'true_relative_residual': relative,
                                          'net_force_N': external.sum(0).cpu().tolist(),
                                          'net_moment_Nmm': torch.cross(inputs['vertices_mm']-inputs['reference_xyz_mm'], external, dim=-1).sum(0).cpu().tolist()})
                    cr.update(solve_seconds=wall, solver=result, endpoints=endpoints)
                    persist()
                    if any(row['true_relative_residual']>1e-8 for row in endpoints):
                        raise ScientificNegative('Fixed new-candidate case not converged')
                a['phase'] = cad+'/full_gradient'; persist(); ft = (time.time(), time.monotonic())
                before = U.clone()
                derivative = qualify_derivative(np, torch, inputs, U, guard)
                features, qualification = recover_kinematic_inputs(torch, inputs['vertices_mm'], inputs['cells10'], U, weights.sum(1), guard)
                if not torch.equal(before, U) or not torch.equal(features[..., :3], U):
                    raise ArithmeticError('Displacement changed during gradient recovery')
                field = save_array(folder/'kinematic_inputs.npy', features)
                fw = elapsed(ft); a['gradient_seconds'] += fw
                if a['gradient_seconds'] > config['gradient_seconds']:
                    raise TimeoutError('Cumulative gradient budget')
                cr.update(gradient_seconds=fw, derivative=derivative, qualification=qualification)
                a['case_records'][cad] = cr; a['feature_inventory'][cad] = field; a['completed_new_feature_CADs'] += 1
                persist()
                del inputs, forces, U, before, features, weights
                if cad != config['existing_comparison_CAD']:
                    del operator, arrays, scalar, parallel, candidate, chol, z
                gc.collect(); torch.cuda.empty_cache()
            if set(a['feature_inventory']) != set(catalog['CADs']) or a['new_zero_fields'] != 12 or a['completed_new_feature_CADs'] != 5:
                raise RuntimeError('Full100 input Cartesian set differs')
            for item in a['feature_inventory'].values():
                if not local_path(item['path']).is_file():
                    raise FileNotFoundError(item['path'])
            index = {'CADs': catalog['CADs'], 'scientific_group': 'TRAIN', 'canonical_loads': ['ver', 'hor', 'tor'],
                     'features': a['feature_inventory'], 'complete_CADs': 100, 'complete_loads': 300,
                     'original_production_seconds': catalog['original_production_seconds'],
                     'comparison_campaign_UTC_seconds': catalog['comparison_campaign_UTC_seconds'],
                     'GT_reads': 0, 'training_released': False}
            (output/'complete_inputs.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
            a.update(status='completed', phase='terminal', complete_inputs=True)
    except Exception as error:
        if type(error).__name__ == 'ScientificNegative' or isinstance(error, ValueError) and str(error).startswith(('Original curved', 'Original nodes', 'Actual P2')):
            a.update(status='completed', phase='scientific_input_negative', complete_inputs=False, reason=str(error))
        else:
            a.update(status='failed', phase='technical_failure', error=type(error).__name__+': '+str(error)); raise
    finally:
        a.update(active=False, closed=True); persist()
        (output/'summary.json').write_text(json.dumps(a, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lock', required=True)
    run(parser.parse_args())
