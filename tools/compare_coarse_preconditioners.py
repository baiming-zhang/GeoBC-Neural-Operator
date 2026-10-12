"""Paired zero-start Jacobi/Chebyshev solves on one previously unstarted CAD."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import json
import os
from pathlib import Path
import signal


def run(args):
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    clocks = json.loads(os.environ['GEOBC_EXPERIMENT_CLOCK'])
    output = Path(args.output)
    state = {'status': 'running', 'phase': 'imports', 'CAD': config['CAD'], 'active': True, 'closed': False,
             'fixed_field_denominator': 6, 'saved_fields': 0, 'source_cache_reads': 0, 'source_members_read': [],
             'GT_reads': 0, 'raw_source_reads': 0, 'PT_reads': 0, 'training_updates': 0,
             'combined_solve_seconds': 0., 'groups': {}, 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    torch = None
    def elapsed(tick):
        return max(time.time()-tick[0], time.monotonic()-tick[1])
    def persist():
        state['native_seconds'] = {'UTC': time.time()-ENTRY_EPOCH, 'monotonic': time.monotonic()-ENTRY_MONOTONIC}
        (output/'status.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    def guard():
        if min(clocks['deadline_epoch']-time.time(), clocks['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Whole experiment deadline')
        if torch is not None and torch.cuda.is_initialized():
            state['peak_allocated_bytes'] = max(state['peak_allocated_bytes'], torch.cuda.max_memory_allocated())
            state['peak_reserved_bytes'] = max(state['peak_reserved_bytes'], torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            if max(state['peak_allocated_bytes'], state['peak_reserved_bytes'], total-free) > config['memory_bytes']:
                raise MemoryError('GPU budget exceeded')
    def save_array(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as file:
            np.save(file, value.detach().cpu().numpy(), allow_pickle=False)
        os.replace(temporary, path)
        return {'path': str(path.relative_to(output)), 'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('External deadline')))
    try:
        import numpy as np
        import torch as torch_module
        torch = torch_module
        from geobc.runtime import GPUQueue, CoarseLedger
        from geobc.elasticity import MatrixFreeElastic, _build_geometry, _algebra_guards, _restore_full
        from geobc.coarse import make_mapping, assemble_coarse, AdditiveCoarse, algebra_guards
        from geobc.parallel_elasticity import CanonicalLanesElastic, CanonicalLanesCoarse
        from geobc.chebyshev import ChebyshevCoarse
        from geobc.pcg import solve
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != clocks['boot_id']:
            raise RuntimeError('Launcher and worker boot differ')
        with GPUQueue(args.lock):
            total = torch.cuda.get_device_properties(0).total_memory
            if total <= config['memory_bytes']:
                raise MemoryError('Device capacity below the declared stage cap')
            torch.cuda.set_per_process_memory_fraction(config['memory_bytes']/total)
            torch.cuda.reset_peak_memory_stats()
            setup = (time.time(), time.monotonic())
            state['phase'] = 'source_cache'; persist()
            members = ['vertices_mm', 'cells10', 'free_node_ids', 'fixed_node_ids', 'force', 'lame',
                       'derived_node_ids', 'derived_source_ids', 'derived_rowptr', 'derived_blocks',
                       'reference_xyz_mm', 'canonical_forces']
            inputs = {}
            with np.load(args.source, allow_pickle=False) as source:
                for name in members:
                    guard(); value = source[name]
                    dtype = np.int64 if name in {'cells10', 'free_node_ids', 'fixed_node_ids', 'derived_node_ids', 'derived_source_ids', 'derived_rowptr'} else np.float64
                    if value.dtype != dtype or not np.isfinite(value).all():
                        raise ValueError('Invalid cached input '+name)
                    inputs[name] = torch.as_tensor(value, device='cuda')
            state['source_cache_reads'] = 1; state['source_members_read'] = members
            state['input_cache'] = str(Path(args.source).resolve())
            forces = inputs.pop('canonical_forces'); n = config['expected_geometry']['nodes']
            if forces.shape != (3, n, 3) or inputs['vertices_mm'].shape != (n, 3):
                raise ValueError('Declared CAD geometry/load shape differs')
            recipe = json.loads(Path('configs/physics.json').read_text(encoding='utf-8'))
            record = {'expected_geometry': config['expected_geometry'], 'recipe': recipe}
            state['phase'] = 'geometry'; persist()
            tick = time.monotonic(); prior = elapsed(setup)
            if args.resume_from:
                failed = json.loads((Path(args.resume_from)/'summary.json').read_text())
                if failed['saved_fields'] != 0 or not failed['K_checks']['fine']['passed']:
                    raise ValueError('This recovery expects saved stiffness before any PCG field')
                arrays = {}
                for name in ['gradients', 'weights', 'free_cholesky']:
                    value = np.load(Path(args.resume_from)/'cache'/(name+'.npy'), allow_pickle=False)
                    if value.dtype != np.float64 or not np.isfinite(value).all():
                        raise ValueError('Malformed recovery stiffness cache')
                    arrays[name] = torch.as_tensor(value, device='cuda')
                gradients, weights, chol = (arrays[key] for key in ['gradients', 'weights', 'free_cholesky'])
                lam, mu = (float(x) for x in inputs['lame'])
                operator = MatrixFreeElastic(torch, inputs['cells10'], gradients, weights, n, lam, mu)
                geometry, fine_checks = failed['geometry'], failed['K_checks']['fine']
                state['recovery'] = {'prior_attempt_seconds': failed['native_seconds'], 'fine_K_rebuilds': 0,
                                     'fine_eight_checks_repeated': 0, 'restored_operator_arrays': 3,
                                     'coarse_assemblies_including_unsaved_first_attempt': 2,
                                     'source_cache_reads_across_attempts': failed['source_cache_reads']+1}
            else:
                operator, chol, diagonal, geometry = _build_geometry(torch, inputs, record, tick, prior)
                fine_checks = _algebra_guards(torch, operator, diagonal, inputs, record, tick, prior)
            parallel = CanonicalLanesElastic(torch, operator, guard)
            gen = torch.Generator(device='cuda').manual_seed(2026)
            free = inputs['free_node_ids']
            w = torch.zeros_like(forces)
            w[:, free] = torch.randn((3, len(free), 3), dtype=torch.float64, device='cuda', generator=gen)
            batched = parallel.apply(w); scalar = torch.stack([operator.apply(value) for value in w])
            parity = torch.linalg.vector_norm(batched-scalar, dim=(1, 2))/torch.linalg.vector_norm(scalar, dim=(1, 2))
            if not bool((parity <= 1e-9).all()):
                raise ArithmeticError('Three-lane stiffness parity failed')
            state['K_checks'] = {'fine': fine_checks, 'three_lane_relative': parity.cpu().tolist()}
            state['geometry'] = geometry
            state['operator_cache'] = {key: save_array(output/'cache'/(key+'.npy'), value)
                                       for key, value in {'gradients': operator.gradients, 'weights': operator.weights, 'free_cholesky': chol}.items()}
            state['source_operator_seconds'] = elapsed(setup)
            if state['source_operator_seconds'] > 300:
                raise TimeoutError('Source/operator setup budget')
            if not args.resume_from:
                del diagonal
            del batched, scalar
            state['phase'] = 'shared_coarse'; persist(); tick = (time.time(), time.monotonic())
            ledger = CoarseLedger(torch, guard)
            P, PT, corners, cmap, fmap, mapping = make_mapping(torch, inputs, ledger)
            Ac, cd, gamma, assembly = assemble_coarse(torch, inputs, {'gradients': operator.gradients, 'weights': operator.weights}, cmap, fmap, ledger)
            for name, matrix in [('P', P), ('PT', PT), ('Ac', Ac)]:
                for component, value in [('rowptr', matrix.crow_indices()), ('columns', matrix.col_indices()), ('values', matrix.values())]:
                    save_array(output/'cache'/(name+'_'+component+'.npy'), value)
            save_array(output/'cache/coarse_diagonal.npy', cd)
            baseline_scalar = AdditiveCoarse(torch, P, PT, Ac, cd, gamma, chol, ledger)
            coarse_checks = algebra_guards(torch, baseline_scalar, operator, inputs, ledger)
            baseline = CanonicalLanesCoarse(torch, baseline_scalar, guard)
            state['shared_coarse'] = {'mapping': mapping, 'assembly': assembly, 'checks': coarse_checks, 'seconds': elapsed(tick)}
            if state['shared_coarse']['seconds'] > 120:
                raise TimeoutError('Shared coarse setup budget')
            snapshots = [value.clone() for value in [inputs['vertices_mm'], inputs['cells10'], forces, operator.gradients, operator.weights, chol]]
            free_w = w[:, free]
            mb = baseline.apply(free_w); ms = torch.stack([baseline_scalar.apply(value) for value in free_w])
            baseline_parity = torch.linalg.vector_norm(mb-ms, dim=(1, 2))/torch.linalg.vector_norm(ms, dim=(1, 2))
            if not bool((baseline_parity <= 1e-9).all()):
                raise ArithmeticError('Baseline three-lane preconditioner parity failed')
            def apply(value):
                full = torch.zeros_like(forces); full[:, free] = value
                return parallel.apply(full)[:, free]
            def checkpoint(folder):
                folder.mkdir(parents=True)
                def save(values):
                    temporary = folder/'last_state.npz.tmp'
                    with temporary.open('wb') as file:
                        np.savez(file, **{key: values[key].detach().cpu().numpy() for key in ['x', 'r', 'p', 'rz', 'active', 'iterations']}, rhs=forces[:, free].cpu().numpy())
                    os.replace(temporary, folder/'last_state.npz')
                    state['running_group'] = {key: values[key] for key in ['step', 'seconds', 'trace']}
                    persist()
                return save
            full_fields = {}
            for name in ['jacobi', 'chebyshev']:
                guard(); state['phase'] = name+'/independent_zero_PCG'; persist()
                construction = (time.time(), time.monotonic())
                if name == 'jacobi':
                    precondition = baseline.apply
                else:
                    candidate = ChebyshevCoarse(torch, P, PT, Ac, cd, gamma, chol, ledger)
                    mc = candidate.apply_lanes(free_w); mr = torch.stack([candidate.apply(value) for value in free_w])
                    cp = torch.linalg.vector_norm(mc-mr, dim=(1, 2))/torch.linalg.vector_norm(mr, dim=(1, 2))
                    symmetry_den = torch.linalg.vector_norm(free_w[0])*torch.linalg.vector_norm(mc[1])+torch.linalg.vector_norm(free_w[1])*torch.linalg.vector_norm(mc[0])
                    symmetry = float(((free_w[0]*mc[1]).sum()-(free_w[1]*mc[0]).sum()).abs()/symmetry_den)
                    energies = (free_w*mc).sum((1, 2))
                    if not bool((cp <= 1e-9).all() and (energies > 0).all() and torch.isfinite(mc).all()) or symmetry > 1e-9:
                        raise ArithmeticError('Real Chebyshev positivity/symmetry/lane parity failed')
                    linear = candidate.apply_lanes(2*free_w[:1]-free_w[1:2])
                    linear_ref = 2*mc[:1]-mc[1:2]
                    linear_error = float(torch.linalg.vector_norm(linear-linear_ref)/torch.linalg.vector_norm(linear_ref))
                    if linear_error > 1e-9:
                        raise ArithmeticError('Real Chebyshev linearity failed')
                    state['candidate_checks'] = {'three_lane_relative': cp.cpu().tolist(), 'symmetry_relative': symmetry,
                                                 'positive_energies': energies.cpu().tolist(), 'linearity_relative': linear_error}
                    precondition = candidate.apply_lanes
                construction_seconds = elapsed(construction)
                tick = (time.time(), time.monotonic())
                seconds = min(config['group_seconds'], config['combined_solve_seconds']-state['combined_solve_seconds'],
                              clocks['deadline_epoch']-time.time(), clocks['deadline_monotonic']-time.monotonic())
                z, result = solve(apply, forces[:, free], precondition, guard, checkpoint(output/name/'checkpoint'),
                                  tolerance=config['residual_tolerance'], maximum_iterations=10000, seconds=seconds)
                torch.cuda.synchronize(); solve_seconds = elapsed(tick)
                state['combined_solve_seconds'] += solve_seconds
                if state['combined_solve_seconds'] > config['combined_solve_seconds']:
                    raise TimeoutError('Combined solve budget')
                fields = torch.stack([_restore_full(torch, value, inputs) for value in z])
                true_residuals = []
                equilibrium = []
                for i, load in enumerate(config['canonical_loads']):
                    guard(); residual = operator.apply(fields[i])-forces[i]
                    true = float(torch.linalg.vector_norm(residual[free])/torch.linalg.vector_norm(forces[i, free]))
                    reaction = torch.zeros_like(residual); reaction[inputs['fixed_node_ids']] = residual[inputs['fixed_node_ids']]
                    external = forces[i]+reaction
                    if not bool(torch.isfinite(fields[i]).all() and torch.isfinite(residual).all()):
                        raise ArithmeticError('Nonfinite endpoint')
                    item = save_array(output/name/load/'displacement_mm.npy', fields[i])
                    equilibrium.append({'load': load, 'field': item, 'true_relative_residual': true,
                                        'net_force_N': external.sum(0).cpu().tolist(),
                                        'net_moment_Nmm': torch.cross(inputs['vertices_mm']-inputs['reference_xyz_mm'], external, dim=-1).sum(0).cpu().tolist()})
                    true_residuals.append(true); state['saved_fields'] += 1
                state['groups'][name] = {'construction_checks_seconds': construction_seconds, 'solve_seconds': solve_seconds,
                                        'solver': result, 'endpoints': equilibrium, 'true_relative_residual': true_residuals}
                full_fields[name] = fields
                persist()
            state['phase'] = 'six_field_barrier'; persist()
            if state['saved_fields'] != 6:
                raise RuntimeError('Incomplete fixed Cartesian fields')
            (output/'field_barrier.json').write_text(json.dumps({'fields': 6, 'groups': state['groups'], 'GT_reads': 0}, indent=2)+'\n', encoding='utf-8')
            # Same-equation comparison uses resident fields only, after all six saves.
            norms = torch.linalg.vector_norm(full_fields['jacobi'], dim=(1, 2))
            errors = torch.linalg.vector_norm(full_fields['chebyshev']-full_fields['jacobi'], dim=(1, 2))
            same = [float(e/n) if float(n) > 0 else None for e, n in zip(errors, norms)]
            zero_abs = (full_fields['chebyshev']-full_fields['jacobi']).abs().amax((1, 2)).cpu().tolist()
            preserved = all(torch.equal(a, b) for a, b in zip(snapshots, [inputs['vertices_mm'], inputs['cells10'], forces, operator.gradients, operator.weights, chol]))
            precision = all(value <= 1e-8 for group in state['groups'].values() for value in group['true_relative_residual'])
            precision &= all(value <= config['same_solution_tolerance'] if value is not None else absolute <= 1e-12 for value, absolute in zip(same, zero_abs)) and preserved
            baseline_seconds = state['groups']['jacobi']['solve_seconds']
            candidate_seconds = state['groups']['chebyshev']['construction_checks_seconds']+state['groups']['chebyshev']['solve_seconds']
            reduction = 1-candidate_seconds/baseline_seconds
            state.update(status='completed', phase='terminal', same_solution_relative=same, input_operator_unchanged=preserved,
                         precision_passed=precision, paired_cost_reduction=reduction,
                         cost_passed=reduction >= config['required_cost_reduction'],
                         candidate_passed=precision and reduction >= config['required_cost_reduction'])
    except Exception as error:
        state.update(status='failed', phase='technical_failure', error=type(error).__name__+': '+str(error))
        raise
    finally:
        state.update(active=False, closed=True)
        persist()
        (output/'summary.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lock', required=True)
    parser.add_argument('--resume-from')
    run(parser.parse_args())
