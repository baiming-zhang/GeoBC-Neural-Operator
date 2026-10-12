"""Reference-U diagnostic, never a deployed stress solver or signed-S rule."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.reference_vm_contract_v87 import approved
    p = approved(plan_path); s = p['scientific']; out = d.path(p['output'])
    outer = json.loads(os.environ['GEOBC_REFERENCE_VM_V87_GUARD'])
    assert outer['scientific_digest'] == p['scientific_digest']
    assert outer['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert out.exists() and not (out/'execution.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, boot_id=outer['boot_id'],
                     scientific_digest=p['scientific_digest'])
    d.write(out/'execution.json', execution)
    a = dict(status='running', active=True, closed=False, execution=execution, phase='setup',
        case_metrics={}, reference_q_arrays={}, training_updates=0, DEV_FINAL_reads=0,
        raw_CSV_H5_reads=0, PT_reads=0, new_physical_solves=0, sign_rule_search=False,
        fit_or_calibration=False, read_receipt=[], GT_U_archives_attempted=0, GT_U_archives_completed=0,
        cached_S_attempted=0, cached_S_completed=0, all_q_committed_before_S=False)
    torch = None; lock = None

    def persist():
        a['elapsed_epoch_seconds'] = time.time()-ENTRY_EPOCH
        a['elapsed_monotonic_seconds'] = time.monotonic()-ENTRY_MONOTONIC
        d.write(out/'status.json', a)
        d.write(out/'clock.json', dict(active=a['active'], closed=a['closed'], status=a['status'],
            elapsed_epoch_seconds=a['elapsed_epoch_seconds'], elapsed_monotonic_seconds=a['elapsed_monotonic_seconds'],
            origin_epoch=ENTRY_EPOCH, origin_monotonic=ENTRY_MONOTONIC, outer=outer))

    def guard():
        assert min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic()) > 0, 'Absolute stage deadline exceeded'
        if torch is not None:
            assert max(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) <= s['memory_limit_bytes'], 'GPU budget exceeded'

    def sha_values(np, x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

    def load_npy(np, rec, kind):
        guard(); d.stat_matches(rec)
        a['read_receipt'].append(dict(kind=kind, path=rec['path'], attempted=True, completed=False)); persist()
        v = np.load(d.path(rec['path']), allow_pickle=False)
        assert np.isfinite(v).all()
        if 'shape' in rec: assert list(v.shape) == rec['shape']
        if 'dtype' in rec and rec['dtype'].startswith('torch.'):
            assert str(v.dtype) == rec['dtype'][6:]
        elif 'dtype' in rec: assert str(v.dtype) == rec['dtype']
        if 'values_sha256' in rec: assert sha_values(np, v) == rec['values_sha256']
        a['read_receipt'][-1]['completed'] = True
        return v

    try:
        persist(); guard()
        import fcntl
        lock = d.path(s['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch as torch_module
        torch = torch_module; torch.cuda.set_device(0)
        assert torch.cuda.is_available()
        q_resident = {}
        for key in s['case_keys']:
            guard(); rec = s['records'][key]; n, t = rec['nodes'], rec['cells']
            a['phase'] = 'reference_tensor'; a['current_case'] = key; persist()
            cache = rec['input_cache']; assert d.sha(cache['path']) == cache['sha256']
            a['read_receipt'].append(dict(kind='input_cache_four_members', path=cache['path'], attempted=True, completed=False)); persist()
            with np.load(d.path(cache['path']), allow_pickle=False) as src:
                values = {name: np.array(src[name], copy=True) for name in ('vertices_mm', 'cells10', 'grid_node_ids', 'lame')}
            for name, v in values.items():
                spec = rec['input_schema'][name]
                assert list(v.shape) == spec['shape'] and str(v.dtype) == spec['dtype'] and sha_values(np, v) == spec['sha256']
            a['read_receipt'][-1]['completed'] = True
            archive = rec['GT_U_archive']; a['GT_U_archives_attempted'] += 1; persist()
            with np.load(d.path(archive['path']), allow_pickle=False) as src:
                assert str(src['sample_key'].item()) == key
                assert str(src['label_alignment'].item()) == 'official CSV GRID coordinates -> H5 vertices'
                ids = np.array(src['grid_node_ids'], dtype=np.int64)
                u = np.array(src['displacement'], dtype=np.float64)
            assert np.array_equal(ids, values['grid_node_ids']) and u.shape == (n, 3) and np.isfinite(u).all()
            a['GT_U_archives_completed'] += 1; persist()
            gc = load_npy(np, rec['operator_arrays']['gradients'], 'accepted_fourQ_gradients')
            wc = load_npy(np, rec['operator_arrays']['weights'], 'accepted_fourQ_weights')
            assert gc.shape == (t, 4, 10, 3) and wc.shape == (t, 4) and gc.dtype == wc.dtype == np.float64 and (wc > 0).all()
            X = torch.as_tensor(values['vertices_mm'], device='cuda', dtype=torch.float64)
            cells = torch.as_tensor(values['cells10'], device='cuda', dtype=torch.int64)
            U = torch.as_tensor(u, device='cuda', dtype=torch.float64)
            grad = torch.as_tensor(gc, device='cuda', dtype=torch.float64)
            weights = torch.as_tensor(wc, device='cuda', dtype=torch.float64)
            lam, mu = map(float, values['lame']); assert mu > 0 and 3*lam+2*mu > 0
            assert int(cells.min()) >= 0 and int(cells.max()) < n
            nodal = torch.zeros((n, 6), device='cuda', dtype=torch.float64)
            mass = torch.zeros(n, device='cuda', dtype=torch.float64); eye = torch.eye(3, device='cuda', dtype=torch.float64)
            partition_max = affine_max = 0.
            for start in range(0, t, 8192):
                guard(); c = cells[start:start+8192]; g = grad[start:start+8192]; w = weights[start:start+8192]
                scale = torch.linalg.vector_norm(g.flatten(2), dim=-1); assert (scale > 0).all()
                partition_max = max(partition_max, float((torch.linalg.vector_norm(g.sum(2), dim=-1)/scale).max()))
                affine = torch.einsum('cai,cqaj->cqij', X[c], g)
                affine_max = max(affine_max, float(torch.linalg.matrix_norm(affine-eye, ord='fro').max()/math.sqrt(3)))
                G = torch.einsum('cai,cqaj->cqij', U[c], g); strain = .5*(G+G.transpose(-1, -2))
                sigma = 2*mu*strain+lam*strain.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]*eye
                assert torch.isfinite(sigma).all()
                six = torch.stack((sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2], sigma[..., 0, 1], sigma[..., 0, 2], sigma[..., 1, 2]), dim=-1)
                volume = w.sum(1); mean = (six*w[..., None]).sum(1)/volume[:, None]
                incident = (volume/10)[:, None].expand(-1, 10)
                mass.index_add_(0, c.flatten(), incident.flatten())
                nodal.index_add_(0, c.flatten(), (mean[:, None, :]*incident[..., None]).reshape(-1, 6))
            assert partition_max <= 1e-9 and affine_max <= 1e-9 and (mass > 0).all()
            nodal /= mass[:, None]
            xx, yy, zz, xy, xz, yz = nodal.unbind(1)
            q = torch.sqrt(.5*((xx-yy)**2+(yy-zz)**2+(zz-xx)**2)+3*(xy**2+xz**2+yz**2))
            assert torch.isfinite(q).all(); q_cpu = q.cpu().numpy(); guard()
            dest = out/(key.replace('/', '_')+'_reference_q_MPa.npy')
            assert not dest.exists(); np.save(dest, q_cpu, allow_pickle=False)
            a['reference_q_arrays'][key] = dict(**d.identity(dest, hashed=True), shape=[n], dtype='float64', values_sha256=sha_values(np, q_cpu),
                nodes=n, cells=t, quadrature_queries=4*t, partition_relative_error=partition_max, affine_relative_error=affine_max,
                source_U_values_sha256=sha_values(np, u), full_node_coverage=True, derived_from_GT_U=True, deployable_prediction=False)
            q_resident[key] = q_cpu
            del X, cells, U, grad, weights, nodal, mass, gc, wc, values, u, ids, q
            torch.cuda.empty_cache(); persist()
        assert set(a['reference_q_arrays']) == set(s['case_keys']) and len(q_resident) == 3
        for value in a['reference_q_arrays'].values():
            d.stat_matches(value); assert d.sha(value['path']) == value['sha256']
        a['all_q_committed_before_S'] = True; a['phase'] = 'magnitude_readout'; persist()
        for key in s['case_keys']:
            guard(); a['cached_S_attempted'] += 1; persist()
            S = load_npy(np, s['records'][key]['signed_stress'], 'cached_original_signed_S')
            a['cached_S_completed'] += 1
            truth = np.abs(S); q = q_resident[key]; assert truth.shape == q.shape
            energy = float(np.sum(truth**2)); assert energy > 0
            order = np.argsort(-truth, kind='stable')[:max(1, math.ceil(len(truth)*.01))]
            hot_energy = float(np.sum(truth[order]**2)); assert hot_energy > 0
            residual = q-truth
            new = dict(relativeL2=float(np.linalg.norm(residual)/math.sqrt(energy)),
                hotspot_relativeL2=float(np.linalg.norm(residual[order])/math.sqrt(hot_energy)),
                RMSE_MPa=float(np.sqrt(np.mean(residual**2))), MAE_MPa=float(np.mean(np.abs(residual))),
                max_abs_error_MPa=float(np.max(np.abs(residual))), original_nodes=len(q), all_finite=True,
                magnitude_only=True, derived_from_GT_U=True, source_sign_predicted=False)
            old = s['records'][key]['old_physical_q_scalar']; new['old_physical_q_relativeL2'] = old['relativeL2']
            new['old_physical_q_hotspot_relativeL2'] = old['hotspot_relativeL2']
            new['relativeL2_change_vs_old_physical_U'] = new['relativeL2']-old['relativeL2']
            a['case_metrics'][key] = new; persist()
        rows = list(a['case_metrics'].values()); assert len(rows) == 3
        a['mean_relativeL2'] = sum(v['relativeL2'] for v in rows)/3
        a['mean_hotspot_relativeL2'] = sum(v['hotspot_relativeL2'] for v in rows)/3
        a['GT_U_removes_magnitude_mismatch_gate'] = all(v['relativeL2'] <= .05 and v['hotspot_relativeL2'] <= .1 for v in rows)
        guard(); torch.cuda.synchronize()
        a.update(status='completed', active=False, closed=True, phase='terminal', budget_true=True,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as err:
        a.update(status='technical_failure', active=False, closed=True, phase='terminal', budget_true=False,
                 error_type=type(err).__name__, error_message=str(err), automatic_replay=False)
        raise
    finally:
        persist(); d.write(out/'summary.json', a)
        if lock is not None: lock.close()
    print(json.dumps(dict(status=a['status'], mean_relativeL2=a['mean_relativeL2'], gate=a['GT_U_removes_magnitude_mismatch_gate'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--start', action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start:
        if not args.plan: parser.error('--start requires frozen --plan')
        run(args.plan)
    else: print('{"status":"source_only","arrays_GT_GPU":0}')
