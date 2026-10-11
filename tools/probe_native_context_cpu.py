"""Explicit CPU-only synthetic contract probe, not an official GPU experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if not args.probe:
        print('Source-only; --probe runs synthetic CPU checks without dataset access.')
        return
    started_epoch, started_mono = time.time(), time.monotonic()
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('CPU120s budget')))
    signal.alarm(120)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    import torch
    from torch import nn
    from geobc.common_grid_solver import CommonGridSolver
    from geobc.native_context_solver import NativeContextSolver, full_native_case_backward
    from geobc.fno_arbitrary_query import EDGES, physical_query_gradients
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(2026)
    root = Path(__file__).resolve().parents[1]
    sources = ['geobc/native_context_solver.py', 'tools/probe_native_context_cpu.py',
               'geobc/common_grid_solver.py', 'geobc/transformers.py',
               'geobc/conditioned_field.py', 'geobc/fno_arbitrary_query.py']
    record = {
        'status': 'running', 'scope': 'Synthetic CPU contract only; actual official model constructors but synthetic forward operator',
        'started_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources},
        'budget_seconds': 120, 'actual_dataset_GT_PT_reads': 0, 'actual_CUDA_operator_forwards': 0,
        'actual_training_updates': 0, 'official_wholeN_resource_gradient_qualified': False,
        'new_training_released': False, 'failure': None}
    try:
        native = NativeContextSolver()
        common = CommonGridSolver('PP256')
        a, b = native.state_dict(), common.state_dict()
        if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
            raise AssertionError('Native/R64 initial parameters differ')
        record['actual_official_initial_parameters_equal'] = True
        record['parameter_counts'] = native.counts()
        record['shared_initial_identity'] = native.shared_identity()
        del common

        class SyntheticGlobalOperator(nn.Module):
            def __init__(self):
                super().__init__()
                self.local = nn.Linear(61, 16)
                self.global_projection = nn.Linear(61, 16)
            def forward(self, tokens):
                return self.local(tokens)+self.global_projection(tokens.mean(1, keepdim=True))
        # The fixture verifies shared-context chain rule only. Its gradients
        # cannot qualify the unchanged official ++ CUDA implementation.
        native.operator = SyntheticGlobalOperator()
        case = {'xyz': torch.rand(23, 3), 'features': torch.randn(23, 25),
                'cloud': torch.rand(1, 1024, 3), 'mask': torch.ones(1, 1024, dtype=torch.bool),
                'target': torch.randn(23, 4)}
        context, latent = native.encode_case(case)
        direct = native.query(case, 0, 23, context, latent)
        loss = (direct-case['target']).square().mean()
        loss.backward()
        expected = {k: v.grad.clone() for k, v in native.named_parameters()}
        gradients = []
        for chunk in (7, 9, 23):
            native.zero_grad(set_to_none=True)
            actual_loss, leafs = full_native_case_backward(native, case, chunk_size=chunk)
            errors = []
            for k, v in native.named_parameters():
                if v.grad is None or not torch.isfinite(v.grad).all():
                    raise AssertionError('Missing/nonfinite synthetic gradient')
                torch.testing.assert_close(v.grad, expected[k], rtol=3e-5, atol=2e-6)
                errors.append(float((v.grad-expected[k]).abs().max()))
            if abs(actual_loss-float(loss.detach())) > 2e-6:
                raise AssertionError('Decoder chunks changed full-node MSE')
            gradients.append({'chunk_size': chunk, 'max_parameter_gradient_absolute_difference': max(errors),
                              'context_latent_gradient_norms': leafs})
        record['synthetic_full_vs_chunked_chain_rule'] = gradients

        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        xyz = torch.cat((vertices, torch.stack([(vertices[i]+vertices[j])/2 for i, j in EDGES])))
        tet_case = dict(case, xyz=xyz, features=torch.randn(10, 25), target=torch.randn(10, 4))
        context, latent = native.encode_case(tet_case)
        cells = torch.arange(10).expand(10, 10).clone()
        nodal, _ = native.query_elements(tet_case, context, latent, cells, xyz.clone(), xyz*100)
        node_direct = native.query(tet_case, 0, 10, context, latent)
        torch.testing.assert_close(nodal, node_direct, rtol=0, atol=0)
        record['TET10_nodal_reproduction_exact'] = True
        r = torch.tensor([[.17, .21, .14]], requires_grad=True)
        one_cell = cells[:1]
        queried, points = native.query_elements(tet_case, context, latent, one_cell, r, xyz*100)
        derivative = physical_query_gradients(queried, r, points['physical_jacobian'])
        finite = []
        for axis in range(3):
            delta = torch.zeros_like(r); delta[:, axis] = .001
            plus, _ = native.query_elements(tet_case, context, latent, one_cell, r.detach()+delta, xyz*100)
            minus, _ = native.query_elements(tet_case, context, latent, one_cell, r.detach()-delta, xyz*100)
            finite.append((plus-minus)/.2)
        fd = torch.stack(finite, -1)
        torch.testing.assert_close(derivative, fd, rtol=.003, atol=1e-6)
        record['synthetic_encoded_field_physical_gradient_AD_FD_max_absolute_difference'] = float((derivative-fd).abs().max())
        rejects = 0
        for invalid_cells, invalid_r in ((one_cell, torch.tensor([[1., 1., 1.]])),
                                         (one_cell+10, r.detach())):
            try:
                native.query_elements(tet_case, context, latent, invalid_cells, invalid_r, xyz*100)
            except ValueError:
                rejects += 1
        if rejects != 2:
            raise AssertionError('Outside-cell/index rejection failed')
        record['invalid_queries_rejected'] = rejects
        record['status'] = 'completed'
        record['CPU_contract_passed'] = True
    except Exception as error:
        record.update(status='failed', CPU_contract_passed=False,
                      failure=type(error).__name__+': '+str(error))
        raise
    finally:
        record['CUDA_initialized'] = torch.cuda.is_initialized()
        record['epoch_seconds'] = time.time()-started_epoch
        record['monotonic_seconds'] = time.monotonic()-started_mono
        record['budget_passed'] = max(record['epoch_seconds'], record['monotonic_seconds']) <= 120
        path = root/'notes/geobc_native_context_CPU_probe.json'
        if path.exists():
            raise RuntimeError('Do not overwrite an existing CPU attempt')
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        signal.alarm(0)
    print(json.dumps({k: record[k] for k in ['status', 'epoch_seconds', 'CUDA_initialized', 'CPU_contract_passed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
