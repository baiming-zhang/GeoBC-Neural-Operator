"""Analytic CPU fixtures for query routing and full physical chain rule."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    if not parser.parse_args().check:
        print('Use --check for synthetic CPU checks; no dataset/model access.')
        return
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from geobc.fno_arbitrary_query import (ARMS, tet10_queries, query_fixed_case,
                                          physical_query_gradients)
    from geobc.fno_local_condition_pair import model_class
    query_method = model_class(torch).query
    assert not torch.cuda.is_initialized()
    # Exercise the actual FNOQueryField.query method with an analytic fixture
    # head and grids. No learned model is created or checkpoint loaded.
    class Fixture:
        query_calls = 0
        def __init__(self, arm):
            self.arm = arm
            self.head = lambda t: torch.cat((t[:, :3], t[:, 3:4]+t[:, 60:61]), dim=1)
        def query(self, *args):
            Fixture.query_calls += 1
            return query_method(self, *args)
        def encode_case(self, *args):
            raise AssertionError('Queries must not re-encode the reference case')
    nodes = torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.],
                          [.5,0.,0.], [.5,.5,0.], [0.,.5,0.], [0.,0.,.5],
                          [.5,0.,.5], [0.,.5,.5]], device='cpu')
    transform = torch.tensor([[2.,.3,0.], [.2,3.,.1], [0.,.1,4.]], dtype=torch.float64)
    physical_nodes = nodes.double() @ transform.T + torch.tensor([10.,20.,30.])
    features = torch.zeros(10, 25)
    features[:, 0] = 4*nodes[:, 0]+.5*nodes[:, 1]-nodes[:, 2]
    axis = torch.linspace(0, 1, 64)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    grid = torch.zeros(1, 29, 64, 64, 64)
    grid[0, 4] = 4*x+.5*y-z
    latent = torch.zeros(1, 16, 64, 64, 64)
    latent[0, 0] = 2*x-3*y+.5*z
    context = torch.zeros(1, 32)
    case = {'grid': grid, 'fixture_identity': object()}
    errors = {}
    for arm in ARMS:
        r = torch.tensor([[.13,.21,.29], [.61,.12,.17], [.25,.25,.25]], requires_grad=True)
        k = len(r)
        q = tet10_queries(physical_nodes.expand(k,-1,-1), nodes.expand(k,-1,-1),
                         features.expand(k,-1,-1), r)
        values = query_fixed_case(Fixture(arm), case, q['xyz'], context, latent,
                                  q['features'], chunk_size=2)
        whole = query_fixed_case(Fixture(arm), case, q['xyz'], context, latent,
                                 q['features'], chunk_size=100)
        expected = torch.cat((r, (6*r[:,0]-2.5*r[:,1]-.5*r[:,2])[:,None]), dim=1)
        gradient = physical_query_gradients(values, r, q['physical_jacobian'])
        dnormal = torch.cat((torch.eye(3, dtype=torch.float64),
                             torch.tensor([[6.,-2.5,-.5]], dtype=torch.float64)), dim=0)
        expected_gradient = dnormal @ torch.linalg.inv(transform)
        errors[arm] = {'value': float((values-expected).detach().abs().max()),
                       'physical_gradient': float((gradient-expected_gradient).detach().abs().max()),
                       'chunk_consistency': float((values-whole).detach().abs().max())}
        assert errors[arm]['value'] < 2e-6
        assert errors[arm]['physical_gradient'] < 5e-5
        assert errors[arm]['chunk_consistency'] < 1e-6
    assert case['grid'] is grid and set(case) == {'grid', 'fixture_identity'}
    assert Fixture.query_calls == 6
    assert not torch.cuda.is_initialized()
    sources = ['geobc/fno_arbitrary_query.py', 'tools/check_fno_arbitrary_query_cpu.py',
               'geobc/fno_local_condition_pair.py', 'geobc/solver_selection_data.py',
               'geobc/fno.py']
    record = {'status': 'passed', 'kind': 'synthetic_CPU_arbitrary_query_adapter',
              'elapsed_seconds_including_imports': time.perf_counter()-began,
              'source_sha256': {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
              'errors': errors, 'device': 'cpu', 'CUDA_initialized': False,
              'learned_model_constructions': 0, 'learned_model_calls': 0,
              'fixture_query_method_calls': Fixture.query_calls, 'geometry_reencodes': 0,
              'GT_reads': 0, 'PT_reads': 0, 'dataset_reads': 0,
              'reference_case_mutated': False,
              'qualification': 'Analytic affine TET10 physical-chain-rule and chunk routing only; actual trained-model/off-node/CAD/BC qualification pending.'}
    (root/'notes/geobc_fno_arbitrary_query_CPU_result.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['status','elapsed_seconds_including_imports',
                     'errors','learned_model_calls','GT_reads','CUDA_initialized']}))


if __name__ == '__main__':
    main()
