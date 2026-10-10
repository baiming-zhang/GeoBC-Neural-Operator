"""CPU synthetic check of fresh spatial stencils; no model or dataset access."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check:
        print('Synthetic check only; use --check for CPU execution.')
        return
    began = time.perf_counter()
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from geobc.solver_selection_data import _stencil
    from geobc.fno import query_grid
    assert not torch.cuda.is_initialized()
    # Coordinates are normalized. This is an analytic affine field, not FEM GT.
    axis = torch.linspace(0, 1, 64, device='cpu', dtype=torch.float32)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    grid = (2*x - 3*y + .5*z)[None]
    q = torch.tensor([[.13, .41, .79], [.61, .27, .33], [.25, .5, .75]],
                     device='cpu', dtype=torch.float32, requires_grad=True)
    indices, weights = _stencil(q, 64)
    values = query_grid(grid, indices, weights)
    expected = 2*q[:, 0] - 3*q[:, 1] + .5*q[:, 2]
    value_error = float((values[:, 0] - expected).abs().max())
    gradients, = torch.autograd.grad(values.sum(), q)
    gradient_error = float((gradients - torch.tensor([2., -3., .5])).abs().max())
    boundary = torch.tensor([[0., 0., 0.], [1., 1., 1.], [0., 1., .5]], device='cpu')
    bi, bw = _stencil(boundary, 64)
    bv = query_grid(grid, bi, bw)[:, 0]
    be = 2*boundary[:, 0] - 3*boundary[:, 1] + .5*boundary[:, 2]
    boundary_error = float((bv-be).abs().max())
    cached = query_grid(grid, indices, weights.detach())
    # Cached weights detach coordinate dependence; gradients to grid values
    # remain possible when the grid requires grad. This alone is not dfield/dx.
    cached_coordinate_dependence = cached.requires_grad
    assert value_error < 2e-6 and boundary_error < 2e-6
    assert gradient_error < 5e-5 and not cached_coordinate_dependence
    assert not torch.cuda.is_initialized()
    sources = ['tools/check_fno_query_stencil_cpu.py', 'geobc/fno.py',
               'geobc/solver_selection_data.py', 'geobc/coverage_comparison_data.py']
    record = {
        'kind': 'synthetic_CPU_spatial_stencil_feasibility', 'status': 'passed',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds_including_imports': time.perf_counter()-began,
        'source_sha256': {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
        'device': 'cpu', 'CUDA_initialized': False, 'dataset_reads': 0,
        'PT_reads': 0, 'GT_reads': 0, 'model_constructions': 0, 'model_calls': 0,
        'resolution': 64, 'interior_queries': 3, 'boundary_value_queries': 3,
        'affine_value_max_abs_error': value_error,
        'normalized_coordinate_gradient_max_abs_error': gradient_error,
        'boundary_value_max_abs_error': boundary_error,
        'fresh_stencil_coordinate_autograd': True,
        'detached_cache_coordinate_autograd': cached_coordinate_dependence,
        'current_new_suite_cache': 'Bare partition-of-unity trilinear _stencil; no occupancy renormalization. Historical masked prepare_grids caches are distinct.',
        'limits': ['Synthetic affine interpolation only, not an actual trained FNO qualification.',
                   'No original-node replay, physical mm gradient, new CAD, BC generalization, or signed-S accuracy tested.',
                   'Native25 must have an explicit off-node extension; raw node slices are insufficient.',
                   'Grid-cell derivative jumps are expected; C0 does not imply C1.',
                   'Current cached coordinate weights cannot supply latent spatial derivatives without recomputation.'],
        'live_training_source_or_protocol_changed': False,
    }
    output = root/'notes/geobc_continuous_fno_query_synthetic_feasibility.json'
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['status', 'elapsed_seconds_including_imports',
                     'affine_value_max_abs_error', 'normalized_coordinate_gradient_max_abs_error',
                     'CUDA_initialized', 'model_calls', 'GT_reads']}))


if __name__ == '__main__':
    main()
