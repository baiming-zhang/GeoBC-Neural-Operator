"""CPU synthetic check of the actual common-grid interpolation functions.

No dataset, checkpoint, labels or CUDA initialization. This is a numerical
continuity check, not off-node FEM accuracy or a trained model qualification.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def main():
    began = time.perf_counter()
    import torch
    from geobc.solver_selection_data import _stencil
    from geobc.fno import query_grid

    root = Path(__file__).resolve().parents[1]
    output = root / 'notes/geobc_synthetic_query_continuity.json'
    if output.exists():
        raise RuntimeError('Existing receipt: do not overwrite or repeat without a reason')
    assert not torch.cuda.is_initialized()
    axis = torch.linspace(0, 1, 4, device='cpu')
    x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
    # Fixed scalar lattice, with differing slopes across cell faces.
    lattice = (x.square() + 2*y.square() - 3*z.square() + x*y).unsqueeze(0)

    def query(points):
        indices, weights = _stencil(points, 4)
        return query_grid(lattice, indices, weights, 8), weights

    rows = []
    for dimension in range(3):
        for plane in (1/3, 2/3):
            for epsilon in (1e-3, 1e-4, 1e-5):
                points = torch.tensor([[.17, .23, .61]]*3, dtype=torch.float32)
                points[:, dimension] = torch.tensor([plane-epsilon, plane, plane+epsilon])
                points.requires_grad_(True)
                values, weights = query(points)
                gradients = torch.autograd.grad(values.sum(), points)[0]
                jump = float((values[2]-values[0]).abs().max())
                # Linear within either adjoining cell. This bound checks values,
                # without incorrectly requiring equal derivatives at the face.
                slope_bound = float(gradients[:, dimension].abs().max())
                passed = jump <= 2*epsilon*slope_bound + 2e-6
                rows.append({'axis': dimension, 'face': plane, 'epsilon': epsilon,
                             'left_value': float(values[0, 0]),
                             'face_value': float(values[1, 0]),
                             'right_value': float(values[2, 0]),
                             'left_right_value_difference': jump,
                             'left_derivative': float(gradients[0, dimension]),
                             'right_derivative': float(gradients[2, dimension]),
                             'partition_unity_max_error': float((weights.sum(1)-1).abs().max()),
                             'value_continuity_bound_passed': passed})
                assert passed
    corner_values, corner_weights = query(torch.tensor([[0.,0.,0.], [1.,1.,1.]]))
    assert torch.allclose(corner_weights.sum(1), torch.ones(2), atol=2e-6, rtol=0)
    assert not torch.cuda.is_initialized()
    source_names = ['geobc/solver_selection_data.py', 'geobc/fno.py',
                    'geobc/common_grid_solver.py', 'geobc/coverage_comparison_data.py',
                    'geobc/available_cad_pilot_data.py', 'geobc/fno_arbitrary_query.py',
                    'tools/check_geobc_query_continuity.py']
    result = {
        'status': 'completed', 'observed_UTC': datetime.now(timezone.utc).isoformat(),
        'actual_seconds': time.perf_counter()-began, 'device': 'cpu',
        'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in source_names},
        'actual_source_trace': [
            'solver_selection_data._stencil forms eight unmasked trilinear weights summing to one',
            'coverage_comparison_data._prepare_CAD uses this same _stencil',
            'available_cad_pilot_data calls original coverage _prepare_CAD',
            'common_grid_solver.query passes cached weights unchanged to fno.query_grid',
            'fno_arbitrary_query.query_fixed_case recomputes the same unmasked _stencil',
            'fno.query_grid multiplies supplied weights; its historical masked-cache docstring does not describe the current caller weights'],
        'checks': rows, 'all_value_checks_passed': all(r['value_continuity_bound_passed'] for r in rows),
        'corner_values': corner_values.flatten().tolist(),
        'interpretation': 'Ordinary fixed-grid trilinear latent interpolation has matching face values (C0), but generally unequal one-sided derivatives (not globally C1). No occupancy-induced zero-weight discontinuity was found in the inspected current caller construction.',
        'limitations': [
            'Synthetic fixed lattice only; no dataset arrays or trained model were inspected',
            'No claim of trained field accuracy, whole-head continuity, global physical derivatives or cross-CAD generalization',
            'Within-cell AD/finite-difference agreement cannot establish global C1 continuity',
            'Current live model, loss, cache, plans and scientific identities are unchanged'],
        'new_GT_PT_dataset_array_reads': 0, 'GPU_work': 0,
        'CUDA_initialized': torch.cuda.is_initialized()}
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status','actual_seconds','all_value_checks_passed','CUDA_initialized')}))


if __name__ == '__main__':
    main()
