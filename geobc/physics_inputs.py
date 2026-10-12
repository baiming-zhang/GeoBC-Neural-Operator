"""Selective local predictor-cache reads and sparse operator restoration."""
from pathlib import Path
import json
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
MEMBERS = ['vertices_mm', 'cells10', 'free_node_ids', 'fixed_node_ids', 'force', 'lame',
           'derived_node_ids', 'derived_source_ids', 'derived_rowptr', 'derived_blocks',
           'reference_xyz_mm', 'canonical_forces']
INTEGER_MEMBERS = {'cells10', 'free_node_ids', 'fixed_node_ids', 'derived_node_ids', 'derived_source_ids', 'derived_rowptr'}


def local_path(value):
    value = str(value)
    if value.startswith('C:/') and str(PROJECT).startswith('/mnt/c/'):
        return Path('/mnt/c/'+value[3:])
    path = Path(value)
    return path if path.is_absolute() else PROJECT/path


def load_source(path, expected, guard):
    inputs = {}
    with np.load(local_path(path), allow_pickle=False) as archive:
        for name in MEMBERS:
            guard(); value = archive[name]
            dtype = np.int64 if name in INTEGER_MEMBERS else np.float64
            if value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError('Invalid predictor member '+name)
            inputs[name] = torch.as_tensor(value, device='cuda')
    forces = inputs.pop('canonical_forces')
    if inputs['vertices_mm'].shape != (expected['nodes'], 3) or inputs['cells10'].shape != (expected['cells'], 10) or forces.shape != (3, expected['nodes'], 3):
        raise ValueError('Geometry/canonical-load dimensions differ')
    return inputs, forces


def load_array(item, guard):
    guard(); value = np.load(local_path(item['path']), allow_pickle=False)
    dtype = {'torch.float64': 'float64', 'torch.float32': 'float32', 'torch.int64': 'int64'}.get(item['dtype'], item['dtype'])
    if str(value.dtype) != dtype or list(value.shape) != item['shape'] or not np.isfinite(value).all():
        raise ValueError('Cached numeric array schema differs')
    return torch.as_tensor(value, device='cuda')


def load_sparse(item, guard):
    rowptr, columns, values = [load_array(item[key], guard) for key in ['rowptr', 'columns', 'values']]
    if rowptr.dtype != torch.int64 or columns.dtype != torch.int64 or values.dtype != torch.float64:
        raise ValueError('Invalid sparse cache types')
    return torch.sparse_csr_tensor(rowptr, columns, values, size=item['shape'], device='cuda', dtype=torch.float64)


def load_coarse(path, chol, ledger, guard):
    from .coarse import AdditiveCoarse
    record = json.loads(local_path(path).read_text(encoding='utf-8'))
    if record['status'] != 'completed' or not record['algebra_guards']['passed']:
        raise ValueError('Existing coarse cache is not qualified')
    arrays = record['arrays']
    P, PT, Ac = [load_sparse(arrays[key], guard) for key in ['P', 'PT', 'Ac']]
    diagonal = load_array(arrays['diagonal'], guard)
    gamma = record['assembly']['symmetric_scaled_Gershgorin_Gamma']
    return AdditiveCoarse(torch, P, PT, Ac, diagonal, gamma, chol, ledger)
