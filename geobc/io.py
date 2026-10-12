"""Explicit local geometry and field loading."""
from pathlib import Path
import json
import numpy as np

def load_geometry(path, device='cuda'):
    import torch
    with np.load(path, allow_pickle=False) as archive:
        vertices = archive['vertices_mm']
        cells = archive['cells10']
    if vertices.ndim != 2 or vertices.shape[1] != 3 or cells.ndim != 2 or cells.shape[1] != 10:
        raise ValueError('Expected vertices [N,3] and TET10 cells [C,10]')
    return torch.as_tensor(vertices, dtype=torch.float64, device=device), torch.as_tensor(cells, dtype=torch.int64, device=device)

def load_array(path, device='cuda'):
    import torch
    value = np.load(path, allow_pickle=False)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite array')
    return torch.as_tensor(value, device=device)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
