#!/usr/bin/env python3
"""Prepare original nodal fields and deck-derived conditioning, without resampling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

# The temporary Windows preprocessing dependency directory is project-local.
ROOT = Path(__file__).resolve().parents[1]
if os.name == 'nt' and (ROOT / '.local_deps').is_dir():
    sys.path.insert(0, str(ROOT / '.local_deps'))
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from scipy.spatial import cKDTree
from tools.parse_deepjeb_deck import parse_deck
from tools.mesh_boundary import quadratic_boundary_triangles, boundary_report

DEFAULT_SELECTION = ['10_30/ver', '10_421/hor', '10_460/dia', '10_492/tor', '10_58/ver']
FEATURE_NAMES = (
    [f'support_reference_constrained_dof_{i}' for i in range(1, 7)]
    + ['loaded_interface_mask']
    + [f'total_force_{axis}_over_42300N' for axis in 'xyz']
    + [f'total_moment_{axis}_over_565000Nmm' for axis in 'xyz']
    + ['E_over_113800MPa', 'poisson_ratio', 'density_over_4.47e-9_t_per_mm3']
    + [f'offset_from_load_reference_{axis}_over_length' for axis in 'xyz']
    + [f'distance_to_support_reference_{i}_over_length' for i in range(1, 5)]
    + ['distance_to_loaded_interface_over_length', 'characteristic_length_over_200mm']
)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def prepare(raw: Path, output: Path, selection: list[str]):
    output.mkdir(parents=True, exist_ok=True)
    report = {'dataset': 'DeepJEB original v1.0', 'selection': selection,
              'task': 'observed FEM node fitting; not a heldout evaluation',
              'feature_names': FEATURE_NAMES,
              'feature_schema_note': 'New explicit deck-derived schema; the original 25-feature schema is absent from the slides.',
              'normalization': 'xyz=(xyz_mm-min_xyz_mm)/max_bbox_extent_mm; isotropic, preserving aspect ratio',
              'target': 'source signed nodal stress(MPa), unchanged; official CSV coordinates aligned to H5 vertices', 'cases': []}
    assert len(FEATURE_NAMES) == 25
    for item in selection:
        cad, load = item.split('/')
        h5_path = raw / 'FieldMesh' / f'{cad}.h5'
        deck_path = raw / 'SolverDeck' / f'{cad}.fem'
        csv_path = raw / 'Field' / f'{cad}.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f'{csv_path}: official Field CSV required; H5 nodal fields are not in vertices order')
        deck = parse_deck(deck_path)
        with h5py.File(h5_path, 'r') as field:
            xyz_mm = field['vertices'][:].astype(np.float32)
            source_faces = field['faces'][:].astype(np.int32)
            cells = field['cells'][:].astype(np.int64)
        with csv_path.open(encoding='utf-8-sig',newline='') as handle:
            columns = next(csv.reader(handle))
        names = ['coord_x(mm)','coord_y(mm)','coord_z(mm)',f'{load}_stress(MPa)']
        values = np.loadtxt(csv_path,delimiter=',',skiprows=1,usecols=[columns.index(name) for name in names])
        distances, field_indices = cKDTree(values[:,:3]).query(xyz_mm,workers=-1)
        if len(values) != len(xyz_mm) or len(np.unique(field_indices)) != len(xyz_mm) or distances.max() > 2e-4:
            raise ValueError(f'{item}: CSV fields do not match the geometry uniquely')
        stress = values[field_indices,3].astype(np.float32)
        faces = quadratic_boundary_triangles(cells, xyz_mm)
        mesh_report = boundary_report(xyz_mm, faces, cells)
        if len(faces) != len(source_faces):
            raise ValueError(f'{item}: reconstructed boundary count differs from source surface')
        if not (np.isfinite(xyz_mm).all() and np.isfinite(stress).all()):
            raise ValueError(f'{item}: nonfinite source values')
        if stress.shape != (len(xyz_mm),) or faces.min() < 0 or faces.max() >= len(xyz_mm):
            raise ValueError(f'{item}: invalid nodal field or source face indexing')
        lower, upper = xyz_mm.min(0), xyz_mm.max(0)
        length = float((upper - lower).max())
        xyz = ((xyz_mm - lower) / length).astype(np.float32)
        tree = cKDTree(xyz_mm)
        node_coordinates = deck['node_coordinates']
        def map_ids(ids):
            query = np.asarray([node_coordinates[int(node)] for node in ids], dtype=np.float64)
            distances, indices = tree.query(query, workers=-1)
            if np.max(distances, initial=0) > 2e-4:
                raise ValueError(f'{item}: deck/field coordinate mismatch {distances.max()} mm')
            if len(np.unique(indices)) != len(ids):
                raise ValueError(f'{item}: ambiguous coincident interface nodes')
            return indices, distances
        features = np.zeros((len(xyz), 25), dtype=np.float32)
        fixed_indices, fixed_distances = [], []
        supports = sorted(deck['fixed_groups'], key=lambda group: group['master_node_id'])
        if len(supports) != 4:
            raise ValueError(f'{item}: expected four support interfaces')
        for group in supports:
            indices, distances = map_ids(group['member_node_ids'])
            fixed_indices.extend(indices.tolist())
            fixed_distances.extend(distances.tolist())
            for dof in group['dofs']:
                features[indices, int(dof) - 1] = 1
        loaded = deck['loaded_group']
        loaded_indices, loaded_distances = map_ids(loaded['member_node_ids'])
        features[loaded_indices, 6] = 1
        condition = deck['cases'][load]
        features[:, 7:10] = np.asarray(condition['force']) / 42300
        features[:, 10:13] = np.asarray(condition['moment']) / 565000
        material = deck['material']
        features[:, 13:16] = [material['E'] / 113800, material['nu'], material['rho'] / 4.47e-9]
        features[:, 16:19] = (xyz_mm - np.asarray(loaded['master_coords'])) / length
        for i, group in enumerate(supports):
            features[:, 19+i] = np.linalg.norm(xyz_mm - np.asarray(group['master_coords']), axis=1) / length
        features[:, 23] = cKDTree(xyz_mm[loaded_indices]).query(xyz_mm, workers=-1)[0] / length
        features[:, 24] = length / 200
        target = output / f'{cad}_{load}.npz'
        np.savez(target, xyz=xyz, xyz_mm=xyz_mm, features=features, stress=stress,
                 faces=faces, sample_key=np.asarray(item), bbox_min_mm=lower,
                 characteristic_length_mm=np.asarray(length), feature_names=np.asarray(FEATURE_NAMES),
                 label_alignment=np.asarray('official CSV GRID coordinates -> H5 vertices'))
        info = {'sample_key': item, 'npz': str(target.resolve()), 'nodes': len(xyz), 'surface_triangles': len(faces),
                'signed_stress_min_MPa': float(stress.min()), 'signed_stress_max_MPa': float(stress.max()),
                'signed_stress_RMS_MPa': float(np.sqrt(np.mean(stress.astype(np.float64)**2))),
                'support_interface_nodes': len(set(fixed_indices)), 'loaded_interface_nodes': len(loaded_indices),
                'deck_field_max_coordinate_error_mm': max(max(fixed_distances), float(loaded_distances.max())),
                'force_N': condition['force'], 'moment_Nmm': condition['moment'], 'material': material,
                'bbox_min_mm': lower.tolist(), 'bbox_max_mm': upper.tolist(),
                'source_h5_sha256': sha256(h5_path), 'source_csv_sha256': sha256(csv_path), 'source_deck_sha256': sha256(deck_path),
                'label_alignment': 'official CSV GRID coordinates -> H5 vertices',
                'boundary_mesh': mesh_report,
                'surface_indexing': 'global boundary triangles reconstructed from source TETRA10 cells; H5 faces use surface-local numbering'}
        report['cases'].append(info)
        print(json.dumps(info), flush=True)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/prepared_stress_aligned')
    parser.add_argument('--cases', nargs='+', default=DEFAULT_SELECTION)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.cases)
