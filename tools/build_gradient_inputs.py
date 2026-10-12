"""Recover U and nine physical-gradient inputs from saved geometry and fields."""
import argparse
from pathlib import Path
import json
import numpy as np
from geobc.io import load_geometry, load_array, write_json
from geobc.kinematics import recover_kinematic_inputs
from geobc.runtime import Budget, GPUQueue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('catalog', help='JSON list of CAD, geometry, weights and displacement file paths')
    parser.add_argument('--output', default='data/physical_inputs')
    parser.add_argument('--seconds', type=float, default=900)
    args = parser.parse_args()
    records = json.loads(Path(args.catalog).read_text(encoding='utf-8'))
    guard = Budget(args.seconds)
    with GPUQueue():
        import torch
        for record in records:
            guard()
            output = Path(args.output)/record['CAD']
            if (output/'metrics.json').exists():
                continue
            vertices, cells = load_geometry(record['geometry'])
            U = load_array(record['displacement'])
            weights = load_array(record['weights'])
            inputs, metrics = recover_kinematic_inputs(torch, vertices, cells, U, weights.sum(1), guard)
            output.mkdir(parents=True, exist_ok=True)
            np.save(output/'kinematic_inputs.npy', inputs.cpu().numpy(), allow_pickle=False)
            write_json(output/'metrics.json', metrics)
            del vertices, cells, U, weights, inputs
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
