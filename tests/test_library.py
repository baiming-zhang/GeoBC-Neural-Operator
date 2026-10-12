import importlib
import unittest
from pathlib import Path
import torch
from geobc.pcg import solve
from geobc.neural_field import FNOField
from geobc.kinematic_neural_field import KinematicCovariateFNO
from geobc.scaler import FieldScaler
import json
import numpy as np

class LibraryTests(unittest.TestCase):
    def test_signed_output_roundtrip(self):
        root = Path(__file__).resolve().parents[1]
        scaler = FieldScaler(**json.loads((root/'configs/output_scaler.json').read_text()))
        fields = np.array([[.1, -.2, .3, -100.], [-.1, .2, -.3, 200.]])
        np.testing.assert_allclose(scaler.decode(scaler.encode(fields)), fields, rtol=1e-6, atol=1e-6)

    def test_module_imports(self):
        for path in (Path(__file__).resolve().parents[1]/'geobc').glob('*.py'):
            importlib.import_module('geobc.'+path.stem)

    def test_independent_pcg_lanes(self):
        for lanes in (2, 3):
            diagonal = torch.tensor([[1., 2., 4.], [3., 5., 7.]], dtype=torch.float64)
            rhs = torch.arange(1, lanes*6+1, dtype=torch.float64).reshape(lanes, 2, 3)
            result, metrics = solve(lambda x: x*diagonal, rhs, lambda x: x/diagonal)
            torch.testing.assert_close(result, rhs/diagonal)
            self.assertTrue(metrics['converged'])

    def test_model_chunk_queries(self):
        model = FNOField()
        latent = torch.randn(1, 16, 2, 2, 2)
        context = torch.randn(1, 32)
        case = {'xyz': torch.randn(7, 3), 'features': torch.randn(7, 25),
                'query_indices': torch.arange(8).expand(7, 8), 'query_weights': torch.full((7, 8), .125),
                'kinematic_inputs': torch.zeros(7, 12)}
        direct = model.query(case, 0, 7, context, latent)
        chunked = torch.cat([model.query(case, 0, 3, context, latent), model.query(case, 3, 7, context, latent)])
        torch.testing.assert_close(direct, chunked)
        zero, actual = KinematicCovariateFNO('ZERO'), KinematicCovariateFNO('ACTUAL')
        for key, value in zero.state_dict().items():
            torch.testing.assert_close(value, actual.state_dict()[key])
        torch.testing.assert_close(zero.query(case, 0, 7, context, latent), actual.query(case, 0, 7, context, latent))

if __name__ == '__main__':
    unittest.main()
