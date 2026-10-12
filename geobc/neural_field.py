"""Geometry-conditioned FNO with a pointwise U/S decoder."""
import torch
from torch import nn
from .geometry_encoder import PointNetGeometryEncoder
from .fno import FNO3d, query_grid

class FNOField(nn.Module):
    def __init__(self, seed=2026):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = PointNetGeometryEncoder(32, 64)
            self.head = nn.Sequential(nn.Linear(76, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 4))
            self.operator = FNO3d(61, 16, width=9, modes=12, layers=4, head_width=128, checkpoint_blocks=True)

    def encode_case(self, case):
        context = self.encoder(case['cloud'], case['mask'])
        grid = torch.cat((case['grid'], context[:, :, None, None, None].expand(1, 32, 64, 64, 64)), dim=1)
        return context, self.operator(grid)

    def query(self, case, start, stop, context, latent):
        values = query_grid(latent[0], case['query_indices'][start:stop], case['query_weights'][start:stop], 32768)
        inputs = torch.cat((case['xyz'][start:stop], case['features'][start:stop], context.expand(stop-start, -1), values), dim=1)
        return self.head(inputs)
