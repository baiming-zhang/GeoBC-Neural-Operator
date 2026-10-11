"""Future R16 operator control, not a selected solver or released experiment."""
import torch
from torch import nn
from .coarse_transolver_v19 import CoarseTransolver, pool_inputs


class CoarseFNOControl(nn.Module):
    def __init__(self, seed=2026, device='cuda'):
        super().__init__()
        from .conditioned_field import PointNetGeometryEncoder
        from .fno import FNO3d
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = PointNetGeometryEncoder(32, 64)
            self.head = nn.Sequential(nn.Linear(76, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 4))
            # Fixed near-PP capacity control, not a width/modes sweep. All16
            # spatial cells remain present; modes8 avoids overlapping FFT bands.
            self.operator = FNO3d(61, 16, width=16, modes=8, layers=4, head_width=128, checkpoint_blocks=True)
        self.method = 'FNO16_control'; self.macro_family = 'R16_FNO16_control'; self.to(device)

    def encode_case(self, case):
        context = self.encoder(case['cloud'], case['mask']); grid = pool_inputs(case['grid'])
        inputs = torch.cat((grid[:, 1:4], grid[:, :1], grid[:, 4:],
                            context[:, :, None, None, None].expand(1, 32, 16, 16, 16)), dim=1)
        assert inputs.shape == (1, 61, 16, 16, 16)
        latent = self.operator(inputs); assert latent.shape == (1, 16, 16, 16, 16)
        return context, latent

    query = CoarseTransolver.query

    def profile(self):
        counts = {k: sum(p.numel() for p in getattr(self, k).parameters()) for k in ('encoder', 'head', 'operator')}
        return dict(method=self.method, family=self.macro_family, parameter_count=sum(counts.values()), counts=counts,
            architecture=self.operator.config.record(), global_cells=4096, grid_resolution=16,
            input_policy='exact same pooled R16/XYZ-first61 tensor as compact TS before token flattening',
            query_policy='unchanged compact TS native25/head/block-center trilinear query',
            precision='highest FP32; actual experiment must disable TF32', exact_capacity_matched=False,
            capacity_control='fixed width16/modes8; compare measured parameter ratio to compactPP, no tuning sweep',
            owns_training_or_data=False, independent_offnode_FEM_accuracy_measured=False)
