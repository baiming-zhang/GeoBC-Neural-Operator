"""One prospective large-capacity R64 control; fixed width16/modes12."""
import torch
from torch import nn
from .common_grid_solver import CommonGridSolver


class WideFNO64(nn.Module):
    def __init__(self, seed=2026, device='cuda'):
        super().__init__()
        from .conditioned_field import PointNetGeometryEncoder
        from .fno import FNO3d
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = PointNetGeometryEncoder(32, 64)
            self.head = nn.Sequential(nn.Linear(76, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 4))
            self.operator = FNO3d(61, 16, width=16, modes=12, layers=4, head_width=128, checkpoint_blocks=True)
        # Existing forward dispatch uses this FNO implementation tag, not a
        # scientific family name. Actual width12/16 fields remain explicit.
        self.arm = 'FNO9'; self.method = 'FNO_R64_W16'; self.macro_family = 'FNO_R64_W16'; self.to(device)

    encode_case = CommonGridSolver.encode_case
    query = CommonGridSolver.query

    def profile(self):
        counts = {k: sum(p.numel() for p in getattr(self, k).parameters()) for k in ('encoder', 'head', 'operator')}
        return dict(method=self.method, family=self.macro_family, parameter_count=sum(counts.values()), counts=counts,
                    architecture=self.operator.config.record(), grid_resolution=64, global_cells=262144,
                    implementation_dispatch_arm='FNO9 selects the existing FNO forward; actual width16 is in architecture',
                    native25_and_head_unchanged=True, query='existing occupied R64 original-node stencil',
                    precision='highest FP32; experiment must disable TF32', capacity_matched=False,
                    owns_training_or_data=False, independent_offnode_FEM_accuracy_measured=False)
