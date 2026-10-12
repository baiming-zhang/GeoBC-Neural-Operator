"""Point-cloud geometry encoder."""

import torch
from torch import Tensor, nn

class PointNetGeometryEncoder(nn.Module):
    """Permutation-invariant coordinates -> bounded global context, [B,D]."""

    def __init__(self, descriptor_dim: int=32, point_width: int=64) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, point_width), nn.ReLU(), nn.Linear(point_width, point_width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * point_width, point_width), nn.ReLU(), nn.Linear(point_width, descriptor_dim), nn.LayerNorm(descriptor_dim, elementwise_affine=False), nn.Tanh())

    def forward(self, cloud: Tensor, mask: Tensor | None=None) -> Tensor:
        if cloud.ndim != 3 or cloud.shape[-1] != 3 or min(cloud.shape[:2]) < 1:
            raise ValueError('geometry_cloud must have shape [B,P,3] with nonempty B/P')
        if mask is None:
            mask = torch.ones(cloud.shape[:2], dtype=torch.bool, device=cloud.device)
        if mask.dtype != torch.bool or mask.shape != cloud.shape[:2] or mask.device != cloud.device:
            raise ValueError('geometry_mask must be bool [B,P] on the cloud device')
        counts = mask.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError('Every CAD cloud needs at least one valid geometry point')
        values = self.point_mlp(cloud.float())
        expanded = mask.unsqueeze(-1)
        maximum = values.masked_fill(~expanded, -torch.inf).amax(dim=1)
        average = (values * expanded).sum(dim=1) / counts.to(values.dtype)
        return self.head(torch.cat((maximum, average), dim=-1))
