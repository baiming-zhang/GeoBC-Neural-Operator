"""Prospective occupancy-aware compact Transolver; no data/launch ownership.

Exclude empty R16 cells from global attention. Scatter their latent values as
zero and keep the existing continuous query head. These two coupled changes
are explicit; no claim that empty tokens caused the earlier negative result.
"""
import torch
from .coarse_transolver_v19 import CoarseTransolver, pool_inputs


class OccupiedTransolver(CoarseTransolver):
    def __init__(self, seed=2026, device='cuda'):
        super().__init__('transolver', seed, device)
        self.macro_family = 'R16_occupied_transolver'

    def encode_case(self, case):
        context = self.encoder(case['cloud'], case['mask'])
        grid = pool_inputs(case['grid'])
        occupied = grid[:, :1].flatten(2)[0, 0] > 0
        indices = torch.nonzero(occupied, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError('At least one occupied input cell required')
        tokens = torch.cat((grid[:, 1:4], grid[:, :1], grid[:, 4:],
                            context[:, :, None, None, None].expand(1, 32, 16, 16, 16)), dim=1)
        tokens = tokens.flatten(2).transpose(1, 2).contiguous()
        selected = self.operator(tokens.index_select(1, indices))
        if selected.shape != (1, len(indices), 16):
            raise RuntimeError('Official operator must return all selected token features16')
        latent = selected.new_zeros((1, 4096, 16)).index_copy(1, indices, selected)
        return context, latent.transpose(1, 2).reshape(1, 16, 16, 16, 16)

    def profile(self):
        result = super().profile()
        result.update(method='occupied_transolver', family=self.macro_family,
                      global_tokens='all occupied R16 cells, input-only selection; at most4096',
                      empty_latent_extension='exact zero; same old center-trilinear query',
                      additional_parameters=0, owns_training_or_data=False)
        return result
