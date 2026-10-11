"""Continuous Hash decoder controlled by a whole-case geometry/BC operator.

This is a candidate composition of existing FNO and tiny-cuda-nn components.
It reuses the common FNO encoder/operator initialization and its complete R64
target-free input. The Hash queries receive both the shared CAD descriptor
and an interpolated operator latent that depends on the whole geometry and
BC/material raster, rather than a shape descriptor alone.

No target, scaler, sample identity, launch or dataset access belongs here.
Original-node accuracy and interface continuity cannot establish independent
off-node FEM accuracy. Native25 off-node extension must be supplied explicitly.
"""
import torch
from .common_grid_solver import CommonGridSolver
from .fno import query_grid
from .tcnn_fields import TCNNField, TCNNFieldConfig


class OperatorConditionedHashSolver(CommonGridSolver):
    def __init__(self, seed=2026, device='cuda', *, injected_decoder=None):
        # Preserve precisely the existing encoder/operator initial RNG history.
        # The MLP is then replaced, not retained as an extra residual prediction.
        super().__init__('FNO9', seed)
        config = TCNNFieldConfig(out_channels=4, input_mode='both',
                                 condition_dim=25 + 32 + 16, seed=seed)
        if injected_decoder is not None:
            if getattr(injected_decoder, 'config', None) != config:
                raise ValueError('Injected CPU decoder must declare the exact config')
            if str(device) != 'cpu':
                raise ValueError('Injection is solely for CPU interface checks')
            self.head = injected_decoder
            self.backend = 'injected_cpu_interface_decoder'
        else:
            self.head = TCNNField(config=config, device=device)
            self.backend = self.head.backend
        self.decoder_config = config
        self.to(device)

    def query(self, case, start, stop, context, latent):
        if not 0 <= start < stop <= len(case['xyz']):
            raise ValueError('Invalid original-node query slice')
        values = query_grid(latent[0], case['query_indices'][start:stop],
                            case['query_weights'][start:stop], 32768)
        return self.decode_points(case['xyz'][start:stop],
                                  case['features'][start:stop], context, values)

    def decode_points(self, xyz, native25, context, values):
        count = len(xyz)
        if (xyz.shape != (count, 3) or not count or
                native25.shape != (count, 25) or context.shape != (1, 32) or
                values.shape != (count, 16)):
            raise ValueError('Fixed-context query dimensions differ')
        if any(t.device != xyz.device or t.dtype != xyz.dtype
               for t in (native25, context, values)):
            raise ValueError('Queries must share dtype/device')
        condition = torch.cat((native25, context.expand(count, -1), values), dim=-1)
        return self.head(xyz, condition)

    def query_points(self, xyz, native25, context, latent, chunk_size=32768):
        """Continuous points with a fixed context and explicit native25 field.

        Trilinear latent weights are recomputed at each point. No context is
        recomputed from the query cloud, and no nearest-node substitution is
        made. At a stencil face the field is C0, not generally C1.
        """
        if (type(chunk_size) is not int or chunk_size < 1 or
                latent.shape != (1, 16, 64, 64, 64) or
                xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or
                native25.shape != (len(xyz), 25)):
            raise ValueError('Explicit continuous query dimensions differ')
        if any(t.device != xyz.device or t.dtype != xyz.dtype
               for t in (native25, context, latent)):
            raise ValueError('Continuous queries must share dtype/device')
        if not all(torch.isfinite(t).all() for t in (xyz, native25, context, latent)):
            raise ValueError('Continuous query inputs must be finite')
        from .solver_selection_data import _stencil
        indices, weights = _stencil(xyz, 64)
        case = dict(xyz=xyz, features=native25, query_indices=indices,
                    query_weights=weights)
        return torch.cat([self.query(case, start, min(start + chunk_size, len(xyz)),
                                    context, latent)
                          for start in range(0, len(xyz), chunk_size)], dim=0)

    @property
    def source_config(self):
        return dict(method='Whole-case FNO-conditioned continuous Hash field',
                    claim='Candidate composition; no novel-algorithm or final-solver claim',
                    global_input='Original target-free R64 occupancy, xyz, native25 and CAD descriptor',
                    operator=self.operator.source_config,
                    decoder=self.decoder_config.record(), counts=self.counts(),
                    backend=self.backend, sample_identity=False,
                    reused_FNO_initial_encoder_operator=True,
                    condition_order=['native25', 'global_geometry32', 'whole_case_operator_latent16'],
                    continuity='Fixed context; trilinear latent and Hash interpolation, explicit native25 extension',
                    precision='FP32 encoder/operator with native TCNN FP16 decoder; mixed precision differs from MLP',
                    offnode_FEM_accuracy_measured=False)
