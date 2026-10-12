"""Matched zero/actual physical-input model for original signed U/S."""

import torch
from torch import nn
from .neural_field import FNOField
from .fno import query_grid
MODES = ("ZERO", "ACTUAL", "U_ONLY", "GRAD_ONLY")

class KinematicCovariateFNO(FNOField):

    def __init__(self, mode, seed=2026):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(seed)
        first = self.head[0]
        assert isinstance(first, nn.Linear) and first.in_features == 76 and (first.out_features == 128)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 125)
            expanded = nn.Linear(88, 128)
        with torch.no_grad():
            expanded.weight[:, :76].copy_(first.weight)
            expanded.bias.copy_(first.bias)
        self.head[0] = expanded
        self.input_mode = mode
        self.model_seed = int(seed)
        self.package_arm = 'FNO9_plain_query_MLP_cov12_' + mode
        self.macro_family = 'FNO9_joint_kinematic_covariates'
        self.backend = 'torch'
        self.covariate_schema = dict(columns=12, archive_precision='FP64', model_input_precision='FP32', old_output_scaler_changed=False, physical_U_passthrough=False, signed_stress_rule_or_VM_label_generated=False, extra_first_layer_parameters=1536)

    def query(self, case, start, stop, context, latent):
        cov = case['kinematic_inputs'][start:stop]
        assert cov.dtype == torch.float32 and cov.shape == (stop - start, 12) and bool(torch.isfinite(cov).all())
        if self.input_mode == 'ZERO':
            cov = torch.zeros_like(cov)
        elif self.input_mode == 'U_ONLY':
            cov = torch.cat((cov[:, :3], torch.zeros_like(cov[:, 3:])), dim=1)
        elif self.input_mode == 'GRAD_ONLY':
            cov = torch.cat((torch.zeros_like(cov[:, :3]), cov[:, 3:]), dim=1)
        values = query_grid(latent[0], case['query_indices'][start:stop], case['query_weights'][start:stop], 32768)
        native = torch.cat((case['xyz'][start:stop], case['features'][start:stop], context.expand(stop - start, -1), values), dim=1)
        assert native.shape == (stop - start, 76)
        return self.head(torch.cat((native, cov), dim=1))
