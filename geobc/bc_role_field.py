"""Matched BC-role-aware PointNet control, with unchanged displacement decoder.

Both zero/actual variants construct the same five-input encoder and consume
identical initialization RNG calls. Only encode_geometry zeroes the two role
inputs in the control. This is a candidate comparison, not a new-method or
generalization claim. Canonical roles are independent of fields and load size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor,nn

from .conditioned_field import ConditionedField,ConditionedFieldConfig
from .physics_field import PhysicsField,PhysicsFieldConfig
from .tcnn_fields import TCNNField

ROLE_ORDER = ("support_union_original_DOF0_to5", "load_member_original_feature6")


def _encoder_record(descriptor_dim=32,point_width=64):
    return {"point_mlp":[5,32,point_width,point_width],"pooling":"concatenated masked max and masked mean",
            "head":[2*point_width,point_width,descriptor_dim],"activation":"ReLU",
            "output":"non-affine LayerNorm then tanh, bounded [-1,1]","batch_norm":False,"dropout":False,
            "input_order":["x_normalized","y_normalized","z_normalized",*ROLE_ORDER],
            "extra_parameters_vs_xyz_encoder":64}


@dataclass(frozen=True)
class PhysFieldRoleConfig(PhysicsFieldConfig):
    condition_dim: int = 38
    bc_role_mode: Literal["zero","actual"] = "zero"

    def __post_init__(self):
        super().__post_init__()
        if self.bc_role_mode not in ("zero","actual") or self.geometry_mode!="shape":
            raise ValueError("BC-role comparison requires shape PointNet and zero/actual role mode")

    def record(self):
        result = super().record()
        result.update(method="GeoBC BC-role-aware global PointNet displacement-basis candidate",
                      claim="matched zero/actual input-condition experiment; efficacy requires measured results",
                      bc_role_mode=self.bc_role_mode,role_order=list(ROLE_ORDER),role_input_dimension=2,
                      global_cloud_input_dimension=5,extra_encoder_parameters_vs_xyz=64,
                      geometry_target_independent=True,geometry_load_independent=True,
                      role_source="same fixed canonical ver cloud nodes; support union any original25 DOF0:6; load member original25 column6",
                      role_control="both models instantiate5-input encoder; zero masks roles only at encode_geometry; same seed/RNG/parameter layout",
                      input_layout="query original25+local13 unchanged; global descriptor32 from xyz3+two canonical binary roles")
        result["conditioned_field"]["encoder"] = _encoder_record(self.descriptor_dim)
        result["conditioned_field"]["geometry_source"] = "fixed normalized xyz and two target-independent canonical CAD support/load membership roles"
        return result


class PointNetBCRoleEncoder(nn.Module):
    """Five-input PointNet with the original layer/pooling/head state names."""
    def __init__(self,descriptor_dim=32,point_width=64):
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,point_width),nn.ReLU(),
                                       nn.Linear(point_width,point_width),nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2*point_width,point_width),nn.ReLU(),nn.Linear(point_width,descriptor_dim),
                                  nn.LayerNorm(descriptor_dim,elementwise_affine=False),nn.Tanh())

    def forward(self,cloud:Tensor,mask:Tensor|None=None):
        if cloud.ndim!=3 or cloud.shape[-1]!=5 or min(cloud.shape[:2])<1:
            raise ValueError("BC-role cloud must have nonempty shape[B,P,5]")
        if mask is None: mask = torch.ones(cloud.shape[:2],dtype=torch.bool,device=cloud.device)
        if mask.dtype!=torch.bool or mask.shape!=cloud.shape[:2] or mask.device!=cloud.device:
            raise ValueError("BC-role mask must be bool[B,P] on cloud device")
        count = mask.sum(1,keepdim=True)
        if torch.any(count==0): raise ValueError("Every CAD needs a valid cloud point")
        values = self.point_mlp(cloud.float()); expanded = mask.unsqueeze(-1)
        maximum = values.masked_fill(~expanded,-torch.inf).amax(1)
        average = (values*expanded).sum(1)/count.to(values.dtype)
        return self.head(torch.cat((maximum,average),-1))


class _BCRoleConditionedField(ConditionedField):
    def __init__(self,config:ConditionedFieldConfig,role_mode:str,device="cuda",decoder=None):
        # Initialize this architecture exactly once; replacing a preconstructed
        # xyz encoder would consume unnecessary initialization RNG calls.
        nn.Module.__init__(self)
        self.config = config; self.bc_role_mode = role_mode
        self.encoder = PointNetBCRoleEncoder(config.descriptor_dim,config.point_width)
        expected = config.decoder_config()
        if decoder is not None and hasattr(decoder,"config") and decoder.config!=expected:
            raise ValueError("Injected decoder config differs from the frozen conditioned decoder")
        self.decoder = decoder if decoder is not None else TCNNField(config=expected,device=device)
        self.backend = getattr(self.decoder,"backend","injected_cpu_interface_decoder")
        self.to(device)

    @property
    def source_config(self):
        result = super().source_config
        result.update(encoder=_encoder_record(self.config.descriptor_dim,self.config.point_width),
                      bc_role_mode=self.bc_role_mode,geometry_target_independent=True,geometry_load_independent=True,
                      geometry_source="same canonical normalized xyz3 plus support union/load membership2",
                      role_input_policy="zero role columns3:5 only at encoder entry" if self.bc_role_mode=="zero" else "retain canonical binary role columns3:5")
        return result

    def encode_geometry(self,geometry_cloud:Tensor,geometry_mask:Tensor|None=None):
        if geometry_cloud.ndim==2:
            geometry_cloud = geometry_cloud.unsqueeze(0)
            if geometry_mask is not None and geometry_mask.ndim==1: geometry_mask = geometry_mask.unsqueeze(0)
        if geometry_cloud.ndim!=3 or geometry_cloud.shape[-1]!=5:
            raise ValueError("BC-role geometry_cloud must have shape[B,P,5]")
        if self.bc_role_mode=="zero":
            geometry_cloud = geometry_cloud.clone()
            geometry_cloud[...,3:5] = 0
        return self.encoder(geometry_cloud,geometry_mask)


class PhysicsRoleField(PhysicsField):
    """PhysicsField API and state paths, with five-input global role encoder.

    encode_geometry(points5,mask)->context; inherited response_basis/forward
    retain original force/moment masking, nine basis outputs and alpha compose.
    The native TCNN decoder module/config/seed and query features are unchanged.
    """
    def __init__(self,config:PhysFieldRoleConfig|None=None,device="cuda",decoder=None):
        nn.Module.__init__(self)
        self.config = config or PhysFieldRoleConfig()
        if not isinstance(self.config,PhysFieldRoleConfig):
            raise TypeError("PhysicsRoleField requires PhysFieldRoleConfig")
        self.field = _BCRoleConditionedField(self.config.conditioned_config(),self.config.bc_role_mode,device,decoder)
        self.backend = self.field.backend


__all__ = ["PhysFieldRoleConfig","PhysicsRoleField","PointNetBCRoleEncoder","ROLE_ORDER"]
