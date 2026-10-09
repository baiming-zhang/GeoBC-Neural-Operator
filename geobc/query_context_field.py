"""Fixed-Gaussian query context from a shared BC-role PointNet cloud.

This matched candidate retains a global32 descriptor and adds query32 from
the same learned point64 features. Both variants have identical parameters;
zero computes the same branch then multiplies by zero. Active query features
increase effective capacity, so this is not an information-only comparison.
No target, coordinate derivative, attention cache or load-dependent cloud is
used. Four-corner parameter-gradient training remains an external recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor,nn

from .bc_role_field import PhysFieldRoleConfig,PointNetBCRoleEncoder
from .conditioned_field import ConditionedField,ConditionedFieldConfig
from .physics_field import PhysicsField
from .tcnn_fields import TCNNField


@dataclass(frozen=True)
class QueryContextFieldConfig(PhysFieldRoleConfig):
    bc_role_mode: Literal["actual"] = "actual"
    query_mode: Literal["zero","actual"] = "zero"
    query_dim: int = 32
    sigma: float = .15
    query_chunk: int = 4096

    def __post_init__(self):
        super().__post_init__()
        if (self.bc_role_mode!="actual" or self.query_mode not in ("zero","actual") or self.query_dim!=32
                or self.descriptor_dim!=32 or self.condition_dim!=38 or self.sigma!=.15 or not 1<=self.query_chunk<=4096):
            raise ValueError("Fixed query comparison needs BCactual, conditions38/global32/query32, sigma.15 and chunk<=4096")

    def conditioned_config(self):
        # Outer PhysicsField sees38 query-local inputs. ConditionedField appends
        # its global32 to the70=(original38 + query32) inputs, giving Identity102.
        return ConditionedFieldConfig(geometry_mode="shape",descriptor_dim=self.descriptor_dim,
            condition_dim=self.condition_dim+self.query_dim,out_channels=9,cloud_points=self.cloud_points,
            hidden_dim=self.hidden_dim,hidden_layers=self.hidden_layers,seed=self.seed)

    def record(self):
        result=super().record()
        result.update(method="GeoBC fixed-Gaussian query-context displacement-basis candidate",
            claim="matched architecture/activation ablation; active query branch increases effective capacity; no new-method/generalization claim",
            input_layout="original38 + query32 + global32 =102 native Identity conditions; xyz separately hash-encoded",
            query_context={"mode":self.query_mode,"projection":[64,32],"projection_parameters":2080,
                "sigma_normalized_xyz":self.sigma,"distance":"direct FP32 coordinate differences squared, no TF32 dot distance",
                "weights":"masked softmax(-squared_distance/(2*sigma^2)); valid cloud weights sum1",
                "pool_matmul_precision":"FP32 tensors; pooling follows the trainer's high/TF32 matmul setting",
                "order":"point64 -> Linear64to32 -> Gaussian weighted pool -> non-affine LayerNorm32 -> tanh",
                "linear_projection_before_pool":"equivalent to postpool linear in exact arithmetic because weights sum1",
                "query_chunk":self.query_chunk,"zero_control":"compute identical query context, multiply by0; projection/token gradients explicitly0",
                "coordinate_derivatives":False,"point_features_reused_for_global_and_query":True,
                "state_cache":"explicit geometry-context bundle per optimizer step, no data_ptr/implicit mutable lookup"},
            capacity_cost={"new_projection_parameters":2080,"old_native_identity_conditions":70,"new_native_identity_conditions":102,
                "old_nominal_encoded_channels":94,"new_nominal_encoded_channels":126,
                "nominal_added_first_layer_weights_at_width128":4096,
                "native_padding_policy":"retain official TCNN Composite/channel padding; actual padded decoder parameter cost recorded after native construction, not inferred from nominal widths",
                "training_saved_Gaussian_weights_at51200x1024_FP32_bytes":51200*1024*4,
                "chunk4096_distance_xyz_temporary_bytes":4096*1024*3*4,
                "chunking_limit":"limits temporary distance buffers, not all saved weights required by token-gradient backward"})
        return result


@dataclass(frozen=True)
class GeometryQueryContext:
    """Explicit differentiable CAD context. Integer/slice keep a batch axis."""
    global_tensor: Tensor
    cloudxyz: Tensor
    mask: Tensor
    projected_tokens: Tensor

    @property
    def shape(self):return self.global_tensor.shape
    @property
    def ndim(self):return self.global_tensor.ndim
    @property
    def device(self):return self.global_tensor.device
    @property
    def dtype(self):return self.global_tensor.dtype
    @property
    def requires_grad(self):return self.global_tensor.requires_grad
    @property
    def grad(self):return self.global_tensor.grad
    def __len__(self):return len(self.global_tensor)
    def __getitem__(self,index):
        if isinstance(index,str):
            if index not in ("global_tensor","cloudxyz","mask","projected_tokens"):raise KeyError(index)
            return getattr(self,index)
        values=[value[index] for value in (self.global_tensor,self.cloudxyz,self.mask,self.projected_tokens)]
        if values[0].ndim==1:values=[value.unsqueeze(0) for value in values]
        return GeometryQueryContext(*values)
    def retain_grad(self):
        if self.global_tensor.requires_grad:self.global_tensor.retain_grad()
        if self.projected_tokens.requires_grad:self.projected_tokens.retain_grad()


class _QueryConditionedField(ConditionedField):
    def __init__(self,role_config:QueryContextFieldConfig,device="cuda",decoder=None):
        nn.Module.__init__(self)
        self.role_config=role_config;self.config=role_config.conditioned_config()
        self.encoder=PointNetBCRoleEncoder(self.config.descriptor_dim,self.config.point_width)
        self.query_projection=nn.Linear(self.config.point_width,role_config.query_dim)
        self.query_normalization=nn.LayerNorm(role_config.query_dim,elementwise_affine=False)
        expected=self.config.decoder_config()
        if decoder is not None and hasattr(decoder,"config") and decoder.config!=expected:
            raise ValueError("Injected decoder must have the exact102-condition native layout")
        self.decoder=decoder if decoder is not None else TCNNField(config=expected,device=device)
        self.backend=getattr(self.decoder,"backend","injected_cpu_interface_decoder")
        self.to(device)

    @property
    def source_config(self):
        result=super().source_config
        result["encoder"]=self.role_config.record()["conditioned_field"]["encoder"]
        result.update(query_context=self.role_config.record()["query_context"],
                      query_projection_parameters=sum(p.numel() for p in self.query_projection.parameters()),
                      geometry_source="same fixed BCactual xyz3+role2 cloud; explicit global/token context")
        return result

    def encode_geometry(self,geometry_cloud:Tensor,geometry_mask:Tensor|None=None):
        if geometry_cloud.ndim==2:
            geometry_cloud=geometry_cloud.unsqueeze(0)
            if geometry_mask is not None and geometry_mask.ndim==1:geometry_mask=geometry_mask.unsqueeze(0)
        if geometry_cloud.ndim!=3 or geometry_cloud.shape[-1]!=5 or min(geometry_cloud.shape[:2])<1:
            raise ValueError("Query context needs nonempty BCactual cloud[B,P,5]")
        if geometry_mask is None:geometry_mask=torch.ones(geometry_cloud.shape[:2],dtype=torch.bool,device=geometry_cloud.device)
        if geometry_mask.dtype!=torch.bool or geometry_mask.shape!=geometry_cloud.shape[:2] or geometry_mask.device!=geometry_cloud.device:
            raise ValueError("Cloud mask must be bool[B,P] on cloud device")
        counts=geometry_mask.sum(1,keepdim=True)
        if torch.any(counts==0):raise ValueError("Every CAD needs a valid cloud node")
        cloud=geometry_cloud.detach().float()  # Fixed geometry/BC inputs, no input derivatives.
        values=self.encoder.point_mlp(cloud)
        expanded=geometry_mask.unsqueeze(-1)
        maximum=values.masked_fill(~expanded,-torch.inf).amax(1)
        average=(values*expanded).sum(1)/counts.to(values.dtype)
        global_tensor=self.encoder.head(torch.cat((maximum,average),-1))
        tokens=self.query_projection(values)
        return GeometryQueryContext(global_tensor,cloud[...,:3],geometry_mask,tokens)

    def query_context(self,xyz:Tensor,bundle:GeometryQueryContext,cad_index:Tensor|None=None):
        if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<1 or not isinstance(bundle,GeometryQueryContext):
            raise ValueError("Nonempty query xyz[N,3] and explicit GeometryQueryContext required")
        if bundle.device!=xyz.device:raise ValueError("Bundle and query must share device")
        if cad_index is None:
            if len(bundle)!=1:raise ValueError("Multiple CAD contexts require cad_index")
            cad_index=torch.zeros(len(xyz),dtype=torch.long,device=xyz.device)
        if cad_index.dtype!=torch.long or cad_index.shape!=(len(xyz),) or cad_index.device!=xyz.device:
            raise ValueError("cad_index must be long[N] on query device")
        if bool((cad_index<0).any()) or bool((cad_index>=len(bundle)).any()):raise ValueError("cad_index outside explicit bundle rows")
        positions=xyz.detach().float();chunks=[];locations=[]
        for cad in range(len(bundle)):
            ids=torch.nonzero(cad_index==cad,as_tuple=False).flatten()
            for start in range(0,len(ids),self.role_config.query_chunk):
                current=ids[start:start+self.role_config.query_chunk]
                # No dot-product distance or TF32 cancellation near cloud nodes.
                squared=(positions[current,None,:]-bundle.cloudxyz[cad,None,:,:]).square().sum(-1)
                logits=(-squared/(2*self.role_config.sigma**2)).masked_fill(~bundle.mask[cad,None,:],-torch.inf)
                weights=torch.softmax(logits,dim=-1)
                pooled=weights@bundle.projected_tokens[cad].float()
                result=torch.tanh(self.query_normalization(pooled))
                if self.role_config.query_mode=="zero":result=result*0
                chunks.append(result);locations.append(current)
        empty=positions.new_empty((len(positions),self.role_config.query_dim))
        return empty.index_copy(0,torch.cat(locations),torch.cat(chunks))

    def forward(self,xyz:Tensor,features:Tensor,geometry_cloud=None,cad_index=None,*,geometry_mask=None,context=None):
        if features.shape!=(len(xyz),self.role_config.condition_dim) or features.device!=xyz.device:
            raise ValueError("Outer query-local features must remain original38")
        if context is not None and geometry_cloud is not None:raise ValueError("Supply context bundle or cloud, not both")
        if context is None and geometry_cloud is None:raise ValueError("A context bundle or BCactual cloud is required")
        bundle=context if context is not None else self.encode_geometry(geometry_cloud,geometry_mask)
        query=self.query_context(xyz,bundle,cad_index)
        features70=torch.cat((features.float(),query),-1)
        return super().forward(xyz.detach(),features70,cad_index=cad_index,context=bundle.global_tensor)


class QueryContextPhysicsField(PhysicsField):
    """PhysicsField API with explicit bundle and38+query32+global32 conditions."""
    def __init__(self,config:QueryContextFieldConfig|None=None,device="cuda",decoder=None):
        nn.Module.__init__(self)
        self.config=config or QueryContextFieldConfig()
        if not isinstance(self.config,QueryContextFieldConfig):raise TypeError("Expected QueryContextFieldConfig")
        self.field=_QueryConditionedField(self.config,device,decoder);self.backend=self.field.backend

    @property
    def query_projection(self):return self.field.query_projection

    @property
    def source_config(self):
        result=super().source_config
        result.update(query_projection_parameters=sum(p.numel() for p in self.query_projection.parameters()),
                      query_context_dim=self.config.query_dim,native_identity_condition_dim=102,
                      query_context_bundle="global_tensor/cloudxyz/mask/projected_tokens; explicit differentiable CAD slices")
        return result


__all__=["QueryContextFieldConfig","QueryContextPhysicsField","GeometryQueryContext"]
