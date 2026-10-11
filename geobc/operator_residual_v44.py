"""Zero-initialized conditional residual on a supplied frozen FNO teacher.

This is an existing-component composition, not a new algorithm. A separate
driver must validate/load the TRAIN-only teacher and own data, clocks and
optimizer. Native Hash residuals retain the official fixed TCNN recipe.
"""
import copy
import torch
from torch import nn
from .fno import query_grid
from .tcnn_fields import TCNNField,TCNNFieldConfig


class FrozenOperatorResidual(nn.Module):
    def __init__(self,teacher,mode,seed=2026,device='cuda',*,injected_hash_decoder=None):
        super().__init__()
        if mode not in ('mlp','hash') or teacher.arm!='FNO9':raise ValueError('Frozen FNO9 with one MLP/Hash residual only')
        self.teacher=teacher.eval().requires_grad_(False)
        self.mode=mode
        self.decoder_config=TCNNFieldConfig(out_channels=4,input_mode='both',condition_dim=73,seed=seed)
        if mode=='mlp':
            if injected_hash_decoder is not None:raise ValueError('No injection for ordinary MLP')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.decoder=nn.Sequential(nn.Linear(76,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,4))
            self.reference=copy.deepcopy(self.decoder)
            self.backend='torch_FP32_MLP'
        elif injected_hash_decoder is not None:
            if str(device)!='cpu' or injected_hash_decoder.config!=self.decoder_config:raise ValueError('Exactconfig CPU interface injection only')
            self.decoder=injected_hash_decoder
            self.reference=copy.deepcopy(self.decoder)
            self.backend='injected_CPU_interface_not_native_Hash'
        else:
            self.decoder=TCNNField(config=self.decoder_config,device=device)
            self.reference=TCNNField(config=self.decoder_config,device=device)
            # Public state interface only; no native flat-parameter offsets.
            self.reference.load_state_dict(self.decoder.state_dict(),strict=True)
            self.backend=self.decoder.backend
        self.reference.eval().requires_grad_(False)
        self.to(device)

    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval();self.reference.eval()
        return self

    def encode_case(self,case):
        with torch.no_grad():
            if 'grid' in case:return self.teacher.encode_case(case)
            from .resolution_fno_v41 import original_input_grid
            grid,_,_=original_input_grid(case,64)
            return self.teacher.encode_case(dict(case,grid=grid))

    def decode_points(self,xyz,native25,context,values):
        n=len(xyz)
        if not n or xyz.shape!=(n,3) or native25.shape!=(n,25) or context.shape!=(1,32) or values.shape!=(n,16):raise ValueError('Explicit fixed-context dimensions required')
        if any(t.device!=xyz.device or t.dtype!=xyz.dtype for t in (native25,context,values)):raise ValueError('Shared device/dtype required')
        condition=torch.cat((native25,context.expand(n,-1),values),dim=-1)
        inputs=torch.cat((xyz,condition),dim=-1)
        baseline=self.teacher.head(inputs)
        if self.mode=='mlp':
            delta=self.decoder(inputs)-self.reference(inputs)
        else:
            delta=self.decoder(xyz,condition)-self.reference(xyz,condition)
        # Both residual networks start with identical weights. Delta is zero
        # while the trainable decoder has an ordinary nonzero first gradient.
        return baseline+delta

    def query(self,case,start,stop,context,latent):
        from .solver_selection_data import _stencil
        xyz=case['xyz'][start:stop]
        if 'query_indices' in case and 'query_weights' in case:
            indices=case['query_indices'][start:stop];weights=case['query_weights'][start:stop]
        else:
            indices,weights=_stencil(xyz,64)
        values=query_grid(latent[0],indices,weights,32768)
        return self.decode_points(xyz,case['features'][start:stop],context,values)

    def query_points(self,xyz,native25,context,latent,chunk_size=32768):
        if type(chunk_size) is not int or chunk_size<1 or latent.shape!=(1,16,64,64,64):raise ValueError('Explicit original R64 field required')
        case=dict(xyz=xyz,features=native25)
        return torch.cat([self.query(case,i,min(i+chunk_size,len(xyz)),context,latent) for i in range(0,len(xyz),chunk_size)])

    def profile(self):
        counts={k:sum(p.numel() for p in getattr(self,k).parameters()) for k in ('teacher','decoder','reference')}
        return dict(method='FNO5_frozen_'+self.mode+'_zero_residual',counts=counts,total_parameters=sum(counts.values()),
            trainable_parameters=counts['decoder'],backend=self.backend,teacher_weights_frozen=True,reference_weights_frozen=True,
            initialization='baseline encoded4 + decoder(theta)-decoder(theta0), exactteacher at identical theta/theta0; no zero gate or native slicing',
            target='Original signed physical U/S loss after common scaler decode, not teacher distillation',
            exposures='Teacher trained5; decoder additional5 planned, not fresh5 or allparameters10',
            continuous_interface='Fixed context/trilinear R64/explicit native25 extension; continuity conditional on that extension, C0 not generally C1',
            offnode_FEM_accuracy_measured=False,owns_data_or_launch=False)
