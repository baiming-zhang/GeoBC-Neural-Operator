"""OriginalTS same parameters, BF16 operator arithmetic and FP32 query latent."""
import torch
from .transolver_conditioned_hash_v64 import TransolverConditionedHash,encode_case,query_case
from .transolver_conditioned_hash_v64 import profile as original_profile

FAMILY='TS_ORIGINAL_BF16_GLOBAL_HASH'


class BF16TransolverHash(TransolverConditionedHash):
    def __init__(self,seed=2026,device='cuda'):
        super().__init__(seed=seed,device=device)
        self.macro_family=FAMILY
        self.arm=FAMILY

    def encode_case(self,case):
        context=self.encoder(case['cloud'],case['mask'])
        grid=torch.cat((case['grid'],context[:,:,None,None,None].expand(1,32,64,64,64)),dim=1)
        grid=torch.cat((grid[:,1:4],grid[:,0:1],grid[:,4:]),dim=1)
        tokens=grid.flatten(2).transpose(1,2).contiguous()
        assert tokens.shape==(1,262144,61) and tokens.device.type=='cuda'
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            latent=self.operator(tokens)
        # Keep the existing float32 trilinear query and decoder contract.
        # Casting is differentiable; FP32 parameters and Adam state remain.
        return context,latent.float().transpose(1,2).reshape(1,16,64,64,64)

    @property
    def source_config(self):
        result=super().source_config
        result.update(method='Whole-case originalTS BF16 arithmetic and same continuous Hash',
            precision='Autocast BF16 only originalTS operator; PointNet/parameters/Adam FP32, returned latent FP32, nativeTCNN FP16',
            parameter_layout_identical_to_FP32_TS=True,architecture_change=False)
        return result


def build_solver(family,device='cuda',seed=2026):
    if family!=FAMILY:raise ValueError('Only separately released BF16 originalTS branch')
    return BF16TransolverHash(seed=seed,device=device)


def profile(model):
    result=original_profile(model)
    result['precision']=model.source_config['precision']
    return result
