"""Target-free R128 representation candidate; no experiment launch ownership.

Use the unchanged R64 FNO9 weights/width/modes and direct original-node splat
at a finer grid. This is a resolution/input/query change, not a new solver.
Do not upsample the old coarse grid or rasterize any target field.
"""
import torch
from .common_grid_solver import CommonGridSolver


def original_input_grid(case, resolution):
    from .solver_selection_data import _stencil, _splat
    if resolution not in (64,128):raise ValueError('One fixed64/128 representation pair only')
    xyz=case['xyz'];features=case['features']
    if xyz.ndim!=2 or xyz.shape[1]!=3 or features.shape!=(len(xyz),25):raise ValueError('Complete original xyz/native25 required')
    if xyz.dtype!=torch.float32 or features.dtype!=torch.float32:raise ValueError('Original FP32 inputs only')
    if not torch.isfinite(xyz).all() or not torch.isfinite(features).all():raise ValueError('Finite input fields required')
    if xyz.min()<-2e-6 or xyz.max()>1+2e-6:raise ValueError('Original normalized reference cube only')
    indices,weights=_stencil(xyz,resolution)
    raster,active=_splat(features,indices,weights,resolution)
    axis=torch.linspace(0,1,resolution,device=xyz.device,dtype=xyz.dtype)
    coordinates=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'))
    grid=torch.cat((active.to(xyz.dtype).reshape(1,resolution,resolution,resolution),coordinates,
        raster.T.reshape(25,resolution,resolution,resolution)),dim=0).unsqueeze(0)
    if not torch.isfinite(grid).all():raise FloatingPointError('Nonfinite target-free grid')
    return grid,indices,weights


class ResolutionFNO(CommonGridSolver):
    def __init__(self,seed=2026,resolution=128,device='cuda'):
        if resolution not in (64,128):raise ValueError('Fixed64/128 resolution pair only')
        super().__init__('FNO9',seed)
        self.resolution=resolution
        self.method='FNO9_R'+str(resolution)
        self.macro_family=self.method
        self.to(device)

    def encode_case(self,case):
        r=self.resolution
        context=self.encoder(case['cloud'],case['mask'])
        # Construct from original nodes every case/update, retaining the graph
        # to context. The fixed inputs themselves are not learnable targets.
        grid,_,_=original_input_grid(case,r)
        inputs=torch.cat((grid,context[:,:,None,None,None].expand(1,32,r,r,r)),dim=1)
        return context,self.operator(inputs)

    def query(self,case,start,stop,context,latent):
        from .solver_selection_data import _stencil
        from .fno import query_grid
        xyz=case['xyz'][start:stop];indices,weights=_stencil(xyz,self.resolution)
        value=query_grid(latent[0],indices,weights,32768)
        return self.head(torch.cat((xyz,case['features'][start:stop],context.expand(stop-start,-1),value),dim=1))

    def profile(self):
        return dict(method=self.method,family=self.macro_family,parameter_count=self.counts()['total'],counts=self.counts(),
            architecture=self.operator.config.record(),grid_resolution=self.resolution,
            input_policy='Unchanged original full-node native25 weighted splat to fresh grid; no coarse-grid upsample or target rasterization',
            query_policy='Original normalized trilinear stencil at actual new resolution; full original-node query',
            changed='Grid rasterization/representation and query resolution64→128 only; width9/modes12/layers4/head/encoder/initial weights unchanged',
            initialization_matches_common_R64_FNO9=True,owns_training_or_data=False,
            precision='highest FP32; caller disables TF32',independent_offnode_FEM_accuracy_measured=False)
