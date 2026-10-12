"""Predictor-only physical U and full displacement-gradient query covariates.

No stress scalar, sign convention, constitutive target or GT is generated.
Actual constructor/body/GPU evaluation needs a separately frozen source gate.
"""
def recover_kinematic_inputs(t,vertices,cells,U,positive_cell_volume,guard,chunk_cells=8192):
    from .physical_U_query_v104 import shape
    assert vertices.is_cuda and vertices.dtype==t.float64 and vertices.ndim==2 and vertices.shape[1]==3
    assert cells.is_cuda and cells.dtype==t.int64 and cells.ndim==2 and cells.shape[1]==10
    assert U.is_cuda and U.dtype==t.float64 and U.shape==(3,len(vertices),3)and bool(t.isfinite(U).all())
    assert positive_cell_volume.is_cuda and positive_cell_volume.dtype==t.float64 and positive_cell_volume.shape==(len(cells),)
    assert bool(t.isfinite(positive_cell_volume).all()and (positive_cell_volume>0).all())and chunk_cells==8192
    bary=t.tensor([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.],
        [.5,.5,0,0],[0,.5,.5,0],[.5,0,.5,0],[.5,0,0,.5],[0,.5,0,.5],[0,0,.5,.5]],dtype=t.float64,device='cuda')
    _,Dr,_=shape(t,bary[:,:3]);_,Dc,_=shape(t,t.full((1,3),.25,dtype=t.float64,device='cuda'))
    numerator=t.zeros((3,len(vertices),9),dtype=t.float64,device='cuda');denominator=t.zeros(len(vertices),dtype=t.float64,device='cuda')
    maximum_condition=0.;minimum_scaled_det=None
    for start in range(0,len(cells),chunk_cells):
        guard();ids=cells[start:start+chunk_cells];x=vertices[ids];vol=positive_cell_volume[start:start+chunk_cells]
        scale=t.linalg.vector_norm(x.amax(1)-x.amin(1),dim=-1);assert bool((scale>0).all())
        J=t.einsum('cia,eib->ceab',x,Dr);center=t.einsum('cia,eib->ceab',x,Dc)
        sv=t.linalg.svdvals(J);det=t.linalg.det(J);condition=sv[...,0]/sv[...,-1];scaled=det.abs()/scale[:,None]**3
        if not bool(t.isfinite(J).all()and (sv[...,-1]>0).all()and (condition<=1e12).all()and (scaled>1e-14).all()and (det.sign()==t.linalg.det(center).sign()).all()):
            raise ValueError('Original curved cells do not support all-node physical gradient inputs; no pseudo/precision fallback')
        maximum_condition=max(maximum_condition,float(condition.max()));m=float(scaled.min());minimum_scaled_det=m if minimum_scaled_det is None else min(minimum_scaled_det,m)
        grad=t.einsum('eib,ceba->ceia',Dr,t.linalg.inv(J))
        du=t.einsum('lcia,ceib->lceab',U[:,ids],grad)
        assert bool(t.isfinite(du).all());flat=ids.reshape(-1);weight=vol[:,None].expand(-1,10).reshape(-1)
        numerator.index_add_(1,flat,(du*vol[None,:,None,None,None]).reshape(3,-1,9));denominator.index_add_(0,flat,weight)
    if not bool((denominator>0).all()):raise ValueError('Original nodes missing actual incident cells; no subsetting')
    gradient=numerator/denominator[None,:,None];features=t.cat((U,1000.*gradient),dim=-1)
    assert features.shape==(3,len(vertices),12)and bool(t.isfinite(features).all())
    return features,dict(actual_allN=len(vertices),actual_cells=len(cells),physical_U_columns=3,full_gradient_columns=9,
        gradient_component_order='rowmajor displacement_component then physical_XYZ_coordinate',gradient_units='mm/mm',fixed_gradient_feature_multiplier=1000.,
        physical_U_units='mm',recovery='Positive curved-cell volume average of actual P2 nodal derivatives,explicit predictor information',
        maximum_node_J_condition=maximum_condition,minimum_node_abs_det_over_scale_cubed=minimum_scaled_det,
        original_stress_targets_generated=0,signed_or_VM_rule_assumed=False,old_output_scaler_must_remain_TRAIN40=True)
