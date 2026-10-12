"""Work-conjugate reference translation/rotation and energy from six U bases.

The RBE3 weighted rigid-fit Hbar is the accepted approximate source transfer.
No point locator projects an outside-mesh reference onto a material cell.
Raw compliance is retained; its symmetric part is used only for the positivity
diagnostic, never to replace predicted motions or rectify reciprocity failure.
"""

class ReferenceMotionResponse:
    def __init__(self,t,Hbar,members,length_mm,basis,B,guard):
        self.t,self.guard=t,guard;guard()
        assert Hbar.is_cuda and members.is_cuda and basis.is_cuda and B.is_cuda
        assert Hbar.dtype==basis.dtype==B.dtype==t.float64 and members.dtype==t.int64
        assert basis.ndim==3 and basis.shape[0]==6 and basis.shape[2]==3
        assert members.ndim==1 and Hbar.shape==(6,3*len(members)) and B.shape==(6,6)
        assert length_mm>0 and bool(((members>=0)&(members<basis.shape[1])).all())
        assert members.unique().numel()==len(members) and bool(t.isfinite(Hbar).all() and t.isfinite(basis).all() and t.isfinite(B).all())
        assert int(t.linalg.matrix_rank(B))==6 and float(t.linalg.cond(B))<=1e12
        scales=t.tensor([1.,1.,1.,length_mm,length_mm,length_mm],dtype=t.float64,device=B.device)
        scaled_motion_columns=Hbar@basis[:,members].reshape(6,-1).T
        physical_motion_columns=scaled_motion_columns/scales[:,None]
        C=t.linalg.solve(B.T,physical_motion_columns.T).T
        normalized_C=scales[:,None]*C*scales[None,:]
        norm=float(t.linalg.vector_norm(normalized_C));assert norm>0
        reciprocity=float(t.linalg.vector_norm(normalized_C-normalized_C.T))/norm
        symmetric_part=(normalized_C+normalized_C.T)*.5
        eigenvalues=t.linalg.eigvalsh(symmetric_part)
        minimum=float(eigenvalues.min());maximum=float(eigenvalues.max())
        self.C,self.motion_columns,self.scales=C,physical_motion_columns,scales
        self.metadata=dict(actual_allN=basis.shape[1],actual_member_nodes=len(members),physical_load_rank=6,
            reference_translation_units='mm',reference_rotation_units='rad',energy_units='Nmm',
            load_units=['N','N','N','Nmm','Nmm','Nmm'],reciprocity_relative=reciprocity,
            energy_normalized_minimum_eigenvalue=minimum,energy_normalized_maximum_eigenvalue=maximum,
            energy_positive=minimum>0,energy_normalized_condition=maximum/minimum if minimum>0 else None,
            raw_C_preserved=True,symmetric_part_only_diagnostic=True,source_transfer='accepted_weightedLS_RBE3_Hbar',
            independent_OptiStruct_reference_rotation_GT=False,signedS_provided=False)
        guard()

    def query_force6(self,forces):
        t=self.t;self.guard()
        assert forces.is_cuda and forces.dtype==t.float64 and forces.ndim==2 and forces.shape[1]==6 and bool(t.isfinite(forces).all())
        motion=forces@self.C.T
        energy=.5*(forces*motion).sum(-1)
        assert bool(t.isfinite(motion).all() and t.isfinite(energy).all())
        return dict(reference_motion6=motion,strain_energy_Nmm=energy,
            motion_units=['mm','mm','mm','rad','rad','rad'],independent_original_GT_accuracy=False)
