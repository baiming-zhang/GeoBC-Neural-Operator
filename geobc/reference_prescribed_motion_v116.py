"""Exact prescribed six-DOF reference motion from unchanged raw compliance.

Independent constraint-nullspace PCG is required before claiming numerical
validity. This is an exact imposed motion, not a large spring approximation.
"""
class PrescribedReferenceMotion:
    def __init__(self,t,reference_api,guard):
        self.t,self.reference_api,self.guard=t,reference_api,guard;guard()
        D=reference_api.scales;C=reference_api.C
        assert C.is_cuda and C.dtype==t.float64 and C.shape==(6,6)
        self.normalized_C=D[:,None]*C*D[None,:]
        assert bool(t.isfinite(self.normalized_C).all())and float(t.linalg.cond(self.normalized_C))<=1e12
        self.metadata=dict(boundary='Exact Hbar u_members = D physical target motion',
            raw_C_unchanged=True,large_spring_limit_used=False,new_original_OptiStruct_GT=False,signedS_provided=False)

    def query_motion6(self,motions,applied_force6):
        t=self.t;self.guard();D=self.reference_api.scales
        assert motions.is_cuda and applied_force6.is_cuda and motions.dtype==applied_force6.dtype==t.float64
        assert motions.ndim==2 and motions.shape[1]==6 and applied_force6.shape==motions.shape
        assert bool(t.isfinite(motions).all()and t.isfinite(applied_force6).all())
        q=motions*D
        total_normalized_force=t.linalg.solve(self.normalized_C,q.T).T
        total_physical_force=total_normalized_force*D
        reaction=total_physical_force-applied_force6
        energy=.5*(total_physical_force*motions).sum(-1)
        assert bool(t.isfinite(total_physical_force).all()and t.isfinite(energy).all())
        return dict(effective_original_force6=total_physical_force,reference_reaction_force6=reaction,
            body_strain_energy_Nmm=energy,reference_motion6=motions,independent_constraint_PCGequation_validation_required=True)
