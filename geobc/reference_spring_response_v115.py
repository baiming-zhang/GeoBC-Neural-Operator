"""Fixed grounded six-DOF reference spring from an unmodified raw compliance.

This changes the approximate physics boundary equation. Its predictions require
independent PCG validation; no original/new-load OptiStruct GT is supplied.
"""
class ReferenceSpringResponse:
    def __init__(self,t,reference_api,guard):
        self.t,self.reference_api,self.guard=t,reference_api,guard;guard()
        C,D=reference_api.C,reference_api.scales
        assert C.is_cuda and C.dtype==t.float64 and C.shape==(6,6)
        Cn=D[:,None]*C*D[None,:]
        mean=float(Cn.diagonal().mean());assert mean>0
        # One predeclared strength, not a sweep or fit to a validation field.
        self.spring_diagonal=t.full((6,),1./mean,dtype=C.dtype,device=C.device)
        self.matrix=t.eye(6,dtype=C.dtype,device=C.device)+Cn*self.spring_diagonal[None,:]
        self.normalized_C=Cn
        assert bool(t.isfinite(self.matrix).all())and float(t.linalg.cond(self.matrix))<=1e12
        self.metadata=dict(normalized_spring_units='N/mm',physical_spring_scope='Grounded loaded RBE reference,6work-conjugate directions',
            fixed_strength='1 divided by mean diagonal normalized raw compliance',raw_C_unchanged=True,
            matrix_condition=float(t.linalg.cond(self.matrix)),new_original_OptiStruct_GT=False,signedS_provided=False)

    def query_force6(self,forces):
        t=self.t;self.guard();D=self.reference_api.scales
        assert forces.is_cuda and forces.dtype==t.float64 and forces.ndim==2 and forces.shape[1]==6 and bool(t.isfinite(forces).all())
        p=forces/D
        q=t.linalg.solve(self.matrix,(p@self.normalized_C.T).T).T
        physical=q/D
        effective=forces-(q*self.spring_diagonal)*D
        spring_energy=.5*(q.square()*self.spring_diagonal).sum(-1)
        total_energy=.5*(forces*physical).sum(-1)
        assert bool(t.isfinite(q).all()and t.isfinite(effective).all()and t.isfinite(total_energy).all())
        return dict(reference_motion6=physical,effective_original_force6=effective,spring_energy_Nmm=spring_energy,
            total_strain_energy_Nmm=total_energy,independent_PCGequation_validation_required=True)
