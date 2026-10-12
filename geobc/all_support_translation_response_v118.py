"""Force6 plus all four source support translation ports; rotations stay zero."""
class AllSupportTranslationResponse:
    def __init__(self,t,B6,force_basis,support_basis,support_reference_ids,support_unit_mm,guard):
        self.t,self.B,self.force_basis,self.support_basis,self.unit,self.guard=t,B6,force_basis,support_basis,support_unit_mm,guard;guard()
        assert B6.is_cuda and force_basis.is_cuda and support_basis.is_cuda
        assert B6.dtype==force_basis.dtype==support_basis.dtype==t.float64 and B6.shape==(6,6)
        assert force_basis.ndim==3 and force_basis.shape[0]==6 and force_basis.shape[2]==3
        assert support_basis.shape==(4,3,force_basis.shape[1],3)and support_unit_mm>0
        assert len(support_reference_ids)==4 and list(support_reference_ids)==sorted(set(support_reference_ids))
        assert bool(t.isfinite(force_basis).all()and t.isfinite(support_basis).all())and int(t.linalg.matrix_rank(B6))==6
        assert float(t.linalg.cond(B6))<=1e12
        self.metadata=dict(actual_allN=force_basis.shape[1],force_directions=6,support_translation_directions=12,
            actual_source_support_reference_ids=list(support_reference_ids),rotations_prescribed_zero=True,support_basis_unit_mm=support_unit_mm,
            arbitrary_support_selection_claim=False,new_original_OptiStruct_GT=False,signedS_provided=False)

    def query(self,force6,support_translations_mm):
        t=self.t;self.guard()
        assert force6.is_cuda and support_translations_mm.is_cuda and force6.dtype==support_translations_mm.dtype==t.float64
        assert force6.ndim==2 and force6.shape[1]==6 and support_translations_mm.shape==(len(force6),4,3)
        assert bool(t.isfinite(force6).all()and t.isfinite(support_translations_mm).all())
        c=t.linalg.solve(self.B,force6.T).T
        u=t.einsum('qk,kna->qna',c,self.force_basis)+t.einsum('qpk,pkna->qna',support_translations_mm/self.unit,self.support_basis)
        assert bool(t.isfinite(u).all())
        return dict(physical_U_mm=u,independent_mixed_equation_validation_required=True)
