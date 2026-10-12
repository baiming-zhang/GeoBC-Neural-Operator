"""Continuous displacement queries with loads and support translations."""

from .query import FullRank6PhysicalUQuery

class AllSupportPhysicalUQuery(FullRank6PhysicalUQuery):

    def __init__(self, t, vertices, cells, force_basis, support_basis, B6, reference_ids, unit_mm, guard):
        super().__init__(t, vertices, cells, force_basis, B6, guard)
        assert support_basis.is_cuda and support_basis.dtype == t.float64
        assert support_basis.shape == (4, 3, len(vertices), 3) and bool(t.isfinite(support_basis).all())
        assert reference_ids == sorted(set(reference_ids)) and len(reference_ids) == 4 and (unit_mm > 0)
        self.basis = t.cat((force_basis, support_basis.reshape(12, len(vertices), 3)), dim=0)
        self.support_unit_mm = unit_mm
        self.metadata.update(actual_source_support_reference_ids=list(reference_ids), support_translation_directions=12, support_rotations_prescribed_zero=True, total_response_directions=18, new_original_support_GT_accuracy=False)

    def response_coefficients(self, force6, support_translation_mm):
        t = self.torch
        c = self.coefficients(force6)
        assert support_translation_mm.dtype == t.float64 and support_translation_mm.is_cuda
        assert support_translation_mm.shape == (len(force6), 4, 3) and bool(t.isfinite(support_translation_mm).all())
        return t.cat((c, (support_translation_mm / self.support_unit_mm).reshape(len(force6), 12)), dim=1)

    def query(self, xyz, element_rows, force6, support_translation_mm):
        c = self.response_coefficients(force6, support_translation_mm)
        values = self.basis_at(xyz, element_rows)
        assert c.shape == (len(xyz), 18)
        return self.torch.einsum('qk,qka->qa', c, values)

class LocatedAllSupportUResponse:

    def __init__(self, api, locator, source_contract):
        assert api.vertices is locator.x and api.cells is locator.cells
        assert source_contract['REFC'] == '123456' and source_contract['material_fixed']
        assert source_contract['support_reference_ids'] == api.metadata['actual_source_support_reference_ids']
        assert source_contract['rotations_prescribed_zero'] and source_contract['physical_units'] == ['mm', 'N', 'Nmm']
        self.api, self.locator, self.source_contract = (api, locator, dict(source_contract))
        self.metadata = dict(interface='XYZmm_force6_support4x3mm_to_Umm', caller_element_IDs_required=False, location_branch_discrete=True, total_response_directions=18, support_rotation_directions=0, new_OptiStruct_GT_accuracy=False, C1_claim=False, signedS_provided=False)

    def query(self, xyz, force6, support_translation_mm):
        t = self.api.torch
        assert xyz.shape == (len(force6), 3) and xyz.dtype == t.float64 and xyz.is_cuda and bool(t.isfinite(xyz).all())
        self.api.response_coefficients(force6, support_translation_mm)
        with t.no_grad():
            ids, records = self.locator.locate(xyz.detach())
        if not bool((ids >= 0).all()) or any((v['status'] == 'ambiguous_negative' for v in records)):
            return dict(status='rejected', physical_U=None, located_cell_rows=ids, location_records=records, complete_requested_point_count=len(xyz), subset_prediction_allowed=False)
        u = self.api.query(xyz, ids, force6, support_translation_mm)
        assert bool(t.isfinite(u).all())
        return dict(status='completed', physical_U=u, located_cell_rows=ids, location_records=records, complete_requested_point_count=len(xyz), subset_prediction_allowed=False)
