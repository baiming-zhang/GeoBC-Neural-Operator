"""Continuous queries of recovered nodal physical covariates."""

from .query import PhysicalUQuery

class KinematicCovariateQuery:

    def __init__(self, torch, vertices, cells, inputs12, reference_load6, guard):
        assert inputs12.dtype == torch.float64 and inputs12.shape == (3, len(vertices), 12)
        assert inputs12.is_cuda and bool(torch.isfinite(inputs12).all())
        self.torch, self.inputs12, self.guard = (torch, inputs12, guard)
        self.reference = PhysicalUQuery(torch, vertices, cells, inputs12[..., :3], reference_load6, guard)
        self.cells = cells
        self.metadata = dict(physical_U_units='mm', gradient_covariate_units='fixed1000_times_mm_per_mm', canonical_load_rank=3, coordinate_input_units='mm', explicit_valid_original_cell_required=True, gradient_channels='P2 interpolation of incident-volume averaged nodal derivative covariates', independent_offnode_FEM_gradient_accuracy_proven=False, signed_stress_target_or_rule_generated=False)

    def query(self, physical_xyz, cell_rows, physical_force6):
        N, D, b, nodes = self.reference.invert(physical_xyz, cell_rows)
        coefficients = self.reference.coefficients(physical_force6)
        values = self.torch.einsum('qi,kqia->qka', N, self.inputs12[:, self.cells[cell_rows]])
        result = self.torch.einsum('qk,qka->qa', coefficients, values)
        assert result.shape == (len(physical_xyz), 12) and bool(self.torch.isfinite(result).all())
        self.guard()
        return result
