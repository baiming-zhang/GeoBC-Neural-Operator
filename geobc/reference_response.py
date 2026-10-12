"""Reference motion, grounded springs and prescribed motion."""



class ReferenceMotionResponse:

    def __init__(self, t, Hbar, members, length_mm, basis, B, guard):
        self.t, self.guard = (t, guard)
        guard()
        assert Hbar.is_cuda and members.is_cuda and basis.is_cuda and B.is_cuda
        assert Hbar.dtype == basis.dtype == B.dtype == t.float64 and members.dtype == t.int64
        assert basis.ndim == 3 and basis.shape[0] == 6 and (basis.shape[2] == 3)
        assert members.ndim == 1 and Hbar.shape == (6, 3 * len(members)) and (B.shape == (6, 6))
        assert length_mm > 0 and bool(((members >= 0) & (members < basis.shape[1])).all())
        assert members.unique().numel() == len(members) and bool(t.isfinite(Hbar).all() and t.isfinite(basis).all() and t.isfinite(B).all())
        assert int(t.linalg.matrix_rank(B)) == 6 and float(t.linalg.cond(B)) <= 1000000000000.0
        scales = t.tensor([1.0, 1.0, 1.0, length_mm, length_mm, length_mm], dtype=t.float64, device=B.device)
        scaled_motion_columns = Hbar @ basis[:, members].reshape(6, -1).T
        physical_motion_columns = scaled_motion_columns / scales[:, None]
        C = t.linalg.solve(B.T, physical_motion_columns.T).T
        normalized_C = scales[:, None] * C * scales[None, :]
        norm = float(t.linalg.vector_norm(normalized_C))
        assert norm > 0
        reciprocity = float(t.linalg.vector_norm(normalized_C - normalized_C.T)) / norm
        symmetric_part = (normalized_C + normalized_C.T) * 0.5
        eigenvalues = t.linalg.eigvalsh(symmetric_part)
        minimum = float(eigenvalues.min())
        maximum = float(eigenvalues.max())
        self.C, self.motion_columns, self.scales = (C, physical_motion_columns, scales)
        self.metadata = dict(actual_allN=basis.shape[1], actual_member_nodes=len(members), physical_load_rank=6, reference_translation_units='mm', reference_rotation_units='rad', energy_units='Nmm', load_units=['N', 'N', 'N', 'Nmm', 'Nmm', 'Nmm'], reciprocity_relative=reciprocity, energy_normalized_minimum_eigenvalue=minimum, energy_normalized_maximum_eigenvalue=maximum, energy_positive=minimum > 0, energy_normalized_condition=maximum / minimum if minimum > 0 else None, raw_C_preserved=True, symmetric_part_only_diagnostic=True, source_transfer='accepted_weightedLS_RBE3_Hbar', independent_OptiStruct_reference_rotation_GT=False, signedS_provided=False)
        guard()

    def query_force6(self, forces):
        t = self.t
        self.guard()
        assert forces.is_cuda and forces.dtype == t.float64 and (forces.ndim == 2) and (forces.shape[1] == 6) and bool(t.isfinite(forces).all())
        motion = forces @ self.C.T
        energy = 0.5 * (forces * motion).sum(-1)
        assert bool(t.isfinite(motion).all() and t.isfinite(energy).all())
        return dict(reference_motion6=motion, strain_energy_Nmm=energy, motion_units=['mm', 'mm', 'mm', 'rad', 'rad', 'rad'], independent_original_GT_accuracy=False)

class ReferenceSpringResponse:

    def __init__(self, t, reference_api, guard):
        self.t, self.reference_api, self.guard = (t, reference_api, guard)
        guard()
        C, D = (reference_api.C, reference_api.scales)
        assert C.is_cuda and C.dtype == t.float64 and (C.shape == (6, 6))
        Cn = D[:, None] * C * D[None, :]
        mean = float(Cn.diagonal().mean())
        assert mean > 0
        self.spring_diagonal = t.full((6,), 1.0 / mean, dtype=C.dtype, device=C.device)
        self.matrix = t.eye(6, dtype=C.dtype, device=C.device) + Cn * self.spring_diagonal[None, :]
        self.normalized_C = Cn
        assert bool(t.isfinite(self.matrix).all()) and float(t.linalg.cond(self.matrix)) <= 1000000000000.0
        self.metadata = dict(normalized_spring_units='N/mm', physical_spring_scope='Grounded loaded RBE reference,6work-conjugate directions', fixed_strength='1 divided by mean diagonal normalized raw compliance', raw_C_unchanged=True, matrix_condition=float(t.linalg.cond(self.matrix)), new_original_OptiStruct_GT=False, signedS_provided=False)

    def query_force6(self, forces):
        t = self.t
        self.guard()
        D = self.reference_api.scales
        assert forces.is_cuda and forces.dtype == t.float64 and (forces.ndim == 2) and (forces.shape[1] == 6) and bool(t.isfinite(forces).all())
        p = forces / D
        q = t.linalg.solve(self.matrix, (p @ self.normalized_C.T).T).T
        physical = q / D
        effective = forces - q * self.spring_diagonal * D
        spring_energy = 0.5 * (q.square() * self.spring_diagonal).sum(-1)
        total_energy = 0.5 * (forces * physical).sum(-1)
        assert bool(t.isfinite(q).all() and t.isfinite(effective).all() and t.isfinite(total_energy).all())
        return dict(reference_motion6=physical, effective_original_force6=effective, spring_energy_Nmm=spring_energy, total_strain_energy_Nmm=total_energy, independent_PCGequation_validation_required=True)

class PrescribedReferenceMotion:

    def __init__(self, t, reference_api, guard):
        self.t, self.reference_api, self.guard = (t, reference_api, guard)
        guard()
        D = reference_api.scales
        C = reference_api.C
        assert C.is_cuda and C.dtype == t.float64 and (C.shape == (6, 6))
        self.normalized_C = D[:, None] * C * D[None, :]
        assert bool(t.isfinite(self.normalized_C).all()) and float(t.linalg.cond(self.normalized_C)) <= 1000000000000.0
        self.metadata = dict(boundary='Exact Hbar u_members = D physical target motion', raw_C_unchanged=True, large_spring_limit_used=False, new_original_OptiStruct_GT=False, signedS_provided=False)

    def query_motion6(self, motions, applied_force6):
        t = self.t
        self.guard()
        D = self.reference_api.scales
        assert motions.is_cuda and applied_force6.is_cuda and (motions.dtype == applied_force6.dtype == t.float64)
        assert motions.ndim == 2 and motions.shape[1] == 6 and (applied_force6.shape == motions.shape)
        assert bool(t.isfinite(motions).all() and t.isfinite(applied_force6).all())
        q = motions * D
        total_normalized_force = t.linalg.solve(self.normalized_C, q.T).T
        total_physical_force = total_normalized_force * D
        reaction = total_physical_force - applied_force6
        energy = 0.5 * (total_physical_force * motions).sum(-1)
        assert bool(t.isfinite(total_physical_force).all() and t.isfinite(energy).all())
        return dict(effective_original_force6=total_physical_force, reference_reaction_force6=reaction, body_strain_energy_Nmm=energy, reference_motion6=motions, independent_constraint_PCGequation_validation_required=True)
