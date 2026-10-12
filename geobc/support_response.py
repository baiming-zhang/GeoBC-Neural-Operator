"""Force and four-port rigid-support response bases."""



def rigid_support_lift(t, vertices, ports, reference_xyz, motions):
    assert vertices.is_cuda and vertices.dtype == t.float64 and (vertices.ndim == 2) and (vertices.shape[1] == 3)
    assert reference_xyz.shape == (4, 3) and reference_xyz.is_cuda and (reference_xyz.dtype == t.float64)
    assert motions.shape == (3, 4, 6) and motions.is_cuda and (motions.dtype == t.float64)
    assert bool(t.isfinite(reference_xyz).all() and t.isfinite(motions).all())
    u = t.zeros((3, len(vertices), 3), dtype=t.float64, device='cuda')
    for j, rows in enumerate(ports):
        offset = vertices[rows] - reference_xyz[j]
        translation = motions[:, j, :3, None].transpose(1, 2)
        rotation = motions[:, j, 3:, None].transpose(1, 2).expand(-1, len(rows), -1)
        u[:, rows] = translation + t.cross(rotation, offset[None].expand(3, -1, -1), dim=-1)
    assert bool(t.isfinite(u).all())
    return u

class SupportRigidMotionResponse:

    def __init__(self, t, B6, force_basis, translation_basis, rotation_basis, source_reference_ids, translation_unit_mm, rotation_unit_rad, guard):
        self.t, self.B, self.force, self.translation, self.rotation, self.guard = (t, B6, force_basis, translation_basis, rotation_basis, guard)
        guard()
        self.tunit, self.runit = (translation_unit_mm, rotation_unit_rad)
        assert B6.is_cuda and B6.dtype == t.float64 and (B6.shape == (6, 6)) and (int(t.linalg.matrix_rank(B6)) == 6) and (float(t.linalg.cond(B6)) <= 1000000000000.0)
        assert force_basis.is_cuda and force_basis.dtype == t.float64 and (force_basis.ndim == 3) and (force_basis.shape[0] == 6) and (force_basis.shape[2] == 3)
        n = force_basis.shape[1]
        assert all((v.is_cuda and v.dtype == t.float64 and (v.shape == (4, 3, n, 3)) and bool(t.isfinite(v).all()) for v in (translation_basis, rotation_basis)))
        assert bool(t.isfinite(force_basis).all()) and translation_unit_mm > 0 and (rotation_unit_rad > 0)
        assert len(source_reference_ids) == 4 and source_reference_ids == sorted(set(source_reference_ids))
        self.metadata = dict(actual_allN=n, source_support_reference_ids=list(source_reference_ids), force_directions=6, support_translation_directions=12, support_rotation_directions=12, total_response_directions=30, rotation_units='rad', translation_units='mm', source_reference_XYZ_required=True, new_OptiStruct_GT_accuracy=False, signedS_provided=False)

    def query(self, force6, support_motion6):
        t = self.t
        self.guard()
        assert force6.is_cuda and force6.dtype == t.float64 and (force6.ndim == 2) and (force6.shape[1] == 6)
        assert support_motion6.is_cuda and support_motion6.dtype == t.float64 and (support_motion6.shape == (len(force6), 4, 6))
        assert bool(t.isfinite(force6).all() and t.isfinite(support_motion6).all())
        c = t.linalg.solve(self.B, force6.T).T
        u = t.einsum('qk,kna->qna', c, self.force) + t.einsum('qpk,pkna->qna', support_motion6[:, :, :3] / self.tunit, self.translation)
        u = u + t.einsum('qpk,pkna->qna', support_motion6[:, :, 3:] / self.runit, self.rotation)
        assert bool(t.isfinite(u).all())
        return dict(physical_U_mm=u, independent_modified_BC_validation_required=True)
