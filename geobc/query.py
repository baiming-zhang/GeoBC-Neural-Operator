"""Curved TET10 interpolation and force-to-displacement queries."""



def shape(torch, r):
    b = torch.cat((r, 1 - r.sum(-1, keepdim=True)), -1)
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    N = torch.cat((b * (2 * b - 1), torch.stack([4 * b[..., i] * b[..., j] for i, j in edges], -1)), -1)
    db = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]], device=r.device, dtype=r.dtype)
    D = torch.cat(((4 * b - 1)[..., None] * db, torch.stack([4 * (b[..., i, None] * db[j] + b[..., j, None] * db[i]) for i, j in edges], -2)), -2)
    return (N, D, b)

class PhysicalUQuery:

    def __init__(self, torch, vertices, cells, basis, load6, guard):
        self.torch, self.vertices, self.cells, self.basis, self.guard = (torch, vertices, cells, basis, guard)
        assert vertices.dtype == basis.dtype == load6.dtype == torch.float64 and cells.dtype == torch.int64
        assert vertices.ndim == 2 and vertices.shape[1] == 3 and (cells.ndim == 2) and (cells.shape[1] == 10)
        assert basis.shape == (3, len(vertices), 3) and load6.shape == (6, 3)
        assert all((torch.isfinite(v).all() for v in (vertices, basis, load6))) and int(cells.min()) >= 0 and (int(cells.max()) < len(vertices))
        singular = torch.linalg.svdvals(load6)
        assert (singular > 0).all() and float(singular[0] / singular[-1]) <= 1000000000000.0
        self.B = load6
        self.gram = load6.T @ load6
        self.metadata = dict(actual_allN=len(vertices), actual_cells=len(cells), canonical_load_rank=3, canonical_load_condition=float(singular[0] / singular[-1]), physical_U_units='mm', model_or_GT_reads=0, global_locator=False, C1_claim=False)
        guard()

    def coefficients(self, load):
        t = self.torch
        assert load.ndim == 2 and load.shape[-1] == 6 and (load.dtype == t.float64) and t.isfinite(load).all()
        self.guard()
        c = t.linalg.solve(self.gram, (load @ self.B).T).T
        residual = t.linalg.vector_norm(load - c @ self.B.T, dim=-1)
        norm = t.linalg.vector_norm(load, dim=-1)
        good = t.where(norm > 0, residual <= 2e-06 * norm, residual == 0)
        if not bool(good.all()):
            raise ValueError('Requested load outside actual rank3 canonical span')
        return c

    def invert(self, xyz, element_rows):
        t = self.torch
        assert xyz.dtype == t.float64 and xyz.ndim == 2 and (xyz.shape[1] == 3) and (element_rows.shape == (len(xyz),)) and (element_rows.dtype == t.int64)
        assert t.isfinite(xyz).all() and int(element_rows.min()) >= 0 and (int(element_rows.max()) < len(self.cells))
        nodes = self.vertices[self.cells[element_rows]]
        scale = t.linalg.vector_norm(nodes.amax(1) - nodes.amin(1), dim=-1)
        assert (scale > 0).all()
        r = t.full_like(xyz, 0.25)
        orientation = None
        for _ in range(16):
            self.guard()
            N, D, b = shape(t, r)
            mapped = t.einsum('qi,qia->qa', N, nodes)
            J = t.einsum('qia,qib->qab', nodes, D)
            sv = t.linalg.svdvals(J)
            det = t.linalg.det(J)
            if not bool(t.isfinite(J).all() and (sv[:, -1] > 0).all() and (sv[:, 0] / sv[:, -1] <= 1000000000000.0).all() and (det.abs() / scale ** 3 > 1e-14).all()):
                raise ValueError('Invalid curved geometry for explicit query')
            if orientation is None:
                orientation = det.sign()
            elif not bool((det.sign() == orientation).all()):
                raise ValueError('Curved query iteration crossed invalid orientation')
            r = r - t.linalg.solve(J, (mapped - xyz)[..., None]).squeeze(-1)
            if not bool(t.isfinite(r).all()):
                raise ValueError('Nonfinite bounded query inversion')
        N, D, b = shape(t, r)
        mapped = t.einsum('qi,qia->qa', N, nodes)
        relative = t.linalg.vector_norm(mapped - xyz, dim=-1) / scale
        if not bool((relative <= 1e-09).all() and (b >= -1e-10).all() and (b <= 1 + 1e-10).all()):
            raise ValueError('Explicit query not inside selected cell or inversion residual failed')
        return (N, D, b, nodes)

    def basis_at(self, xyz, element_rows):
        N, D, b, nodes = self.invert(xyz, element_rows)
        values = self.torch.einsum('qi,kqia->qka', N, self.basis[:, self.cells[element_rows]])
        assert self.torch.isfinite(values).all()
        return values

    def query(self, xyz, element_rows, load6):
        values = self.basis_at(xyz, element_rows)
        c = self.coefficients(load6)
        assert c.shape == (len(xyz), 3)
        return self.torch.einsum('qk,qka->qa', c, values)

    def query_coefficients(self, xyz, element_rows, c):
        assert c.shape == (len(xyz), 3) and c.dtype == self.torch.float64 and self.torch.isfinite(c).all()
        return self.query(xyz, element_rows, c @ self.B.T)

class FullRank6PhysicalUQuery(PhysicalUQuery):

    def __init__(self, torch, vertices, cells, basis, load6, guard):
        self.torch, self.vertices, self.cells, self.basis, self.B, self.guard = (torch, vertices, cells, basis, load6, guard)
        t = torch
        guard()
        assert vertices.is_cuda and cells.is_cuda and basis.is_cuda and load6.is_cuda
        assert vertices.dtype == basis.dtype == load6.dtype == t.float64 and cells.dtype == t.int64
        assert vertices.ndim == 2 and vertices.shape[1] == 3 and (cells.ndim == 2) and (cells.shape[1] == 10)
        assert basis.shape == (6, len(vertices), 3) and load6.shape == (6, 6)
        assert bool(t.isfinite(vertices).all() and t.isfinite(basis).all() and t.isfinite(load6).all())
        assert bool(((cells >= 0) & (cells < len(vertices))).all())
        rank = int(t.linalg.matrix_rank(load6))
        condition = float(t.linalg.cond(load6))
        if rank != 6 or condition > 1000000000000.0:
            raise ValueError('Six physical load directions are not independent')
        self.metadata = dict(actual_allN=len(vertices), actual_cells=len(cells), physical_load_rank=rank, physical_load_condition=condition, physical_U_units='mm', force_units='N', moment_units='Nmm', scope='same_geometry_material_constraints_RBEreference', new_OptiStruct_GT_accuracy=False, C1_claim=False)

    def coefficients(self, load6):
        t = self.torch
        self.guard()
        assert load6.dtype == t.float64 and load6.ndim == 2 and (load6.shape[1] == 6) and bool(t.isfinite(load6).all())
        return t.linalg.solve(self.B, load6.T).T

    def query(self, xyz, element_rows, load6):
        values = self.basis_at(xyz, element_rows)
        c = self.coefficients(load6)
        assert c.shape == (len(xyz), 6)
        return self.torch.einsum('qk,qka->qa', c, values)

    def query_coefficients(self, xyz, element_rows, c):
        assert c.shape == (len(xyz), 6) and c.dtype == self.torch.float64 and bool(self.torch.isfinite(c).all())
        return self.query(xyz, element_rows, c @ self.B.T)
