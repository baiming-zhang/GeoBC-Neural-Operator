"""Independent two/three-RHS matrix-free stiffness lanes."""



class CanonicalLanesElastic:

    def __init__(self, torch, operator, guard):
        self.torch = torch
        self.original = operator
        self.guard = guard
        self.calls = 0

    def apply(self, u):
        torch = self.torch
        op = self.original
        assert u.ndim == 3 and u.shape[0] in (2, 3) and (u.shape[1:] == (op.node_count, 3))
        lanes = u.shape[0]
        result = torch.zeros_like(u)
        eye = torch.eye(3, dtype=u.dtype, device=u.device)
        for start in range(0, len(op.cells), 8192):
            self.guard()
            stop = min(start + 8192, len(op.cells))
            cells = op.cells[start:stop]
            grad = op.gradients[start:stop]
            weight = op.weights[start:stop]
            deriv = torch.einsum('knia,nqib->knqab', u[:, cells], grad)
            strain = 0.5 * (deriv + deriv.transpose(-1, -2))
            trace = strain.diagonal(dim1=-2, dim2=-1).sum(-1)
            stress = 2 * op.mu * strain + op.lam * trace[..., None, None] * eye
            local = torch.einsum('knqab,nqib,nq->knia', stress, grad, weight)
            result.index_add_(1, cells.reshape(-1), local.reshape(lanes, -1, 3))
        self.calls += 1
        self.guard()
        return result

class CanonicalLanesCoarse:
    """Unchanged fixed32 additive coarse math with two or three RHS columns."""

    def __init__(self, torch, scalar, guard):
        self.torch, self.scalar, self.guard, self.calls = (torch, scalar, guard, 0)

    def apply(self, value):
        t = self.torch
        c = self.scalar
        lanes = len(value)
        assert lanes in (2, 3) and value.shape == (lanes, c.P.shape[0], 3)
        self.guard()
        self.calls += 1
        fine = t.cholesky_solve(value[..., None], c.chol[None]).squeeze(-1)
        rhs = t.sparse.mm(c.PT, value.permute(1, 0, 2).reshape(c.P.shape[0], lanes * 3)).reshape(c.P.shape[1], lanes, 3).permute(0, 2, 1).reshape(-1, lanes)
        z = t.zeros_like(rhs)
        for _ in range(32):
            z = z + c.omega * (rhs - t.sparse.mm(c.Ac, z)) / c.diagonal[:, None]
        result = fine + t.sparse.mm(c.P, z.reshape(c.P.shape[1], 3, lanes).permute(0, 2, 1).reshape(c.P.shape[1], lanes * 3)).reshape(c.P.shape[0], lanes, 3).permute(1, 0, 2)
        self.guard()
        return result
