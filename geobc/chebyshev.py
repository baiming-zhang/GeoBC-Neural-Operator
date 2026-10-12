"""Fixed-degree Chebyshev coarse correction for a symmetric scaled operator."""
import math
from .coarse import AdditiveCoarse


def inverse_polynomial(apply, rhs, upper, degree=32, lower_ratio=1/30):
    """Apply p(A) where 1-lambda*p(lambda)=T_n((c-lambda)/d)/T_n(c/d)."""
    if not math.isfinite(upper) or upper <= 0 or degree < 1 or not 0 < lower_ratio < 1:
        raise ValueError('Invalid polynomial interval')
    lower = upper*lower_ratio
    center, radius = (upper+lower)/2, (upper-lower)/2
    sigma = center/radius
    rho_old = 1/sigma
    direction = rhs/center
    result = direction.clone()
    for _ in range(1, degree):
        rho = 1/(2*sigma-rho_old)
        direction = rho*rho_old*direction + (2*rho/radius)*(rhs-apply(result))
        result = result+direction
        rho_old = rho
    return result


class ChebyshevCoarse(AdditiveCoarse):
    def __init__(self, torch, P, PT, Ac, diagonal, gamma, fine_cholesky, ledger):
        super().__init__(torch, P, PT, Ac, diagonal, gamma, fine_cholesky, ledger)
        self.sqrt_diagonal = diagonal.sqrt()
        self.upper = gamma
        self.degree = 32
        self.lower_ratio = 1/30
        self.calls = 0

    def _inverse(self, rhs):
        scale = self.sqrt_diagonal[:, None]
        def apply(value):
            self.ledger.check()
            self.ledger.data['coarse_matvec_calls'] += 1
            return self.torch.sparse.mm(self.Ac, value/scale)/scale
        value = inverse_polynomial(apply, rhs/scale, self.upper, self.degree, self.lower_ratio)
        return value/scale

    def apply(self, value):
        self.ledger.check()
        t = self.torch
        fine = t.cholesky_solve(value[..., None], self.chol).squeeze(-1)
        rhs = self.restrict(value).reshape(-1, 1)
        correction = self._inverse(rhs).reshape(-1, 3)
        result = fine+self.prolong(correction)
        self.calls += 1
        self.ledger.check()
        return result

    def apply_lanes(self, value):
        t = self.torch
        self.ledger.check()
        lanes = len(value)
        fine = t.cholesky_solve(value[..., None], self.chol[None]).squeeze(-1)
        rhs = t.sparse.mm(self.PT, value.permute(1, 0, 2).reshape(self.P.shape[0], lanes*3))
        rhs = rhs.reshape(self.P.shape[1], lanes, 3).permute(0, 2, 1).reshape(-1, lanes)
        correction = self._inverse(rhs)
        correction = correction.reshape(self.P.shape[1], 3, lanes).permute(0, 2, 1).reshape(self.P.shape[1], lanes*3)
        result = fine+t.sparse.mm(self.P, correction).reshape(self.P.shape[0], lanes, 3).permute(1, 0, 2)
        self.calls += 1
        self.ledger.check()
        return result
