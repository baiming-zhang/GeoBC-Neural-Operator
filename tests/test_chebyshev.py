import math
from decimal import Decimal, localcontext
import unittest
import torch
from geobc.chebyshev import inverse_polynomial


class ChebyshevTests(unittest.TestCase):
    def test_dense_spectral_reference(self):
        eigenvalues = torch.tensor([1e-6, .001, .1, 1., 3.], dtype=torch.float64)
        rhs = torch.arange(1., 16., dtype=torch.float64).reshape(5, 3)
        actual = inverse_polynomial(lambda x: eigenvalues[:, None]*x, rhs, 3.)
        def chebyshev(x):
            a, b = Decimal(1), x
            for _ in range(1, 32):
                a, b = b, 2*x*b-a
            return b
        # High precision avoids cancellation in 1-T_n(x)/T_n(sigma) near lambda=0.
        with localcontext() as context:
            context.prec = 60
            center, radius = Decimal('1.55'), Decimal('1.45')
            denominator = chebyshev(center/radius)
            reference = torch.tensor([float((1-chebyshev((center-Decimal(str(value)))/radius)/denominator)/Decimal(str(value)))
                                      for value in eigenvalues.tolist()], dtype=torch.float64)
        torch.testing.assert_close(actual, rhs*reference[:, None], rtol=1e-10, atol=1e-9)
        self.assertTrue(bool((reference > 0).all()))

    def test_zero_start_linearity(self):
        diagonal = torch.tensor([.1, .5, 1.], dtype=torch.float64)
        apply = lambda x: diagonal[:, None]*x
        a, b = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 2, dtype=torch.float64)
        torch.testing.assert_close(inverse_polynomial(apply, 2*a-b, 1.),
                                   2*inverse_polynomial(apply, a, 1.)-inverse_polynomial(apply, b, 1.))


if __name__ == '__main__':
    unittest.main()
