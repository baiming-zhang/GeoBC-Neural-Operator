# Coarse polynomial correction

Let S = D^(-1/2) Ac D^(-1/2), where Ac is the existing masked Galerkin coarse matrix. Its Gershgorin row-sum bound is Gamma. The fixed candidate uses a degree-32 Chebyshev polynomial with interval [Gamma/30, Gamma]. The lower endpoint is a smoothing interval, not an estimate proved to bound the smallest eigenvalue.

For c=(a+b)/2 and d=(b-a)/2, the polynomial obeys:

`1 - lambda*p(lambda) = T_32((c-lambda)/d) / T_32(c/d)`.

The inverse action is D^(-1/2) p(S) D^(-1/2), added to the unchanged fine block-Jacobi action. It is linear, starts at zero, and uses 31 sparse coarse matrix applications for degree 32. Geometry, material, constraints and the fine equation stay the same. The actual comparison checks full-node lane parity, positive energies, symmetry and independent zero-start PCG residuals.

[PETSc's Chebyshev notes](https://petsc.org/release/manualpages/KSP/KSPCHEBYSHEV/) describe its SPD assumptions; [eigenvalue-interval documentation](https://petsc.org/release/manualpages/KSP/KSPChebyshevSetEigenvalues/) distinguishes smoothing intervals from complete spectral bounds.
