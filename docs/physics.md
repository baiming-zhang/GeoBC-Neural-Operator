# Physical response

The displacement solver uses FP64 curved TET10 elements, four quadrature points and a matrix-free stiffness operator. The preconditioner adds block Jacobi and a constraint-masked Galerkin coarse correction with 32 fixed Jacobi steps. Each load starts from zero and stops at a true relative residual of 1e-8.

Reusable bases cover six force/moment directions and translations and infinitesimal rotations at four RBE2 supports. Reference motion, grounded springs and prescribed motion are separate APIs. Continuous queries use a Bezier-hull index and a bounded curved-cell inverse.

This is an approximate P2/RBE3 model. The original DEBUG/PSOLID numerical settings and signed stress export definition remain unresolved. Current tests establish original-node U accuracy and same-equation interface consistency, not new OptiStruct off-node truth.
