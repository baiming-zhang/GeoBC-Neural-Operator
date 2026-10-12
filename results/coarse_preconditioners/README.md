# Coarse preconditioner comparison

On fixed training CAD 80_483, the degree-32 Chebyshev candidate met the precision and 25% cost-reduction criteria. Paired Jacobi solve: 56.595s; Chebyshev construction/checks plus solve: 22.179s (60.81% reduction).

Both methods use the same FP64 curved-P2 equation and three independent zero-start loads. Shared geometry and coarse assembly costs are separate from the paired comparison. Six fields were saved before the same-equation displacement comparison; no target labels were read.

The first attempt stopped after a statistics-key error, before any PCG field. Its cost and the unsaved coarse reconstruction remain included in campaign accounting; saved stiffness was reused. This single-CAD result does not establish full-training-set cost or original stress accuracy.
