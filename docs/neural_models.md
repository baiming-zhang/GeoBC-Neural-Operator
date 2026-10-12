# Neural models

The default joint-field reference is a geometry-conditioned FNO with a pointwise decoder for three U components and the original signed S. The latest matched comparison is in results/neural_baselines/.

The next information experiment adds 12 physical covariates: three displacement components and nine volume-averaged displacement-gradient components multiplied by 1000. ZERO and ACTUAL inputs use the same enlarged decoder and initialization. The three-seed experiment improved mean U by 6.384 pp and signed S by 13.421 pp, with all three paired primary wins. The result is in results/physical_information_learning/.

The located U/S API accepts physical XYZ(mm) and a registered canonical load. Its three-CAD check preserves nodal trained-model predictions within 6.2e-7 relative difference and keeps actual coordinate gradients. See results/continuous_neural_response/.

The next component experiment isolates U3 and gradient9 in two fresh models per seed, retaining the same capacity and 20-pass schedule. Existing ZERO and joint ACTUAL results remain frozen references.
