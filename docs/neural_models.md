# Neural models

The default joint-field reference is a geometry-conditioned FNO with a pointwise decoder for three U components and the original signed S. The latest matched comparison is in results/neural_baselines/.

The next information experiment adds 12 physical covariates: three displacement components and nine volume-averaged displacement-gradient components multiplied by 1000. ZERO and ACTUAL inputs use the same enlarged decoder and initialization. The three-seed experiment improved mean U by 6.384 pp and signed S by 13.421 pp, with all three paired primary wins. The result is in results/physical_information_learning/.
