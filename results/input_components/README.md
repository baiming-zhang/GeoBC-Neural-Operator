# Displacement and gradient input contributions

This is the input-group ablation between [matched physical-information learning](../physical_information_learning/README.md) and [shared-trunk Fieldwise design](../shared_trunk_field_learning/README.md). It motivates output-specific covariate routing; it does not establish universal architectural necessity or directly identify a gradient-conflict cause.

Three fresh matched seeds isolate U3 and gradient9, with the same FNO, initialization, original training40 scaler and 20 training passes. ZERO and ACTUAL are frozen references from the previous experiment.

| Input | Mean U error | Mean S error | Mean primary |
|---|---:|---:|---:|
| ZERO | 21.746% | 49.419% | 35.582% |
| ACTUAL | 15.361% | 35.998% | 25.680% |
| U_ONLY | 11.717% | 48.652% | 30.185% |
| GRAD_ONLY | 17.058% | 35.471% | 26.265% |

U-only inputs improve U by 3.644 percentage points over joint inputs; gradient-only inputs retain the joint S accuracy (0.527 points lower error). This motivates routing U from the displacement-input model and S from the gradient-input model. That routing is selected from development results and uses two networks; its combined mean score inferred from the separate field errors is 23.594%, rather than a new validation result.

New trial cost: UTC 4293.857s / monotonic 3936.816s. All 360 new predictions were saved before shared original development GT60 reads. The 20 development CADs are iterated; no final data was used in this trial.
