# Neural models

The default joint-field reference is a geometry-conditioned FNO with a pointwise decoder for three U components and the original signed S. The latest matched comparison is in results/neural_baselines/.

The next information experiment adds 12 physical covariates: three displacement components and nine volume-averaged displacement-gradient components multiplied by 1000. ZERO and ACTUAL inputs use the same enlarged decoder and initialization. The three-seed experiment improved mean U by 6.384 pp and signed S by 13.421 pp, with all three paired primary wins. The result is in results/physical_information_learning/.

The located U/S API accepts physical XYZ(mm) and a registered canonical load, with explicit outside-domain rejection and coordinate gradients. Its current compact-model checks are in results/compact_neural_response/.

The component experiment isolated U3 and gradient9 in two fresh models per seed, retaining the same capacity and 20-pass schedule. U-only inputs give 11.717% U / 48.652% S; gradient-only inputs give 17.058% U / 35.471% S. Displacement benefits most from U inputs, while stress retains the benefit from gradient inputs. See results/input_components/.

The preceding two-network route combined U from U_ONLY and S from GRAD_ONLY. The shared-trunk model replaces this route with a single 4,554,762-parameter network.

The compact `FieldwiseCovariateFNO` shares the encoder and FNO, while the U head consumes U3 and the S head consumes gradient9. Three fresh seeds with the same 20-pass schedule reached mean U 11.569% / S 34.819%, satisfying the within-1pp component-accuracy criterion. It has 4,554,762 parameters (+0.58% over the joint reference). The architecture was chosen from development results; these remain the same iterated 20 CADs. See results/shared_trunk_field_learning/.

`LocatedNeuralResponse` also serves the compact model. Its first-seed interface preserved saved nodal U/S predictions within 4.38e-7 / 3.14e-7 relative difference across 504 new physical-coordinate rows, with coordinate gradients and explicit outside-domain rejection. Total check cost was 31.85 s. See results/compact_neural_response/.

The architecture and all three trained seeds were fixed before final readout. On the fixed final20/60 set, FIELDWISE reaches U 11.629% / S 34.247%, versus ZERO 21.255% / 47.003%. The mean primary reduction is 11.191 pp with 3/3 paired seed wins, passing the predeclared primary and stress improvement criteria. Actual checkpoint/cache/inference/save/readout cost is 96.07 s UTC / 87.20 s monotonic; source preparation, physical-input generation and training are separate costs. See results/final_evaluation/. No final result is used to select a seed or modify the model.
