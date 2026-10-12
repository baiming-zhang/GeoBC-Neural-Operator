# GeoBC Neural Operator

Geometry, loads and boundary conditions for displacement and signed-stress prediction on original DeepJEB.

The physical displacement route reached **0.01589% mean error** and **0.02621% worst error** across 20 known development CADs and three loads. The joint U/S neural baseline remains the smaller FNO; signed stress is the main unresolved accuracy bottleneck.

Physical U and full-gradient inputs are complete for **100/100 training CADs and 300 loads**. The final stage reused 95 qualified inputs and completed the remaining five in **114.51 s UTC / 104.26 s monotonic**. Development inputs are also complete for **20/20 CADs and 60 loads** (62.40 s UTC / 56.49 s monotonic). The three-seed matched FNO experiment is complete: actual physical inputs reduced mean U error from **21.746% to 15.361%** and signed S error from **49.419% to 35.998%**. All three seeds improved the joint score. This hybrid route uses finite-element inputs in addition to the original neural inputs.

A fixed comparison on the next unused training CAD reduced three-load solve cost from **56.60 s to 22.18 s** with a degree-32 Chebyshev coarse correction, including its construction/checks: **60.81% lower cost**. All six true residuals met 1e-8 and same-equation U differed by at most 1.23e-10. The changed method completed the remaining inputs; the original budget-limited run remains a separate cost.

## Code

- `geobc/neural_field.py`: geometry-conditioned FNO and U/S decoder.
- `geobc/elasticity.py`, `geobc/coarse.py`: curved-tetrahedron physical operator and coarse correction.
- `geobc/query.py`, `geobc/locator.py`: continuous interpolation and point location.
- `geobc/support_response.py`, `geobc/reference_response.py`: load and boundary response bases.
- `geobc/kinematics.py`: physical U and full-gradient input recovery.
- `geobc/kinematic_neural_field.py`: prepared ZERO/ACTUAL input comparison.

## Results

- [Physical-information learning](results/physical_information_learning/README.md)
- [Continuous learned U/S response](results/continuous_neural_response/README.md)
- [Neural baselines](results/neural_baselines/README.md)
- [Development displacement accuracy](results/displacement_accuracy/README.md)
- [Reserve displacement accuracy](results/reserve_displacement_accuracy/README.md)
- [Support motion](results/support_motion/README.md)
- [Continuous boundary response](results/continuous_boundary_response/README.md)
- [Training sources](results/training_sources/README.md)
- [Latest physical inputs](results/complete_training_inputs/README.md)
- [Coarse preconditioner comparison](results/coarse_preconditioners/README.md)

Install with `pip install -r requirements.txt` and `pip install -e .`. GPU physics uses FP64 CUDA. Raw data and generated arrays stay local.

[Data protocol](docs/data.md) 路 [Physical model](docs/physics.md) 路 [Neural models](docs/neural_models.md)
[Signed stress definitions](docs/stress_labels.md)
