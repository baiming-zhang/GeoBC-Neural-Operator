# GeoBC Neural Operator

Geometry, loads and boundary conditions for displacement and signed-stress prediction on original DeepJEB.

The [paper evidence chain](docs/paper_logic.md) separates **historical backbone/package selection → matched physical-information gains → input-group ablation and compact Fieldwise design**, followed by frozen final evaluation. Historical Transolver results are preserved in [backbone_comparison](results/backbone_comparison/README.md), separately from the latest three-seed physically enhanced experiments. The [five original historical reports](results/research_history/README.md) are retained unchanged as the source layer.

The frozen compact hybrid model reaches **11.629% mean U error / 34.247% signed S error** across three seeds on the fixed 20 final CADs and three loads. Its matched zero-physical-input control reaches 21.255% / 47.003%; the mean joint score improves by 11.191 percentage points, with all three paired seed wins. All 360 full-node predictions were saved before one shared final-label readout. The predeclared practical-benefit criterion passed, without final-set tuning.

The model shares one geometry/FNO trunk and uses field-specific input heads. Its 4,554,762 parameters retain the component models' development accuracy within the predeclared 1-point margin, using about half the two-network route's parameters. This is an identity-disjoint, availability-conditional evaluation within the official training pool; shape-family separation is not established. Signed stress remains substantially less accurate than displacement.

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
- `geobc/fieldwise_neural_field.py`: compact shared-trunk U/S model.
- `geobc/deployment.py`, `tools/query_response.py`: one-model prepared-cache XYZ request session and JSONL command.

[Query usage](docs/response_usage.md) describes the private prepared catalog, physical units and explicit rejection behavior.

## Results

- [Historical backbone/package comparison](results/backbone_comparison/README.md): FNO, original Transolver, Transolver++, Transolver3 and MLP; separate experiment groups with split, capacity, precision and cost scope. Archival selection evidence, not the latest final-set ranking.
- [Original research reports](results/research_history/README.md): 30 early experiments; geometry/BC/kinematics exploration; FNO/Transolver comparison; common-grid capacity control; physical-solver research history. Original versions are dated historical records.
- [Saved-model field maps](results/field_comparison/README.md): displacement, original signed stress and RMS-normalized spatial errors for ZERO and FIELDWISE, using the original presentation's layout and palette.
- [Fixed final-set model evaluation](results/final_evaluation/README.md): checkpoint/cache/inference/save/readout cost 96.07 s UTC / 87.20 s monotonic. Physical inputs and training are charged separately.
- [Prepared coordinate-response delivery](results/prepared_response_delivery/README.md): all 20 final CADs, 1,200 new interior responses and 60 outside refusals; one first-seed model load, 70.90 s UTC / 64.99 s monotonic, no GT or training.
- [Displacement and gradient contributions](results/input_components/README.md): U-only inputs reach 11.717% U error; gradient-only inputs reach 35.471% S error across the same three seeds and 20 development CADs.
- [Shared-trunk field-specific learning](results/shared_trunk_field_learning/README.md)
- [Compact continuous U/S response](results/compact_neural_response/README.md): 504 new query rows passed using the first registered shared-trunk seed.
- [Final-set target-free sources](results/final_sources/README.md): all 20 fixed CADs have qualified geometry and three canonical loads, produced before the label readout.
- [Final-set physical inputs](results/final_physical_inputs/README.md): all 20 CADs and 60 loads passed the full-node checks in 613.90 s UTC / 562.01 s monotonic, without reading target fields.
- [Physical-information learning](results/physical_information_learning/README.md)
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
