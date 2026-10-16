# GeoBC Neural Operator

Geometry, loads and boundary conditions for displacement and signed-stress prediction on original DeepJEB.

The [latest **1 / 5 / 50-case U-only comparison**](results/fifty_u_overfitting/README.md) is complete: **all 24 arms have exactly 5,000 full original-node updates**, common OneCycle peak **1e-3**, and approximately 4.6 million registered/active parameters (±1%). The [50 distinct TRAIN CADs](configs/fifty_u_samples.json), one load each, include the original five. Each receives 100 updates, versus 1,000 per case in the five-case group. FNO has the lowest 50-case mean/worst U error in this run, **9.9989% / 17.0790%**; the other means are 21.3929%–29.5333%. No 50-case endpoint is below 1% U error. These are fixed-seed fitting results, not generalization or architecture-limit claims.

The [24-arm summary](results/fifty_u_overfitting/all_methods.csv), [448 case errors](results/fifty_u_overfitting/all_per_case.csv), [complete curves](results/fifty_u_overfitting/learning_curves.png), and [same-five-case comparison](results/fifty_u_overfitting/same_five_cases.csv) retain the existing 1/5-case scalars and add eight new 50-case endpoints. Transolver++ hit the memory cap after 21 complete updates, then restored its model, Adam, scheduler, four RNG states and cursor for the remaining 4,979 updates, using CPU data residency. Both stages exited 0 within unchanged deadlines. New 50-case work cost **15,030.74 s UTC / 14,743.00 s monotonic**; source-case reads total 100 including recovery, stress/held-out reads zero. Scientific arrays and checkpoints remain local.

The retained [single/five-case comparison](results/fullcycle_u_overfitting/README.md) has **16 arms with exactly 5,000 full original-node updates**, approximately 4.6 million registered and active parameters (±1%), seed 2026, and the same Adam / OneCycle peak **1e-3** / gradient clipping recipe. All eight single-case endpoints are below 1% U error. TorchHash reaches **0.002353%** on the single case and is the only five-case implementation with every case below 1%: **0.6831% mean / 0.7506% worst U error**.

Transolver / Transolver++ / TS3 now reach **0.3628% / 0.3549% / 0.3971%** single-case U error, and **3.2076% / 2.7529% / 3.2396%** five-case mean error. Their five-case worst errors are 4.4722% / 3.4265% / 4.4200%. MLP and NGP4 five-case means are 4.7428% and 4.4985%; completing the cycle improves them relative to the interrupted early-stop endpoints, but neither meets every-case 1%. These are fixed-seed fitting results, not generalization results or architecture limits.

The [16-arm summary](results/fullcycle_u_overfitting/summary.csv), [48 case errors](results/fullcycle_u_overfitting/per_case.csv), and [complete learning curves](results/fullcycle_u_overfitting/learning_curves.png) use only actual update-5,000 endpoints. Four early-stopped arms were rerun fresh with early stopping disabled; twelve existing full endpoints were retained. Both stages exited successfully within their declared budgets, costing **14,412.99 s UTC / 14,202.69 s monotonic** together, including discarded early-stop work. Each stage reads the same five TRAIN caches once; stress-label and held-out reads are zero. The [earlier 1,000-update Transolver reruns](results/stabilized_u_overfitting/README.md) and [fixed .001 comparison](results/matched_capacity_u_overfitting/README.md) remain historical records.

The earlier [joint U/S matched comparison](results/matched_capacity_overfitting/README.md) and [unequal-capacity fitting with extended Hash training](results/sample_overfitting/README.md) remain separate. Removing S loss does not improve every fixed recipe; the U-only and joint runs also have different output heads and initialization.

The [Transolver correctness and learning-rate diagnostic](results/transolver_fit_diagnostics/README.md) found identical native/adapter forward results, gradients and one-step Adam updates for all three fixed backbones. Original Transolver's 76.09% single-case error is close to the 76.15% best-constant baseline. Changing only its fixed learning rate from 1e-3 to 1e-4, with the same seed and 1,000 updates, reduces that error to 7.99%; it still misses 1%. The earlier fixed-recipe results do not establish that Transolver cannot fit the sample.

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

- [Latest 1/5/50-case U-only comparison](results/fifty_u_overfitting/README.md): all 24 arms exactly 5,000 updates with common OneCycle peak 1e-3; 448 case errors, 2,424 curve evaluations, same-five-case scalar analysis, and transparent memory recovery/costs.
- [Retained 1/5-case full-cycle comparison](results/fullcycle_u_overfitting/README.md): 16 complete endpoints; four fresh replacements for early-stopped arms and explicit historical costs.
- [Earlier 1,000-update Transolver-family reruns](results/stabilized_u_overfitting/README.md): six warmup/cosine endpoints and a historical mixed-recipe index, superseded by the complete 5,000-update comparison.
- [Transolver correctness and learning-rate diagnostic](results/transolver_fit_diagnostics/README.md): native/checkpoint parity, saved-model reproduction, constant-field collapse and a separate same-seed 1,000-step learning-rate control.
- [U-only matched capacity and exact 1,000-update fitting](results/matched_capacity_u_overfitting/README.md): 16 complete arms, 48 per-case rows, U3 outputs/loss only, zero stress-label or held-out reads; complete endpoint predictions and checkpoints stay local.
- [Matched capacity and exact 1,000-update fitting](results/matched_capacity_overfitting/README.md): 16 complete arms, 48 per-case rows, 3,723.25 s UTC / 3,351.35 s monotonic total training-stage cost; no held-out evaluation.
- [Earlier sample fitting and Hash continuation](results/sample_overfitting/README.md): different capacities, variable update counts, best-checkpoint selection and separate continuation costs.
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
