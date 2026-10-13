# Fixed final-set model evaluation

The compact shared-trunk field model and its matched zero-physical-input control were selected using development results. Both use three registered seeds, the same original training40 output scaler, and all 20 fixed final CADs with three loads. No model is retrained or chosen using final errors.

| Model | Mean U error | Mean signed S error | Mean primary |
|---|---:|---:|---:|
| ZERO | 21.255% | 47.003% | 34.129% |
| FIELDWISE | 11.629% | 34.247% | 22.938% |

Primary improves by 11.191 pp, with 3/3 paired seed wins. The predeclared practical-benefit criterion passed. All 360 full-node predictions preceded one shared read of original final U/S fields (20 CSV parses, 60 load cases), aligned by existing GRID row maps.

The seeds share the same 20 CADs. This is an availability-conditional, identity-disjoint evaluation within the official training pool; separation by shape family is not established. Original signed S labels are used directly, without assuming an export sign rule.

Actual checkpoint/cache/inference/save/readout cost: UTC 96.068s / monotonic 87.197s. Physical-input generation and model training are reported separately.

## Load and CAD consistency

Derived only from the saved scalar case errors; no field arrays or labels were reopened. Each load mean uses the same 20 CADs and three registered seeds.

| Load | ZERO U / S | FIELDWISE U / S | Primary reduction |
|---|---:|---:|---:|
| ver | 21.447% / 49.603% | 10.490% / 34.350% | 13.106 pp |
| hor | 20.010% / 50.331% | 8.815% / 36.540% | 12.493 pp |
| tor | 22.306% / 41.074% | 15.584% / 31.850% | 7.973 pp |

Averaging the three loads and seeds within each CAD, FIELDWISE improves U on 20/20 CADs, S on 20/20, and primary on 20/20. These are paired descriptive counts over 20 shared CADs, not 60 independent seed-CAD samples. No final-case or seed selection follows this analysis.
