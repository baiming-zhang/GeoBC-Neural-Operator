# Fixed final-set model evaluation

The compact shared-trunk field model and its matched zero-physical-input control were selected using development results. Both use three registered seeds, the same original training40 output scaler, and all 20 fixed final CADs with three loads. No model is retrained or chosen using final errors.

| Model | Mean U error | Mean signed S error | Mean primary |
|---|---:|---:|---:|
| ZERO | 21.255% | 47.003% | 34.129% |
| FIELDWISE | 11.629% | 34.247% | 22.938% |

Primary improves by 11.191 pp, with 3/3 paired seed wins. The predeclared practical-benefit criterion passed. All 360 full-node predictions preceded one shared read of original final U/S fields (20 CSV parses, 60 load cases), aligned by existing GRID row maps.

The seeds share the same 20 CADs. This is an availability-conditional, identity-disjoint evaluation within the official training pool; separation by shape family is not established. Original signed S labels are used directly, without assuming an export sign rule.

Actual checkpoint/cache/inference/save/readout cost: UTC 96.068s / monotonic 87.197s. Physical-input generation and model training are reported separately.
