# Shared-trunk field-specific learning

Three fresh seeds use one geometry/FNO trunk and separate U3-input and gradient9-input heads. Each seed sees the same 100 CADs, 300 load cases and 20 training passes. The original training40 output scaler remains fixed.

| Model | Mean U error | Mean S error | Mean primary |
|---|---:|---:|---:|
| Joint input model | 15.361% | 35.998% | 25.680% |
| Two-model field route | 11.717% | 35.471% | 23.594% |
| Shared-trunk field heads | 11.569% | 34.819% | 23.194% |

The compact model has 4,554,762 parameters, versus 9,056,788 for the two-network route. The predeclared within-1pp field gate passed. All 180 predictions preceded shared GT60 reads. These are the same 20 iterated development CADs; field routing was chosen using development results. The two-model reference is inferred from component field metrics.

Actual full-stage cost: UTC 3127.181s / monotonic 2830.391s. Physics input construction is a separate cost.
