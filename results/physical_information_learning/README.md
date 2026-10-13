# Physical information in joint U/S learning

This stage tests added predictor information after [backbone/package selection](../backbone_comparison/README.md). Its matched ZERO control uses the same enlarged decoder as ACTUAL; an older Transolver or smaller plain-FNO result cannot replace that control. See the [paper evidence chain](../../docs/paper_logic.md).

Three matched seeds on 100 training CADs and 20 known development CADs. ACTUAL met the fixed practical-benefit criterion. Mean improvement over ZERO: U 6.384 pp, S 13.421 pp, primary 9.903 pp.

Models share capacity, initialization, case order, 20 passes, loss and the original training40 output scaler. All 360 full-node development predictions were saved before the 60 shared original U/S targets were read. The development CADs are iterated rather than blind. Physics input production costs are reported separately.

| Input | Mean U error | Mean S error | Mean primary |
|---|---:|---:|---:|
| ZERO | 21.746% | 49.419% | 35.582% |
| ACTUAL | 15.361% | 35.998% | 25.680% |

Recovery run cost: UTC 2864.185 s / monotonic 2625.506 s. The campaign took 3143.708 s UTC including the initial 166.660 s cache failure and adaptation wait. Development inputs stayed in CPU memory during recovery and moved to the GPU one case at a time.
