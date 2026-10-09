# GINO 与既有六家族：固定480更新初筛

固定480端点，GINO的U/S均值均未低于旧FNO；实测可比训练时间为740.1 vs 99.8秒。GINO480耗时已超过四个600秒确认端点；不能把多耗时后的更新数比较称作成本效率胜出。单seed短筛只提供后续候选依据，不能据此选定最终solver。已观察训练端点仍下降，不将慢路线直接淘汰。

| 方法 | 480端点 U mean / worst (%) | signed S mean / worst (%) | 可比训练秒 |
|---|---:|---:|---:|
| GINO | 54.42 / 76.24 | 72.46 / 84.63 | 740.1 |
| MLP | 71.59 / 97.12 | 97.60 / 99.76 | 43.7 |
| Hash | 92.53 / 191.41 | 92.76 / 96.64 | 41.0 |
| Transolver | 100.58 / 166.04 | 99.27 / 104.11 | 122.4 |
| Transolver++ | 94.52 / 115.70 | 99.70 / 108.17 | 382.9 |
| Transolver3 | 103.17 / 133.85 | 99.15 / 102.19 | 402.6 |
| FNO | 47.51 / 71.86 | 66.71 / 78.73 | 99.8 |

主结果固定480；GINO共同VAL-score best次结果是step480（共同VAL-score best（次结果）），同点U/S均值=54.42%/72.46%，不替换端点或以某一场单独选模。初始0、GINO120/240/360/480与旧族真实端点均完整保留于[metrics.json](metrics.json)。

共同strict40TRAIN/5VAL×known3、PointNet32＋原25、TRAIN-only signed4/asinh scaler、seed2026和四次完整case遍历；零final/dia/新test使用。GINO train_seconds包括case/缓存读取与H2D，首建CSR/save另报，不加进图横轴；总wall=864.2s。GINO原参数numel=7,478,973，complex条目按1计；源形状换算实DOF=14,556,861。旧FNO14,177,284按实数存储计；频带相当但lifting/skip/projection/GNO不同，非严格容量或单组件对照。GINO Adam foreach/非fused，旧族Adam fused，成本差异不能只归因网络。

成本背景为已完成FNO/MLP两seed的600累计秒预算端点（含真实超预算）：FNO/2026：608.7s，U/S 36.28/56.45%；FNO/2027：601.6s，U/S 36.22/56.07%；MLP/2026：604.2s，U/S 51.69/75.62%；MLP/2027：607.7s，U/S 43.01/73.24%。图中空心点单列，不称GINO480等600或按它们外推等时精度；两种预算视角分开。
[实际精度-时间点图](error_cost.png)仅画正累计训练秒与joint VAL score散点，横轴log；不插值、不跨时间范围外推或捏造等时精度。每场最差case、hotspot、推理/显存、逐pass与CSR build/read/save成本见JSON。本轮不证明收敛、连续场/梯度、硬BC/PDE满足或工程可用；角色region是描述符而非节点Dirichlet。
