# GeoBC solver 筛选

| 方法 | status / steps | best checkpoint | U mean / worst | signed stress mean / worst | 训练秒 |
|---|---|---|---|---|---|
| Transolver | completed / 480/480 | 240（共同VAL分数best） | 99.50% / 110.08% | 99.41% / 100.40% | 122.38 |
| MLP | completed / 480/480 | 480（共同VAL分数best） | 71.59% / 97.12% | 97.60% / 99.76% | 43.70 |
| FNO | completed / 480/480 | 480（共同VAL分数best） | 47.51% / 71.86% | 66.71% / 78.73% | 99.78 |
| Transolver++ | failed / 0/480 | — | — | — | 0.00 |
| Transolver3 | pending / 0/480 | — | — | — | — |
| Hash | pending / 0/480 | — | — | — | — |

满足本轮双均值门槛的已训练候选：FNO、MLP、Transolver（不据四轮结果宣称最终solver胜出）。 Transolver两均值均接近零场基线，尚无明显有效性证据（≥95%仅为文字诊断区间）。

固定 strict TRAIN40 / VAL5 CAD × ver/hor/tor 三个已知载荷；各家共享 PointNet 1024点→global32 与原25维 BC/material，四次完整TRAIN遍历、共同仅TRAIN拟合的含stress asinh四通道scaler。表中U与signed stress来自同一共同物理VAL分数best checkpoint；step 0是未训练基线，不作为已训练候选。完整480步与全部观测端点保留于 metrics.json。原生preset容量不同；FNO使用point→grid特征适配及grid→node查询，有分辨率与插值限制。本轮预算尚不能证明收敛，未达门槛或下降较慢的家族不作淘汰。

[实测端点图](error_cost.png)只绘制正累计训练秒的joint VAL score散点，不作插值；全节点推理耗时、参数及峰值显存见 [metrics.json](metrics.json)。
