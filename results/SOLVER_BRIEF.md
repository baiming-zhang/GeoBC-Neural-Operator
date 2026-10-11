# GeoBC：一分钟结果

**当前充分训练的主候选仍是FNO100／12000：新CAD位移／应力误差23.17%／50.85%。全局Hash已成为值得继续比较的候选，最终选型未定。**

同100训练CAD、4800步／16遍、20开发CAD×三载荷全原节点：

| 方法 | 位移误差 | 应力误差 |
|---|---:|---:|
| FNO | 28.59% | 56.49% |
| Geometry Hash | 41.27% | 61.61% |
| Whole-BC operator Hash | 29.86% | 54.01% |

全局Hash相对FNO：位移差1.27个百分点，应力好2.48、热点好6.74个百分点；20/20 CAD应力／热点更好，位移仅5/20更好。总容量12.25M vs4.53M且混合精度不同，不是纯容量匹配对照。

全局latent置零，误差升至83.73%／100.00%；几何descriptor置零35.43%／64.29%。这是未重训的分布外依赖诊断，不能把零输入版本当求解器或因果消融。[结果与成本](geobc_operator_hash_v5/BRIEF.md)。

训练后真实CUDA连续查询已通过，目标读取和训练更新均0，整轮4.83秒。[接口范围](geobc_trained_operator_hash_queries/BRIEF.md)。仅给有竞争力的全局Hash追加7200步到12000／40遍，保留model／Adam／四类RNG／曝光cursor，不改配方；追加训练上限10分钟，整体20分钟。

历史同40CAD／12000／约4.53M公平对照：FNO26.50%／55.55%，Transolver++52.35%／83.99%。TS系列以++为主；当前配方FNO领先，不宣布架构上限。[TS比较](geobc_short_screen/BRIEF.md)。

Hash<1%属于已知单／双样本应力拟合，不能直接比较新CAD联合位移／应力。全部保留原版GRID对齐CSV、signed U(mm)/S(MPa)、原节点；oldTRAIN40-only scaler、knownDEV非FINAL，FINAL数值标签未读。单seed、可用性条件池，尚未证明独立离节点FEM精度。
