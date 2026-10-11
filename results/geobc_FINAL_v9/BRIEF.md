# FINAL20：冻结后一次CAD隔离测试

**主模型FNO在读取FINAL标签前由knownDEV联合误差选定，保持不变；全局Hash作热点对照。**

| 方法 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|---:|
| FNO100_12000 | 22.39% | 48.54% | 41.69% |
| GLOBAL_HASH100_12000 | 24.29% | 48.44% | 34.05% |

同100训练CAD×三载荷／12000步／seed2026／oldTRAIN40-only scaler；固定20 FINAL CAD×三载荷共60例，全原节点物理U(mm)／原signed S(MPa)。

120份完整有限预测先提交／重验，再20个官方CSV各数值解析一次，60份场共享两个候选；完整阶段155.38秒。没有训练更新、scalerfit、ID替换或根据FINAL改primary。

该FINAL来自可用官方TRAIN池，CAD与训练和knownDEV隔离；不是官方独立DeepJEB测试或无偏可用性抽样。单seed、容量4.53M vs12.25M、精度FP32 MLP vsFP16 TCNN不同，不能称纯架构因果或显著性结论。

Stress保留CSV原列和符号，不重建Cauchy应力或伪造符号。真实连续查询一致性／梯度已验证，但独立离节点FEM准确性未测。场图固定第一FINAL CAD／ver，所有原节点参与物理评分；图为normalized XY投影，前后节点可能重叠，两个预测使用相同色阶。
