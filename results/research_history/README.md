# 原始历史报告档案

这层保留指定Git提交中的五份原报告正文，供论文追溯实验、配置范围、失败结果和当时判断。原稿的日期、“当前”“下一步”和标签封存状态都属于对应历史阶段，不作为当前系统状态。

面向论文的骨干证据整理见 [backbone_comparison](../backbone_comparison/README.md)，完整叙事见 [论文证据链](../../docs/paper_logic.md)。最新物理增强、字段消融、Fieldwise和FINAL结果各保留在自己的当前报告中，避免把历史拟合数字与最终泛化结果混排。

| 原始报告 | 恢复版本 | 内容与定位 |
|---|---|---|
| [30项早期实验总表](original_reports/ALL_METHODS_RESULTS.md) | f621a3bc5 | Slides复现、MLP、NGP、FNO、Transolver、残差、蒸馏、50CAD共享拟合；主要是拟合证据 |
| [几何、边界与运动学探索](original_reports/results/geobc_exploration/REPORT.md) | f621a3bc5 | 几何／BC、载荷响应、局部条件、P1/P2、梯度、Query Context和负结果 |
| [FNO与Transolver系列比较](original_reports/results/TRANSOLVER_COMPARISON.md) | f621a3bc5 | 当时固定4800端点、跨CAD误差、容量、成本与配方边界 |
| [统一表示与容量的对照](original_reports/results/geobc_common_grid_comparison/BRIEF.md) | e5a052535 | 同R64输入、native25查询头、约4.53M容量的FNO／Transolver++重要控制 |
| [物理求解与学习系统研究记录](original_reports/results/RESEARCH_HISTORY.md) | f621a3bc5 | 求解器筛选、误差分析、张量尝试、成本与连续场研究过程 |

原目录层级在 `original_reports/` 内保留，未改写报告正文。历史原文中的相对链接仍按原提交解释；本档案不是整个旧代码树的镜像。需要查阅当时的其他配置或报告时，请用 [sources.json](sources.json) 中的原始GitHub链接进入相应提交。这次不恢复notes、模型权重、数据数组或聊天材料，也不重跑旧实验。
