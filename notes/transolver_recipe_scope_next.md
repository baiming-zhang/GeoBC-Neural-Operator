# 三版 Transolver：当前配方与后续选择范围

2026-10-10；仅设计分析，不改变正在运行的十臂配方，也不释放未来实验。复用已有 [架构来源记录](../third_party/transformer_sources.json) 与 [适配范围说明](solver_selection_evidence_review.md)，未重复接口审计；未读取数组、标签或权重。

**当前能比较什么。** 冻结的 [4800 更新方案](geobc_coverage_comparison_plan.json) 比较五家族在同原版 DeepJEB、真实条件25、共享几何32、旧 TRAIN40-only scaler、signed UXYZ/S 四场 MSE、Adam 常数学习率及同尺度案例顺序下的实际效果、失败率和成本。40 CAD×3载荷为40遍，200 CAD×3载荷为8遍；因此规模对照同时改变覆盖面与每例曝光，不能单独归因“数据量”。固定0/1200/2400/4800读出，DEV20×3按完整节点、固定60分母报告 mean/worst/hotspot；预算不足保留缺失，不算已收敛的家族失败。当前固定 TRAIN3 读出只是学习迹象，不代表 DEV 或全 TRAIN。

**影响家族上限解释的官方差异。** 以下均为作者仓库固定提交中的具体任务配方，不是推荐直接替换当前 GeoBC 设置。

| 家族 | 当前预设 | 官方示例中的显著差异 |
|---|---|---|
| 原版 | 5层×256、32 slices | 弹性示例8层×128、64 slices；500 epochs、1000训练例、batch1；TRAIN标准化后解码算相对 L2，AdamW+cosine。见 [启动器](https://github.com/thuml/Transolver/blob/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/PDE-Solving-StandardBenchmark/scripts/Transolver_Elas.sh) 与 [训练源](https://github.com/thuml/Transolver/blob/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/PDE-Solving-StandardBenchmark/exp_elas.py)。 |
| ++ | 16层×256、ratio1、dropout0 | Airplane示例4层×256、ratio2、dropout0.1；200 epochs、标准化 CFD 六场 MSE、Adam+OneCycle、全输入点。见 [构造源](https://github.com/thuml/Transolver_plus/blob/d5a23bc734a0ebac56384cf72049a26af9673452/main_airplane.py) 与 [训练源](https://github.com/thuml/Transolver_plus/blob/d5a23bc734a0ebac56384cf72049a26af9673452/train_airplane.py)。 |
| 3 | 官方 volume 16层×256、64 slices、ratio2 | 600 epochs、AdamW+cosine；每个训练 run 随机取一个分块，再每四点取一点，字段预标准化。当前则整 CAD 全节点、全局上下文，官方 epoch 不能直接换算为当前遍历数。见 [配方](https://github.com/thuml/Transolver-3/blob/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37/main_drivaerml_volume.py)、[训练源](https://github.com/thuml/Transolver-3/blob/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37/train_volume.py) 与 [数据源](https://github.com/thuml/Transolver-3/blob/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37/dataset/dataset_drivaerml_volume_numpy_chunk.py)。 |

**不能断言什么。** 当前容量、节点曝光和 GPU 秒不相等；结果不是纯机制因果、官方论文任务重放或最佳优化器下的家族上限。三版官方示例的输出与输入也不同于 GeoBC；Transolver++ 架构名称不意味着使用 DeepJEB++。不能从封存前的 TRAIN3 或短跑结果选赢家，也不能据本轮节点精度宣称连续梯度、组合载荷或工程资格。

**十臂固定读出后的至多两条条件分支。**

1. 若有候选在 signed S、U、最差案例及实际成本上形成实质优势：保留至多两种不同家族，下一阶段预先锁端点/种子，用另一独立 CAD/载荷集合及连续查询、真实服务冷成本作资格验证；本轮 DEV 是开发证据，不能兼作该资格检验。
2. 若没有合格候选：先结合完整 TRAIN 曲线与曝光判断是否仍明显欠拟合。只有持续学习且成本可接受时，才值得一次预先冻结的“官方完整训练制度移植”确认，保留原版 DeepJEB/完整节点/共同任务，不扫宽度、学习率或根据新 DEV 反复救配方；若 TRAIN 已拟合而 DEV 失败，则关闭“只需更长训练”的解释，转向数据覆盖或几何/BC 信息的宏观对照。

本分析没有证实调度造成当前误差；下一方案与门限须在实际十臂读出后由 root 单独冻结。
