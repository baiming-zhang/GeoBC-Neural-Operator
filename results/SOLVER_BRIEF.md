# GeoBC：一分钟结果

**目前主模型是FNO；全局Hash在应力热点上更好。两者跨CAD误差仍高，尚不能替代FEM。**

冻结后20个独立CAD×三载荷、60例全原节点物理评估：

| 方法 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO | 22.39% | 48.54% | 41.69% |
| 全局BC控制Hash | 24.29% | 48.44% | 34.05% |

FNO位移低1.89个百分点，16/20 CAD更好；总体应力基本相同。Hash热点低7.63个百分点，20/20 CAD改善。FNO在读取这些标签前已由knownDEV联合误差选为主模型，此后没有改变。[独立测试](geobc_FINAL_v9/BRIEF.md)。

两者同100训练CAD×三载荷、12000更新＝40遍、seed2026、oldTRAIN40-only scaler。FNO整轮16.61分钟；Hash纯训练合计12.21分钟，两个训练阶段wall合计26.95分钟，含原元数据故障及修复等待，另不含成功阶段间分析等待。独立测试155.38秒。40遍包含在这些总耗时中，不再乘40。

原Hash<1%来自已知单样本拟合。共享Hash50迁移到新几何／新载荷组合约92%；三个高误差案例各自拟合能降到0.055%–0.125%。因此单场拟合能力与新几何／边界求解能力之间有很大差距。整体算子上下文确实改善Hash：同4800步几何Hash U/S41.27%/61.61%，全局Hash29.86%/54.01%；全局latent置零误差显著增大，但这是同权重依赖诊断，不是重训消融。[宏观对照](geobc_operator_hash_v5/BRIEF.md)。

与学长建议的Transolver对比：已有同40CAD／12000步／约4.53M对照中，FNO26.50%/55.55%，Transolver++52.35%/83.99%；目前++是TS系列重点候选。这是当前配方的结果，不能判定Transolver架构上限。[公平对照](geobc_short_screen/BRIEF.md)。100CAD同5遍短筛及knownDEV60对照也已完成：FNO U/S60.56%/74.58%、训练1.45分钟；native++67.43%/94.86%、训练32.08分钟。FNO初筛仍领先；两者还没充分拟合，不是架构上限。[5遍对比](geobc_macro100_short5_v11/BRIEF.md)。新的同5遍FNO物理损失验证已完成：knownDEV位移／应力31.61%／56.10%，热点45.08%，整轮4.27分钟。相对旧损失大幅改善，但损失同时改变了病例幅值权重、U/S平衡与asinh梯度，不能归因于单一因素。[新配方简报](geobc_shared_physical_v12/BRIEF.md)。native++相同物理损失的5遍对照已于北京时间13:03:22启动，预计约35分钟；终态后再比较两种模型。不默认40遍，也不继续Hash微调。

原CSV GRID→H5 vertices对齐，signed U(mm)/S(MPa)原列／符号保留；teacher、grid与FEM误差分开。TRAIN／DEV／FINAL按CAD隔离，scaler仅训练集拟合；这次FINAL来自可用官方TRAIN池，非官方独立测试、单seed且容量／精度不同。它已揭盲，未来选型不能再当盲测。真实CUDA连续查询／梯度通过，但独立离节点FEM精度未测。[连续接口](geobc_trained_operator_hash_queries/BRIEF.md) · [位移云图](geobc_FINAL_v9/plots_v2/field_U.png) · [应力云图](geobc_FINAL_v9/plots_v2/field_S.png)。
