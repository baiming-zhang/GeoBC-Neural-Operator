# 固定 TRAIN 局部 solver 能力诊断

唯一实际诊断 completed，3.718 秒；固定 `101_479/ver` 的 32 cells / 8 faces / 504 logical queries（248 unique 原节点、128 Gauss）。四个已锁 endpoint 全部保留，两 seed 均报告。

| Solver / seed（endpoint step） | 原节点 U / signed stress relative L2 (%) | P2 proxy relative L2：共同 full10 / 原生 (%) |
|---|---:|---:|
| FNO / 2026 (2880) | 7.654 / 30.480 | 609.688 / 640.433 |
| FNO / 2027 (2880) | 11.206 / 36.493 | 506.976 / 534.839 |
| MLP / 2026 (7080) | 17.432 / 54.710 | 412.646 / 未实现 |
| MLP / 2027 (7080) | 22.272 / 55.455 | 420.844 / 未实现 |

此固定局部，两 seed 的 FNO 节点 U/S 误差均低于 MLP；节点拟合优势没有转化为准确的 P2 运动学 proxy，FNO 原生查询也未消除这一差距。 例如 FNO2026 的 GT-P2 Frobenius RMS=0.00085849，误差 RMS=0.0052341（无量纲）；差距不只表现为相对百分比。

几何与数值 guards PASS；详见 [metrics.json](metrics.json) 的绝对 RMS、signed cross、最差 cell、源身份、查询可用性、成本与显存。共同 full10 的 C0 是结构接口检查，不能当学习成功；FNO 网格面导数允许跳变。MLP 原生 offnode 因原25输入延拓未定义而未实现。

这是固定 TRAIN 局部诊断，不能外推全 CAD 或 final，也没有 solver strain/offnode GT；P1/P2 是 GT 节点派生 proxy。原25角色不自动定义节点 Dirichlet，此结果不归因于 BC 模糊、不修复输入、不再选 seed 或 checkpoint。
