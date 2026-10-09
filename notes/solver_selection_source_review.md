# 六家族首轮：公共数据与模型接口聚焦源审阅

2026-10-09，**PASS，无实质阻塞**。仅阅读新 data/models、必要已有公式接口和冻结配方的小 JSON；未 import/执行工具、读取科学数组/GT/权重、运行 fixture/native/GPU 或修改模型、数据、计划。此结论是静态源审阅，不是实际运行验收或放行授权。

`solver_selection_data.py` 只选择互斥40/5/5 CAD split中的 TRAIN120/VAL15、ver/hor/tor，检查 sample_key/源标签坐标对齐并复用同 CAD 几何；四场按 displacement XYZ 与源 signed stress 排列。FieldScaler 仅用 TRAIN 原节点物理标签拟合，VAL 仅编码/评估。保存的29通道 FNO grid 来自 occupancy+解析网格xyz+原25输入，GT encoded-splat只用于显式、临时的roundtrip诊断，不写为模型输入。8角 stencil 的 XYZ flatten 次序、分区权重及 encoded4 query→物理解码的接口一致。Python独立Random生成四轮固定案例顺序，与模型/Gumbel RNG分离。`solver_selection_models.py` 在同seed下先构造公共PointNet，每次由调用者重新encode；所有前端保留global32梯度，FNO用live expand+cat而不detach。point60、Hash57条件、staticgrid29/FNO61、四个signed输出一致；原Transformer完整上下文和TS3官方全局chunk保留，旧描述被新schema覆盖。未发现target进入输入、验证label参与scaler或条件梯度在这些模块中被截断。

训练端仍需保证完整case/chunk节点权重与live context累计正确、每step调用encoder、++评估RNG隔离；该trainer由根另行聚焦审阅，此note不声称已经核验。输入adapter为节点密度加权的trilinear point-to-grid新FNO候选，BC标记可能平滑；GT往返是同adapter诊断，非架构误差下界。不同family容量/后端与四轮未收敛限制应继续如实报告，无需因此更改已冻结配方或追加数据总审计。

审阅时身份：

- `geobc/solver_selection_data.py` SHA256 `974aff62e56a405e93fd2c65cda9832b8c2a2130595f64a82c484275d83e67a6`
- `geobc/solver_selection_models.py` SHA256 `d53c5672e0dcecb2b87f1fe9401a759503ea05dc72c337a9b2d96cfc1a03382c`
- `notes/geobc_solver_selection_plan.json` SHA256 `0c45275b280dca66e9d9ec66dd31026d30bca5134848f88802372c9e5768a4fe`
