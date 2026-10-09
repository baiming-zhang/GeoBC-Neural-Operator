# 单 TRAIN/ver 近似物理基线

范围：101_479/ver，全原节点179316；报告状态 `completed`，读出 `finite_budget_endpoint_unconverged`。

输入/算子：全 TET10 关联与真实约束/载荷 receipt 已保存；8次 K 检查，几何/代数可行性 `True`。输入与算子共同耗时 6.60568 s。

求解：`iteration_budget_limit`，迭代 2000，终点真实相对残差 0.00190791（门限1e−8）；累计 solve wall 27.9769 s。预算终点或 breakdown 均不算数值成功。

冻结解后仅一次 TRAIN 位移读出：全N U relative-L2 0.0273%，vector RMS 4.47614e-05 mm，最大节点 vector error 0.000362391 mm；固定 support/load/interior 分项在 metrics.json。

平衡：自由残差 0.00190791；含反力净力 N `[0.8098091293013567, -23.664526653952635, 6.055631588326833]`、净矩 N·mm `[102.08399191976059, -150.6786992760608, -1086.437561990926]`。预测与平衡先保存再读GT；stress读取0。

本次调用成本分列：setup/read 1.50891 s，checkpoint 3.42032 s，终点verification 0.0475095 s，GT/metrics 0.025822 s；GPU peak allocated/reserved 245.252/272.63 MB。checkpoint/K/preconditioner属于solve子计时，不能再次相加；完整累计与本次调用字段保留在 JSON。

历史背景：原 FNO480 对同一个 TRAIN/ver 的 U relative-L2 24.8549%（480次、多CAD训练总耗时 99.7806 s）。只作这一案例背景，不与600秒物理求解或VAL排行混为同预算对照。

这是一例近似物理基线：4Q曲线P2积分和 weighted-LS RBE3均非 OptiStruct exact；小残差不保证标签一致或工程验证。未比较语义未明的标量应力，未训练/拟合scaler，旧final、DEVdia及新sealed关闭。GT不能用于停机或调参。
