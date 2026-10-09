# 五CAD冷启动物理屏幕（draft，未授权执行）

2026-10-10。依据已完成的数值确认：TRAIN101_479/ver 在4093步达到真实残差9.977e−9，累计solve57.011秒（含原27.977秒），continuation没有读取GT。这支持检查物理方法的跨CAD兼容性与成本，尚不证明OptiStruct等价或工程资格。

**固定范围。** 新物理案例按顺序：TRAIN `101_428/ver,101_56/ver,101_78/ver`，DEV `102_249/ver,102_38/ver`。这五CAD已用于神经训练/开发，本轮不是新封存泛化测试；排除101_479，失败不换CAD。已有gradient manifest元数据合计1,092,529节点、684,470 TET10；本文未读数组、GT、raw deck、权重或模型，未运行GPU。

**同一个已接受近似模型。** 每CAD从实际deck读取材料、零SPC、support RBE2和loaded RBE3；沿原明确specialization与控制排除，不猜role flags约束，不放宽未知模式。真实GRID IDs对齐原H5行；free物理节点identity三自由度，loaded孤立reference仅输出恢复。曲线TET10小应变各向同性弹性：physical梯度由 `J[c,r]` 的转置solve得到，完整四点积分、matrix-free K和精确自由节点3×3 block-Jacobi与原模型相同。RBE3加权刚体拟合、四点近似积分及未公开DEBUG排除的限制不变；不是原求解器精确重放。

**首次实际阶段兼验。** 一次input-only准备读取H5 `vertices`、gradient NPZ `cells10`、canonical NPZ `xyz/grid_node_ids`和deck；只用本CAD manifest中的N/T、xyz/Grid IDs SHA、`bbox_min_mm/L`与源stat。现manifest没有该五CAD的vertices/cells数组SHA，首次实际读取后从已读内存计算记录；禁止借用101_479的hash。每CAD完整geometry、load-transfer、约束与block-Cholesky门，并且恰8次全K守卫（六刚体+固定2026 u/v）各执行一次；不再toy/preflight或在solve重建。

**统一预算与停机。** 零初值PCG，最多10000步或每CAD累计solve300秒，真实残差≤1e−8才记收敛；每25步与终点真实残差/保存，恢复预算不重置。全5按固定顺序累计实际stage900秒，包含source/cache IO、setup/守卫、solve、写盘、endpoint验证及标签指标；case solve也受剩余stage预算约束。旧input/operator组合300秒上限同时受stage剩余量约束。CUDA计时同步边界实际记录；900秒到点停止新迭代/后续CAD，但完成当前有限endpoint的预测/平衡冻结并明列finalization overshoot，不重置计数。后续项记 `stage_budget_exhausted_negative` 且 `attempted=false`，不能删去或伪记模型不兼容。非finite/非正能量/未知约束等明确停止该CAD，不换预条件器、容差、积分或CAD；其他固定CAD仅在剩余预算允许时继续。异常与科学负结果分别记录，完整journal/状态/检查点严格复用，已完成input/operator/8guards不重建。

**预测先于标签。** 有限有效endpoint先持久化全N physical u、free真实残差、支撑反力与全局force/moment balance，且GT打开前stage剩余预算必须大于0，才仅一次读取该CAD/ver的signed `displacement`。否则 `metrics=null/displacement_reads=0`，不补读GT、不把缺指标或负结果说成精度成功。无效输入/算子或非finite解不读GT；有限未收敛端点可报告误差但不能进入“已验证精度”集合。仅报告allN向量relativeL2、RMS mm、max mm及固定support/load/interior区域；零参考用null相对值与绝对值，不造denominator floor。stress、其他loads、旧final、sealed test、训练/optimizer/scaler fits均为0。

**判别结果。** 主表五行含兼容/收敛/预算或异常状态、allN误差、冷启动stage秒与峰值；汇总分母固定5，分别报兼容率、收敛率、GT已读数、预算未尝试数及最差case。记录data/operator/guard/solve/K调用/verify/写盘/GT成本，误差仅对有指标case计算并标注分母。若兼容且收敛、误差与冷启动成本均有机会，root才选择后续known-load扩展或冻结warmstart对照；若已收敛但标签差，则检查近似模型忠实度，不能用GT救容差。它是传统zero-start物理屏幕，不能和神经训练成本混成泛化排名。

**API/schema草案。** 新隔离 `geobc/elastic_crosscad.py`：`request_contract(plan)->(contract,plan)`，`run(plan,output=None)`，CLI `--plan --output`；`parent_plan`只继承recipe JSON，不调用101_479的contract/prepare。配套 `geobc_elastic_crosscad_plan_draft.json` 只保存几何metadata、顺序、预算、路径与建议receipt，`scientific_rules_frozen=false/actual_execution_released=false`；正式source pins、root审阅和最终冻结后才可执行。
