# 固定4800之后：只保留两个宏观问题

2026-10-10；仅设计，未释放训练或新标签读取。对应 [draft](geobc_after_fixed4800_solver_plan_draft.json)。原十臂20个源码pin、4800端点、正式DEV20×3与未齐全的317/540下载记录保持原协议。

**先判断问题。** FNO 全TRAIN120的 U/S 均值为13.98%/34.86%，五独立CAD-ver为20.62%/55.06%；++分别为37.64%/77.59%与25.91%/74.46%。TRAIN120含三载荷，不能直接把这些差当同分布泛化差。只取既存TRAIN-ver40标量，FNO U/S为7.86%/35.05%，++为29.77%/74.61%；++未显示应力进一步恶化，当前更应先验证拟合制度。五CAD的U已被历史物理路线读取，是known-U；其S是本次field-blind读出，二者不能都称全盲。

FNO有14.18M参数，++有4.49M；虽然原25条件来源相同，FNO经过R64节点splat均值，++直接读取原节点条件。当前跨家族结果包含容量、表示和优化差异，不能宣称Fourier机制因果或家族上限。

**1．++作者训练制度确认，优先执行。** 保留当前16×256、ratio1、dropout0构造、同TRAIN40×3、原signed4/asinh scaler、完整节点MSE、相同从零初始化/种子2026与完整case-order。只新开一臂Adam+OneCycle+gradient-clip1，200完整遍历/24000更新；固定4800与24000全TRAIN120读出，既有恒定Adam4800不可变结果作为参照。新4800是同曝光制度bundle比较，24000是制度加曝光的完整确认；不拆成学习率扫描，不把单项clip/schedule认作原因，也不再花一臂恒定24000成本。

200遍历/24000是本项目的预冻结曝光量。作者Airplane例为200 epochs，Adam、OneCycle max_lr1e-3/final_div_factor1000和clip_norm1；这是六场CFD到本项目signed四场的制度移植，**不是作者DeepJEB配置或精确重放**。官方4层/ratio2/dropout0.1保持为未测试结构差异；本次不同时改架构。OneCycle遗漏的默认项在draft明确列作本地假设，正式源准备须与固定Torch源码绑定。

当前++4800训练4112.77秒，线性估计24000约5.71小时，非性能承诺；实际训练上限8小时，另给保存/初始化/固定读出20分钟，总墙上限30000秒。不足24000是曝光未完成，不能据此淘汰++。所有恢复/失败时间累计，调度器、Adam、RNG和pass cursor一并恢复。此确认可仅使用已完成的TRAIN120开展，不依赖540下载齐全或正式DEV开放。

**2．FNO局部输入表示配对，不再扩家族。** 两臂共享PointNet32、R64/width16/modes12/four-block FNO产生16维latent、同查询MLP76→128→128→4及共同初始化；均使用原canonical xyz3、global32、同latent16。唯一差别是local25：控制臂取原输入grid的query-roundtrip条件，候选臂直接取原节点25条件；参数、点编码、输入scale、完整4800更新和训练制度完全相同。原节点位置只决定同一查询与坐标编码，不加Hash、物理U/σ、标签衍生输入或新归一化。

这检验是否保留局部几何/BC/material细节有助于应力；它不是现有grid4-FNO的容量匹配重放，也不证明连续梯度资格。复用`FNO3d(out_channels=16)`、`query_grid`和既有global-latent/local-head链式反传模式，另写隔离adapter；旧physical pair的输入、loader/run/clock不能移植。两臂各4800完整更新/40遍，固定0/4800，训练上限各1小时，含保存与统一读出总2.5小时；预算不足保留负结果。

**共同验收。** 全TRAIN120的U/S/hot/最差案例与实际成本必须完整报告；不以TRAIN3曲线替代。++对既有4800参照、FNO对同参数控制臂，预定推进门为最终signed-S均值至少下降20%，U均值不恶化超过5%、S/hot最差不恶化；这是结构推进门，不是GeoBC合格门。正式DEV60仍封存：只在完整输入可用、计划评估的全部最终全60预测落盘后共享读一次标签，并同时报告父协议的绝对U/S/hot资格门。已有五CAD不再用于early-stop、参数选择或救配方。若只有TRAIN收益而DEV无收益，关闭“制度/局部接口足以解决泛化”的解释；不追加第三条候选。

**来源与口径。** 数字仅由 [TRAIN120](../results/geobc_train_fit/summary.json)、[S5](../runs/geobc_exploration/coverage_S_field_blind5_fixed4800/summary.json)、[known-U5](../runs/geobc_exploration/coverage_known_U5_fixed4800/summary.json) 已完成JSON标量计算；没有重新读数组、GT或权重。[末段制度证据](geobc_fixed4800_training_regime_evidence.md)还显示++在线loss末五遍比前五遍下降11.60%，并未证明收敛。官方固定commit `d5a23bc734a0ebac56384cf72049a26af9673452` 的 `train_airplane.py:38–42,92–98` / `main_airplane.py:25–27,74–80` 已读；[既有范围说明](transolver_recipe_scope_next.md)给出对应任务限制。两实验均为draft/actual=false；暂不创建新训练module，等当前完成快照发布后再单独准备源码与真正发布。
