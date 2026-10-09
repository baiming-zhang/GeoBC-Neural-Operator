# 求解器确认轮协议

状态：六家首轮完成后于2026-10-09冻结并启动。FNO与MLP为主确认候选，Transolver++保留一个低优先级慢结构名额；执行配置见 `notes/geobc_solver_confirmation_plan.json`。本文件设计阶段只读了小型元数据；正式确认沿唯一GPU队列续训/新训。

**目标：**确认固定配方下的精度–成本与种子稳定性，不继续 Hash 局部优化，也不把四轮曝光等同收敛。

- **候选：**FNO两场共同改善，MLP是低成本位移对照；++共同分数继续下降，但应力仍接近零场，保留它仅为慢结构机会，不称成功。Hash最差位移191.41%且后半程位移反升；原Transolver/3最后共同分数回退，暂缓。此筛选不等于永久淘汰；确认结束依据证据决定下一步，不开展家族参数扫搜。
- **配方固定：**沿用40 TRAIN/5 development-VAL CAD×ver/hor/tor、原节点完整case loss、公共25/global32、同一 TRAIN-only FieldScaler/asinh、模型配置、Adam .001与后端。scaler不重拟合，cloud_seed保持2026；不新增梯度损失、teacher、特征或半径。
- **两种子与成本：**每候选 seed2026/2027 各累计约600训练秒。2026从screen的 **last480** 接续，保留Adam、所有模型RNG、独立案例shuffle状态与既有历史；screen的train_seconds计入600，不从best重启。2027 fresh参数/Adam/模型RNG，独立seed2027案例顺序但相同cloud/scaler。新目录记录parent与累计成本，原screen不覆写。现有train_seconds口径包含case读取/H2D、forward/backward、step及CUDA同步；eval、保存、一次性cache准备分别报告，不能叫纯CUDA kernel秒。
- **停止与曝光：**完整120case pass才停：达到600秒的那个pass走完；若screen已超过600则2026无额外更新。实际成本及最后pass超额如实记，预算不是精确600秒。每seed案例均匀曝光，报告pass/step/节点数；更快模型获得更多pass是成本比较的一部分，不伪称等steps比较。每完整pass保存last与成本/物理指标曲线，不按中途VAL波动早停。
- **选模和读出：**仍只用上述共同VAL score挑一个checkpoint，U/stress及最差指标全部来自它；同时把固定停止端点作为首要成本读出，不能只展示valbest。记录两seed分别的误差、差值/范围、实际训练成本、全节点推断时间、峰值显存及完整wall time；两seed不足以宣称统计显著。++继续固定案例eval RNG且隔离训练RNG。若600秒末仍明显下降，结论写“此预算未收敛”，不是架构淘汰；下一预算仅在全候选结果后另行预声明，不现场调参救某一家。
- **最终检验：**确认轮结束并锁定路线、训练预算、seed/checkpoint规则与指标后，才对final5的三种known loads做一次跨CAD检验；预定两seed均报，不用test选seed或继续修改。dia不用于本轮选型，可在同一冻结最终阶段分栏报告组合载荷迁移；旧VAL-dia已观测，不能再称盲测。final5/dia结果若负也如实保留，不把最终集转成下一轮开发集。

依据：现有screen已单列train/evaluation/inference_seconds，并保存last480的Adam、RNG、shuffle状态与costs；因此累计成本续训可重放。此协议需要新确认轮入口/contract允许延长预算与seed2027，不能直接修改或绕过冻结screen480源码的固定步骤/身份检查。
