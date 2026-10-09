# 下一轮 GINO 筛选决定（训练前）

用户要求先找到合适求解器，尚未选择Hash。六家族初筛、FNO/MLP/++成本确认和一次冻结最终验证已完成；FNO与MLP保留为现有对照，GINO只增加一个新的求解器候选，不转向Hash小幅优化。

官方GINO低频配置的完整179316输入、512原节点查询和一次无标签反向已通过（4.985s、allocated1.829GB）。这证明所用缓存CSR和积分核后端可运行，没有训练或精度结论。

训练前源码比较发现频带口径不同：旧FNO `modes12` 在x/y各含正12及负12，z保留12；NeuralOperator `n_modes12` 在x/y总12、z为7。唯一训练候选使用官方 `(24,24,22)`，对应x/y总24、z12；hidden16、4层、R64保持。没有读取GINO结果后调整频率，也不增加低频候选或扫参。来源见 [实现边界说明](gino_vs_fno_implementation_scope.md)。

其余采用相同40TRAIN/5VAL CAD×ver/hor/tor、固定TRAIN-only scaler与25特征、共享1024点PointNet/global32、seed2026、完整120-case顺序的480更新/4passes、Adam lr0.001、完整原节点signed四场均方损失。每个case输入编码一次、输出分块累积精确链式梯度、一个Adam step；计时匹配原入口并另列首次CSR准备与总wall。官方复参数需明确使用非fused Adam，记录实现差异。

这是完整候选比较。积分前端、lifting、skip、projection和参数计数仍有差异，不能称严格容量匹配，也不能把精度变化只归因某个GNO核。已有六家族480结果复用，禁止重复GPU训练以制造对照。低频资源receipt不是新频带完整输出的证明：首个实际训练case是第一次新配置full-output前反向检验，若失败记录直接原因并保留已准备缓存，不默默减少节点、邻域或改配置。

训练规则先冻结、再启动唯一GPU队列；只用TRAIN及原VAL。新独立test5的15文件已取得并保持字段封存，旧final5也不作开发集。根据共同验证平均/最差U/S、实际成本与稳定性决定是否值得成本确认；未经训练结果不预称GINO优于任何候选。连续梯度和组合载荷问题保持待解。