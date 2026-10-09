# Solver 最终验证设计（proposal_only；未读 final 标签/权重，未执行）

**锁定后一次验证。** 六项 TOTAL600×两seed 确认全部完成后，仅用 TRAIN/development-VAL，按两seed的预算停止端点共同U/S改善、最差case和实测成本选≤2条路线；共同score之外同时列两field，近100%的微小改善不称成功，无合格路线不强凑名单。锁定现有600秒累计预算/完整pass超额、native配置、TRAIN-only scaler、cloud_seed2026、seed2026/2027、指标与输入身份。每seed固定两种读出：预算停止端点为主，既有共同VAL-best为次（继承screen best也记录origin；step0明确未训练）；同时冻结路径/文件SHA，重合checkpoint只算一次。两seed分别报告及均值/范围，不以final选seed、checkpoint、路线或延长预算；两seed不支持显著性结论。

**final5 known3与dia分栏。** 冻结后对既有final5 CAD×ver/hor/tor的15个完整原节点case一次推理：每seed/每checkpoint给U向量与signed stress相对L2的case均值、最差及case ID，stress hotspot沿现有真值绝对值top1%定义，另报位移向量RMS/最大误差mm、stress RMSE/MAE MPa。两field来自同一checkpoint，不合并不同seed优势；沿当前零参考范数报错语义，不加分母floor。保留每case结果，单列训练累计秒/曝光、输入准备、物理metrics、全节点encode/forward/CPU输出推理秒、峰值显存及参数；最多2路线×2seed×2读出×15=120次case推理。final5/dia仅为冻结路线的组合load迁移（最多40次），不混入known3均值，不据此选模；已观测的VAL-dia不是盲测。负结果照报并结束本轮，final不转开发集。

**最小API复用与缺口。** `solver_selection_data._selected/cache_contract/prepare_data`目前只选TRAIN/VAL，不能绕过它重拟合scaler或把final混入cache；`load_case`立即打开target/physical，不适合输入先行检查。下一入口应只新增final只读eval-cache：按冻结split/原CSV GRID→H5 vertices身份，先读xyz/25features及同款cloud/mask，复用TRAIN scaler；释放后才按显式成员读signed uXYZ/stress。模型复用 `build_selection_model → encode_geometry → forward_points`（MLP可chunk，++须整case并固定case eval RNG）；FNO复用 `forward_grid`与 `solver_selection_data.query_grid`，编码域trilinear查询后再decode，不作网格标签监督。无需新trainer或训练目标。

**连续field/BC能力是另一项验收。** 在固定TRAIN101_479的既定≤32 cells/8 shared faces/≤512物理queries上，沿已验收full10形函数/物理逆定位检查全10节点重放、共享面迹、失败状态及物理残差。所有候选均可把已decode的节点预测做同款full10有符号重建，公平比较这个“节点场重建”，但它不证明原生off-node solver能力，也不是精确曲单元GT。MLP原生查询还缺25features的语义明确延拓；FNO需新query stencil，保持固定输入grid，原生grid查询与上述重建分栏；++目前 `native_continuous_query_implemented=False`，不能用独立query chunk替代整case上下文，须另有保留全局状态的查询接口才称原生连续。失败写能力缺口，不用final修adapter再重测。

**BC与P2 proxy的边界。** 原25的0:6是support-reference DOF角色、6是loaded-interface角色，不自动等价每个成员节点的精确Dirichlet条件；FNO `_splat`会平均这些角色，fractional/串扰不是物理BC。先在固定TRAIN输入对support/load/interior分组量化splat→query角色失真，再按已有明确约束语义报告节点BC残差（mm）；不阈值化修复网格、不改输入救模型。full10延拓角色只能称描述符插值，不能称真实BC flags。需要运动学检验时只复用冻结的物理FP64 `G=(uᵀD)J⁻¹`/对称梯度定义，在同32cells×4正Gauss点做P1/P2及signed midside cross分解；全部J通过才算，不能跳坏点。P2是节点位移派生的kinematic proxy，非solver strain GT；不从单个signed stress通道反推应变/牵引，也不把网格连续性或proxy一致性当精度证明。

**可执行下一步：** 确认6/6后先写一份小冻结eval manifest（≤2路线、两seed、endpoint/VAL-best身份、上述case/metrics/次数界限），随后实现一个隔离的只读final evaluator与输入型eval-cache helper，根唯一队列一次执行known3+分栏dia；能力诊断沿固定TRAIN范围另发有界release。报告一次后停止本轮，不自动扩大预算或替换holdout。
