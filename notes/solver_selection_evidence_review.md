# GeoBC 求解器选型：现有证据与有界广筛建议

状态：研究建议，未选择/启动新实验。2026-10-09。仅阅读报告、配置和小型结果 JSON；未读取数组、标签、权重或运行 GPU。用户已明确要求停止 Hash 局部微调，先探索最适合 GeoBC 的求解器。

## 现有证据能支持什么

原 30 项复现检验训练样本拟合；后续 48 项包含训练、冻结评估/诊断及输入可行性，并非 48 种求解器。原始 DeepJEB 已对齐的结果有效；撤回的旧 H5 字段/坐标错配结果不纳入。

| 证据 | 可以比较的部分 | 不能据此得出的结论 |
|---|---|---|
| Transolver_two 与 ++_two_linear | 同两案例、uXYZ+signed stress、linear scaler、1000 full-case updates、physical-MSE 选模；实际参数约 1.82M/4.46M | 容量不相同，且无未见 CAD；只能比较该拟合配方 |
| Transolver_five 与 TS3_five | 同五案例、linear scaler、2500 full-case updates；参数约 1.82M/7.61M | 不是等容量、等时间排名；++_five 使用 asinh，不能混作相同数值处理 |
| TS3 two short/long、MLP、Hash、FNO | 各自原稿预算或明确替代配置的拟合与成本证据 | 800/1000/10000 updates、full-case/sample、四场/stress-only、选模目标和 adapter 不同，不能直接总排名 |
| NGP+Transformer residual、FNO→Hash 蒸馏 | 复合 pipeline 的拟合结果 | 冻结 teacher 带来额外监督/成本，不是纯 Transformer 或纯 Hash 对照 |
| 40 TRAIN/5 validation CAD 的探索 | 同一配对内部的初始化、scaler、采样、布局及预算有冻结匹配记录 | 主要覆盖 Hash 条件与损失变体；尚无该 split 上六家族公平选型 |
| topology self/neighbors TRAIN4×3 的 1000-step pilot | 一个具体粗图消息模块的固定训练端点对照，四门均未通过 | 无 validation；不能淘汰 MeshGraphNets/GNN 家族 |

Hash 有明确的节点拟合和速度优势，但这不能预定它是 GeoBC 求解器：共享 50-CAD stress 模型遇新 CAD/同 CAD 新载荷组合约 92% 误差；最新 full12→coarse8 的开发验证 U 均值 22.233%→19.967%、最差 41.293%→34.394%，P1 代理 626.229%→165.966%，四门中三门通过、P1≤100%失败。单 CAD 专训 U/P1 很好，但同 TRAIN101_479/ver 的完整十节点四点积分代理为 P1 22.391%/P2-u 187.313%。后者说明节点拟合与旧角点恢复指标不足以验收完整二次场梯度；它不是 solver 应变 GT，也不是泛化测试。

证据来自 [原30项报告](../results/REPORT.md)、[探索报告](../results/geobc_exploration/REPORT.md)、[复现范围](../references/reproduction_scope.md)；局部公平性核对了上述运行的 config.json。最终测试五 CAD 尚隔离；5-CAD validation 已反复开发使用，dia 已作一次冻结验证，不能重新称为全新盲测。

## 首轮只比较六家族，不继续叠加 Hash 改动

| 候选 | 本地基础与主要适配点 |
|---|---|
| MLP | 廉价点查询基线；必须注明是否使用共同完整几何编码，否则 xyz+局部摘要无法与全场 operator 等信息比较 |
| Hash | 冻结一个明确现有配方作候选；不再试层数、gate、半径、loss 微调。新的四场/选型协议需 fresh train，旧 19.97% 不直接作新轮成绩 |
| Transolver | 已有 pinned 官方不规则点适配，直接接原节点/25输入；保持完整 CAD 上下文 |
| Transolver++ | 已有官方适配；上游 eval 仍有 Gumbel 随机性，固定公共 eval RNG 并报告，不用最好一次随机输出 |
| Transolver-3 | 已有官方 volume preset；chunk 必须每层保留全局状态，不能独立空间块冒充原架构 |
| FNO | 已有 64³/192³ 适配；共享几何 occupancy/BC/material 输入，记录栅格化和查询回原节点损失、内存与准备成本。64³ 失败只能判该预算/分辨率，不能淘汰所有 FNO |

这些家族的官方来源已核对：[Transolver](https://github.com/thuml/Transolver)、[Transolver++](https://github.com/thuml/Transolver_plus)、[Transolver-3](https://github.com/thuml/Transolver-3)、[NeuralOperator](https://github.com/neuraloperator/neuraloperator)。任务使用本地已 pinned 版本，不因网页更新自动换实现。

如必须加一个不同的几何 operator，优先仅加 GINO，理由是其官方接口本就连接任意坐标网格与谱 latent，而不是给 Hash 再加手工通道；它需要新的 GNO 邻域/latent-grid adapter 和成本记录，尚未本地验收，不应拖住六个已有候选。[官方 GINO 实现](https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/gino.py) 不保证 DeepJEB 上有效。已有原网格也支持图家族，但其完整 solver adapter 尚缺；当前负 topology pilot 不足以替代正式 [MeshGraphNets](https://arxiv.org/abs/2010.03409) 比较。本轮不同时新增 GINO、GNN 等多个实现。

## 建议的有限成本协议

1. 沿用已冻结 CAD 分组：40 TRAIN×ver/hor/tor、5 validation×同三载荷；每 CAD 所有载荷同组，final5 与 dia 不用于本轮选型。先锁协议再训练，不重新挑容易案例。首轮只有 seed2026、每家一个预声明配置；优化器可用明确的官方家族配方，不能把历史所有方法 LR=.003 当成充分优化，也不做超参数网格搜索。
2. 共同预测物理 uXYZ 与源 signed stress，使用训练集拟合的共同线性输出缩放、共同案例相对损失/四场权重、共同选模目标。第一轮不把 Hash 专用 P1 λ、载荷基或 teacher 单独送给某一家。各模型访问同样的 target-independent 几何/BC/material 信息；允许架构原生点集/网格表示，但明确并计入 encoder/adapter。绝不输入 CAD ID 或 GT 条件。应力单独直接监督，不从 u 构造未知 scalar stress。
3. 建议每家最多 10 GPU 分钟，六家首轮约 60 GPU 分钟的**资源上限而非完成 ETA**；记录固定时间点误差与实际 steps、CAD 曝光、监督节点数。完整几何上下文与 loss 查询采样要分开：Transformer 的上下文不能为了匹配 Hash batch 而随意删点。若监督可用同一固定抽样 witness，就在完整上下文上计算相同节点 loss；FNO 回原节点监督需单独确认可微 adapter。相同 updates/epochs 不等于相同计算预算。
4. 只把精度/成本 Pareto 前两名推进一次 seed2027 的同预算复验；曲线仍明显未稳定的模型写“该预算下未收敛”，不写“该架构不适用”。不因首轮 Hash 领先就开启新的 Hash feature/loss 扫描。最终测试仅在候选及协议锁定后一次使用。

## 选型指标与结论边界

主表使用每个 validation CAD/载荷全原始节点的物理 U 向量 L2 和 signed-stress L2：CAD 宏平均、最差、按载荷分项及 pooled 分开；同时报告 mm/MPa RMSE、stress hotspot、输入 SPC 位上的位移残差。预声明一个共同 checkpoint 标量，避免各家/各指标各挑最好 checkpoint。最终选择依据精度–训练/推断时间–峰值显存 Pareto，而不是参数数目或单节点拟合最低值。

完整十节点 P2-u 体积积分梯度代理宜作为共同场质量诊断，使用同一固定原单元/积分规则，保留物理 Jacobian 与签名交叉项；第一轮无需为了该诊断先改所有模型的训练损失。旧 recovered-P1 与这个指标分栏，P2-u 不能叫 solver stress/strain GT。连续点适配、离节点场准确性、重网格/分辨率泛化也分别验收，不能由 node L2 或已通过的几何 locator gate 推断。

当前可下的选型结论是“尚未选出 solver”。六家族已有 overfit 实现能缩短广筛准备；严格 CAD 验证和统一成本记录才是下一项有效证据。

## 根已选首轮配方的聚焦评估

根随后选定共同40/5 CAD、三载荷、PointNet1024/global32+原25、直接四场、TRAIN-only共同FieldScaler/asinh stress、480完整案例更新（4 shuffled passes）、seed2026/Adam.001；六家各用一个固定配置。上面的10分钟方案仅为先前资源建议，**当前首轮按根选定的四轮曝光执行设计**，本 note 不启动实验。

该配方可以粗筛，须避免三个足以改变比较含义的问题：

1. FNO 的静态几何网格可缓存，但 global32 必须每步可微重算/广播，不能缓存或 detach 公共 PointNet 输出。61通道 node-splat/trilinear 前端是新的几何条件 FNO 候选，不能写成旧 volume-FNO 重放；节点密度加权与 BC 标记平滑是其适配限制。
2. asinh 解码与插值不交换。FNO 统一先 query 编码 grid 输出，再 decode 为节点物理值；roundtrip 应复刻同 encode→splat→query→decode。该 oracle 往返仅描述 adapter，不是架构误差数学下界。
3. 共同 cloud indices 和案例 shuffle 用独立预生成 schedule/hash，不让 ++ 的 Gumbel 或模型初始化 RNG 改变案例顺序。共同 PointNet 结构/初始化也应显式记录，而非宣称所有不同架构参数相同。

480完整case更新是同案例曝光，不是同GPU时间或充分收敛；记录损失/物理误差曲线、实际时间和峰值显存。统一Adam.001是共同首轮配方，不能当各家最佳优化证据。确认轮宜推进2个精度–成本Pareto候选，并至多保留1个仍明确下降的慢结构候选；扩预算而不扫描多个feature/loss/半径。首轮既有Hash成绩也需重新以该共同四场配方训练，不能直接搬旧u/P1结果作基线。
