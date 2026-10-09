# 直接 masked Galerkin 粗解：低保真物理候选

**proposal_only，2026-10-10；未冻结、未派发、未执行。建议值得一次两例筛选。** 它把 trial space 从完整P2自由度改为 `range(P)`，不是调固定32步预条件；可筛出廉价全局力学场，也可能暴露不可接受的空间误差。它首先是U低保真候选，不能填补尚未知的source signed-S接口。

本次只读两例 `elastic_coarse2_ver/{CAD}_ver/{coarse_manifest,summary}.json`、现BRIEF及既存 `AdditiveCoarse` 的P/PT/CSR公开方法；科学计算、数组/GT/权重/final/sealed/GPU访问均0。既有小JSON证据：

| 固定case | coarse向量DOF / fine自由节点 | Ac CSR nnz / P nnz | accepted P2+coarse冷成本；其中solve |
|---|---:|---:|---:|
| TRAIN101_428/ver | 116,865 / 268,516 | 4,448,403 / 496,172 | 42.108s；29.785s |
| DEV102_249/ver | 91,734 / 209,413 | 3,462,192 / 386,468 | 36.492s；29.098s |

现账本fine输入/算子成本8.755/6.033s、coarse构造1.421/0.631s，另有setup读取/写盘等；缓存复用不消除这些单请求冷成本。直接coarse能否更快未知，不能从DOF比例预测实际迭代数/时间。当前两例 `fixed_mid_with_any_free_corner_count=0`、`strict_nested_P1_at_fixed_rows=true`；该事实不能外推其他CAD。

**唯一方程/API：** `b_f=force[free_node_ids]`，`b_c=(PT@b_f).reshape(-1)`，求 `Ac y=b_c`；原公开 `coarse_apply` 是向量CSR乘法，`prolong(y.reshape(-1,3))` 返回fine自由节点3向量，随后仅复用既有fixed/derived-reference完整原节点恢复。`Ac=PᵀAP` 是原曲线P2/四Q模型的Galerkin限制，绝非另建直边四角刚度。建议FP64零初值PCG＋既有positive scalar diagonal，固定真coarse残差≤1e−8/原10000迭代安全上限，无参数/容差救援；实际预算与早停由root预先冻结。

**一项两例门：** 两例均coarse真收敛，完整原节点对各自既有收敛fine数值场的U向量relativeL2差距≤5%，且完整accounted cold各比上述accepted P2+coarse下降≥50%，才扩scope。相应成本上限约21.054/18.246s。门适合作为激进低保真粗筛，不能当工程公差或最终选型：5%是对近似fine模型的数值差距，非原数据GT误差；50%要求足以补偿新增低保真误差。缺失成本/端点、不收敛或任一例不过门均关闭此固定近似，不换CAD/延长预算/调空间。

成本须计已有fine输入/算子、coarse构造与必要验收的实际账本价，加新rhs/模型读取、coarse solve、P/完整节点恢复、端点验证与保存；不要把fine的旧迭代成本收进candidate，也不要把缓存setup写成0。cached latency另列，继承组成账本不叫fresh联合冷测。GT0、solve预测先持久化后才读**数值参考**；参考IO成本同样计入诊断并独立列明。固定fine参考来自原zero-start收敛端点，不挑更有利field；既有accepted P2+coarse的同解门只是保证基准可信，不转移其GT精度到新近似。

除主要门，报告U绝对RMS/max(mm)、support/load/interior差距、coarse真实残差，以及 `Pᵀ(b−A Py)` 与fine残差；coarse残差小不代表fine方程满足。检查/报告投影正交性不能冒充全fine平衡。已有P满列秩为条件SPD提供基础，不需要本设计重复旧整套guards。

**场与约束限制：** P以真实corner/mid身份定义且fixed行屏蔽；别把它称h粗网格或必然严格nested P1。即使本两例固定行条件成立，曲线几何中 `u_mid=.5(u_a+u_b)` 是参考单元插值关系，不自动重现物理坐标中的所有affine/rigid-rotation位移。完整P2重建可保持共享节点C0，但其physical gradient仍用原曲线J；节点5%通过不保证梯度、热点、张量恢复或源signed-S合格。若需梯度数值差距只能对同fine模型明确报告，不能称strainGT或选用S公式。

两例过门后只说明低保真条件场值得扩展；下一用途可是另一个学习family的多保真残差/物理条件，但须另立公平对照、计入物理冷成本，不自动选定Hash或开始训练。完整GeoBC目标仍是U与原signed-S、跨CAD/load、合法连续查询及完整成本；本候选与点云-only学习模型接口信息不同，不能跨旧集合拼排名。
