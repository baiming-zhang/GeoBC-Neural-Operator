# 单 TRAIN101_479：真实弹性算子输入-only pilot

2026-10-10，**proposal_only，仅在根代理判断 PointConv 没有确认机会后考虑**。当前 PointConv 结果未定；本页不是新任务释放、模型实现或参数救援。目的只有判断真实弱形式输入是否闭合，不求位移解、不训练、不读 Field、应力/位移 GT、预测、权重、旧 final 或新 sealed，也不扩到45CAD。

**唯一输入契约。** 未来单job只取已下载 `data/raw/SolverDeck/101_479.fem` 的静态 ver 卡片、原 H5 vertices/cells 与已有 canonical GRID↔H5 行映射/物理 mm 归一化；或复用梯度缓存的 `cells10` 几何成员，绝不加载其 strain/reference 成员。逐卡保留 SPC/SPCADD 的实际 DOF/值、RBE2/RBE3 全连接/参考位置/DOF/weight、FORCE/MOMENT 的作用点与 LOAD 展开、CTETRA→PSOLID→MAT1(E/ν/显式G或其推导)及积分/solver控制。必须解析 GRID CP/CD/**PS永久约束**、GRDSET 默认和相关坐标系；“global”不是自动等于 basic xyz。[GRID 官方定义](https://help.altair.com/hwsolvers/os/topics/solvers/os/grid_bulk_r.htm) 影响ver且未覆盖的卡片、重复 dependent DOF、UM/ALPHA、无法解释的旋转 DOF或单元/材料映射均停止，不默默忽略。已有解析器不是完整力学 reader；roleflag 不替代约束。

**可恢复的最小矩阵子问题。** 已有纯数值 `geobc/continuous_query_geometry.py` 提供完整10节点二次形函数、参考导数和 J，原 fullCAD 代码提供分块几何判据；不调用其标签诊断入口。以物理 mm 坐标作 FP64 matrix-free `Kq.apply(u)`：每个 cell/积分点 gather10节点、计算 `∇N=J⁻ᵀ∇ξN`、`ε=sym∇u`、`σ=2με+λtr(ε)I`，再按 `w·|detJ|·σ∇N_i` scatter 回全原节点；λ/μ由 E/ν取得。[线弹性弱形式](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_elasticity.html) 不存全局K或全部30×30单元K，不用 V/10 或角点P1体积。只提议复用已有四点 `(a,b,b,b)` 置换、`a=(5+3√5)/20,b=(5−√5)/20,w=1/24`：这是曲线P2的**近似积分**，不宣称精确积分、收敛或与原solver选项相同；官方也区分 regular/contact-friendly 默认与模型控制。[积分说明](https://help.altair.com/hwsolvers/os/topics/solvers/os/elements_user_guide_os.htm) 一次全CAD分块处理，GPU FP64可并行，资源与秒数尚未测；不另做小GPU toy/规则扫描。

**RBE3 是当前硬缺口。** OptiStruct 官方确认加权最小二乘运动插值、REFC 默认 dependent 及 UM 重分区，但所查公开页没有完整几何矩阵/系数与数值退化策略。[官方 RBE3/约束原则](https://help.altair.com/hwsolvers/os/topics/solvers/os/rbe3_bulk_r.htm) [官方 least-square 定义](https://help.altair.com/hwsolvers/os/topics/solvers/os/elements_user_guide_os.htm) Sandia 的 `H=(SᵀWS)⁻¹SᵀW` 有 primary 方程，却是其 Sierra/MSC 系实现依据，**不能据此认证 OptiStruct exact**。[Sandia theory](https://www.osti.gov/servlets/purl/1031884/) 需版本匹配的明确 formulation 或 solver-generated constraint coefficients 才锁定H；禁止直接采纳任意L2刚体fit、伪逆阈值、均分参考力矩或夹紧角色节点。

**有限 stop gate（执行前一次冻结数值规则，不看标签改门限）。**

- 输入/J：行映射、TET10边中点顺序、全积分点有限/非退化且局部orientation一致；只继承已有归一化几何阈值，不修复几何。仅8次全K apply：6个归一化刚体列及固定 seed2026 的 u/v；验 `uᵀKv=vᵀKu`、两项非负能量和6个刚体 null residual。提议相对门限1e−9，分母以两向量能量/作用范数定义并记录；通过只证明 sampled algebra，不能证明无其他零模态或积分正确。
- SPC/RBE2：按真实卡片建立刚体关系与参考6DOF约束，验证 dependent冲突、消元一致性及约束对刚体列的秩；成员旋转自由度的处理未证实就停止，不能以“支撑组全零”替代卡片方程。
- 只有H依据闭合才验 RBE3：原参考点的刚体运动矩阵G满足 `HG=I6`；由虚功**推导** `f_members=Hᵀg_ref`，检验 `uᵀf=(Hu)ᵀg`、合力与绕该参考点合矩，N与N·mm分开量纲。等权不是等分力矩；只测6个算子基列和 ver 元数据载荷，不新增场工况。

**明确停止与恢复。** 当前 H 未闭合，最多产出 `unforced_Kq/SPC-RBE2` 的单CAD输入与代数 receipt，状态 `completed_partial / blocked_RBE3`，真实载荷求解与variational路线仍 blocked。保存源身份、契约、已完成单元块及部分累加向量、固定种子、实际 wall/peak和未执行原因；corrupt/exception记 failed，任何几何/代数失败不修输入、不换CAD、不提高预算。不会把矩阵guard、P2运动学代理或 source signed scalar stress称作 solver strainGT/工程验证。

本页仅阅读源码、小元数据及 primary 文献；未读取 rawdeck/科学数组/GT/权重，未 import/AST/构造模型/运行GPU或实际pilot。
