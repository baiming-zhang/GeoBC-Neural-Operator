# 近似物理求解器：单 CAD 位移精度—求解成本候选

2026-10-10，**proposal_only**。建议值得一次有限尝试：曲线 TET10 各向同性小变形弹性、真实 SPC/RBE2 消元、明确定义的 weighted-LS RBE3、GPU FP64 matrix-free PCG。它直接测试物理离散能否解释位移，既不训练新黑箱，也不宣称复现 OptiStruct。PointConv 已有结果不触发追加预算；本页不释放任何实际任务，不修改原 mechanics note。

**输入先锁规则，随后只读一个 TRAIN101_479。** 已有元数据为179316节点、111754 TET10；旧解析器保留 SPC、RBE/载荷与材料，却跳过 free-field 卡片并丢失 GRID CD/PS，不能直接充当力学 reader。新契约只取 `data/raw/SolverDeck/101_479.fem` 的 ver 静态子工况及已下载 Mesh/H5 的输入几何、canonical GRID↔H5 行映射。逐卡保留 SPC/SPCADD DOF/值、RBE2 CM/连接、RBE3 REFC/Ci/weight、FORCE/MOMENT 参考点和 LOAD 展开、CTETRA→PSOLID→MAT1 E/G/ν；单位 mm/N/MPa，E>0、−1<ν<0.5，显式 G 与 E/ν 不一致不能猜。首版明确只支持展开 GRDSET 后 CP=CD=0：空字段继承，显式0覆盖；PS展开为永久约束。非零坐标系、UM/热载荷、约束冲突/依赖环，以及影响静态方程的未支持卡片均明确停止；输出控制只能按预先列明的非物理 allowlist 忽略，不再跳过整类 free-field。[GRID/GRDSET 语义](https://help.altair.com/hwsolvers/os/topics/solvers/os/grid_bulk_r.htm)

**RBE3 采用公开方程的明确新模型近似。** 对真实成员 Ci=123、正权重 W，在 basic frame 取 rᵢ=xᵢ−xref、L=mean‖rᵢ‖>0，Bᵢ=[I,−[rᵢ/L]×]，
\(\bar q=Hu,\ H=(B^TWB)^{-1}B^TW,\ \bar q=[t,L\theta]\)。这是已发表 weighted rigid-motion fit 的平移成员特例；L只作代数列缩放，不改变目标。用实际 REFC 选行；载荷 \(\bar g=[F,M/L]\) 由虚功转移为 \(f=H^T\bar g\)，不均分力矩。缩放 normal matrix 条件数>10¹²或 rank<6直接停止，不换伪逆/权重。检查 HB=I、虚功、合力与关于原参考点的合矩；反力是求解后的约束残量，不是 H 的输入。[Sandia 原始 theory，附录C](https://digital.library.unt.edu/ark:/67531/metadc843452/m2/1/high_res_d/1031884.pdf) 这不是 OptiStruct exact，尤其不覆盖其退化/转动权重/UM策略。[官方 RBE3](https://help.altair.com/hwsolvers/os/topics/solvers/os/rbe3_bulk_r.htm)

**完整离散，不把角色当约束。** 保留全部节点的3个位移 DOF、10节点 P2 几何与位移；RBE2 按实际 CM 使用 uᵢ=tmaster+θmaster×rᵢ，参考旋转及 CM456 所需辅助转动通过真实 RBE2/SPC 方程消元，不凭空添加转动刚度。孤立零能量转动、RBE3参考点额外约束等首版未覆盖关系需给具体 unsupported 原因。建立稀疏 u=Tz+u₀，求 \(A=T^TK_qT,\ b=T^T(f-K_qu_0)\)，不把不定约束鞍点系统交给 CG。复用 `geobc/continuous_query_geometry.py` 的 `shape_gradient/J[c,r]`，物理梯度 J⁻ᵀ∇ξN；\(ε=\mathrm{sym}\nabla u,\ σ=2με+λ\mathrm{tr}(ε)I\)，μ=E/[2(1+ν)]、λ=Eν/[(1+ν)(1−2ν)]，Kq由 w|detJ|σ∇N scatter 得到。[弱形式与材料系数](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_elasticity.html) 固定四点 (a,b,b,b) 置换，a=(5+3√5)/20、b=(5−√5)/20、w=1/24；曲线映射下是近似积分，不能凭 sampled J 与刚体检查证明无额外零模态/全域可逆，更不等于原软件积分选项。局部梯度是本模型位移导数，不是 strainGT；signed scalar stress 语义未知，本轮不读取/比较它。

**固定预条件与资源边界。** 分块 GPU gather/scatter，不存 cell30×30 K 或全局 K。缓存4Q物理梯度约107.3MB（102.3MiB，仅梯度，其他工作内存另计）；完整 FP64 位移向量约4.30MB。唯一预条件是消元后独立 DOF 的 block Jacobi：普通节点≤3×3，保留刚体主节点≤6×6，\(D_J=\sum_{e,q}w|\det J|(\mathcal B_{eq}T_{eJ})^TC(\mathcal B_{eq}T_{eJ})\)。必须包含同一块经 RBE 映射产生的交叉项；小块 Cholesky失败不加 shift 救援。[MFEM 矩阵无关弹性块 Jacobi](https://docs.mfem.org/4.10/classmfem_1_1ElasticityDiagonalPreconditioner.html) 仅作算法依据，不声称本机后端已可用。局部块不能解决全局弯曲/细长结构低频、劣质网格或近不可压缩病态；CG需要算子与预条件正定。[PETSc CG 条件](https://petsc.org/release/manualpages/KSP/KSPCG/) 本轮不临时切 AMG 或增加粗层，实际工程工作是完整卡片/稀疏消元、FP64几何与可恢复 PCG，明显多于一个网络 adapter；GPU性能与工时均未验证。

**只有两阶段，终点在读标签前冻结。**

1. **一次输入/算子 stage**：上述单 CAD 输入、全积分点几何、真实约束/材料/载荷映射与预条件构建；最多8次全K apply（6刚体列+seed2026的u/v），检查对称、非负能量/刚体null、约束消元/刚体约束秩、H reproduction/虚功/力矩，采用记录分母的无量纲相对1e−9政策阈值。总 wall 上限300s，超时保留 receipt/可恢复输入块，不另开无界审计；通过只表示这一离散可进入求解。无标签/预测/训练，失败不改CAD、积分或边界。
2. **一次冻结 ver solve**：同一输入、零初值 PCG，最多2000 iterations **或600s累计求解wall先到即停**（含K apply、预条件、reduction、真实残差验证；setup单列）。每25步及终点重算 \(\|b-Az\|/\|b\|\)，仅真实相对残差≤1e−8算数值收敛；非有限、pᵀAp≤0或rᵀM⁻¹r≤0立即失败。保存迭代向量/计数/累计成本/契约，恢复不重置预算。有限预算终点仍保留有限解和未收敛标记，不追加救参。先完成全N物理位移、自由残差、约束残量/反力及全局force–moment balance，再只读一次既有 TRAIN101_479/ver 的 canonical aligned displacement，报告 allN U relative-L2、vector RMS/mm、最大节点误差与固定区域分项；禁止挑载荷或用GT调容差。无训练/scaler fit，旧final、DEVdia、新sealed均关闭。

**判断只基于这次真实证据。** receipt分列 input/setup/preconditioner/solve/verification/GT-read/wall、GPU peak、迭代/K调用、约束与载荷守恒及 allN U误差，不将算子检查当精度结果、不将小残差当原软件等价或工程验证。收敛且U仍差，说明该材料/边界/近似积分模型与标签存在差异，停止当前候选；未收敛则结论是固定 Jacobi-PCG 成本不合适，保留残差—cost证据，另行决定是否值得物理粗层，不开启第二路线扫描。只有结果显示有机会才另行冻结跨CAD研究，不扩本轮45CAD。

本页仅源码、小元数据与 primary 文献分析；未读取 rawdeck/科学数组/Field/GT/权重，未 import/AST/模型/GPU/任何实际pilot。
