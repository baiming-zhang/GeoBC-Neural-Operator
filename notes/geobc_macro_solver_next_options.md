# 下一宏观数值求解器候选（proposal_only）

日期：2026-10-10。仅供 root 在应力读出粗筛后决定；未实现、冻结或派发实验。
依据：现有 `solver_exploration_next_decision.md`、少量 elastic 源码及依赖文本；没有读数组、GT、权重，未运行检查、安装、训练或 GPU。

**建议：检验约束掩码 P1-derived Galerkin 两层加性预条件，先不用新依赖。**
已接受的五 CAD 物理解 U≈0.0162%，真实残差全部≤1e-8，cold≈77–101秒；FNO 初值未过成本门。
新的宏观问题是：能否用粗空间同时修正多个相邻自由节点的误差，显著减少整体求解成本。
这与进一步换初值、调 Hash 宽度不同；它仍不解决原 signed scalar stress 的读出语义。

## 保留的科学对象

- fine 算子仍是原 curved-P2、同四点正权积分、MAT1、全部十节点及相同 free/fixed 分区。
- SPC/RBE2 产生的零约束保持原值；不得拿角色 flags 代替约束，也不减少 fine 自由度。
- 已接受的 RBE3 weighted-LS 载荷转移 b 不变；输出-only reference 仍由原 Hbar 恢复。
- 不重网格、不把弯曲中点挪到直边、不换成直四面体 P1 物理解、不改载荷或积分。
- 不读取 GT；旧零初值的收敛物理解只作为数值同解参考，不作为监督或精度标签。

## 粗空间与 SPD 条件

令 fine reduced A=R_free K E_free；D 是已接受的自由节点 3×3 block-Jacobi 对角。
原 CTETRA 角点作为粗节点；P0 在角点注入、在每条对应中点取两端角点值的 1/2。
仅保留实际 free 角点列，取 P=R_free P0 E_corner，再逐分量使用 P⊗I3。
角点注入行使列独立；先按原 GRID/cells10 证明全局 corner/mid 身份一致、每个中点唯一对应同一条边。
fixed 行自然不在 reduced P 中，因此 coarse 修正不会违反 fine 约束，derived reference 不成为粗自由度。
若 fixed 中点的两个角点仍自由，这个掩码空间**不是严格嵌套 P1 空间**；如实命名 P1-derived algebraic coarse，不声称纯 P1 多重网格。

粗算子必须采用 Ac=PᵀAP，以原 P2 梯度和物理权重做局部 Galerkin 装配。
不得另用 straight-P1 的几何/积分公式；这样 coarse 修正作用于原 fine 方程。
在 A 为 SPD、P 满列秩的条件下，yᵀAc y=(Py)ᵀA(Py)>0；改变预条件器不会改变目标解。
现有八个代数守卫只是必要检验，不是任意网格 SPD 的完整证明，也不表示工程资格。

冻结一个廉价粗逆，避免先引入 AMG 框架：Dc=diag(Ac)，B=Dc^-1/2 Ac Dc^-1/2。
取 γ=max_i Σ_j |B_ij|，ω=1/γ；从零初值执行固定 m=32 次阻尼 Jacobi。
对应 Sc=Dc^-1/2 [ω Σ_(j=0..31)(I−ωB)^j] Dc^-1/2，是固定线性对称算子。
若 Ac SPD，则其特征值满足 0<λ(B)≤γ，因此上述多项式在谱上为正，Sc SPD。
新预条件 W=D^-1+P Sc Pᵀ；对非零 r，rᵀWr>0。两项系数均固定为1，不扫权重/次数。
用这个固定 W 的普通 PCG；不得把残差停的内层 CG 或变化 V-cycle 直接塞入普通 PCG。
Ac 需 FP64 对称装配，Dc>0、γ有限，粗应用数值对称/能量必要守卫只在真实阶段执行一次。
若失败，显式失败；不加 shift、伪逆、降精度或忽略元素来救回。

## 最小两 CAD 粗筛与硬门

固定 101_428/ver（升序首 TRAIN）与 102_249/ver（升序首 DEV）；不按新结果替换样本。
各从零初值求同 fine 系统；旧同 CAD 完成的 zero/block-Jacobi 解及成本是冻结参考，不重新跑 baseline。
每 CAD 新 coarse 构造最多60秒；求解最多10000次或累计300秒，先到即停；总新阶段累计 wall≤900秒。
总钟覆盖源码/元数据、输入载入、映射、稀疏装配、合并、传输、粗缓存、守卫、求解、真残差、同解检查和写盘。
fine 已接受 cache 可复用，不能把旧 fine setup 成本记为0：完整 cold 账本加上其可核验 setup，注明它是 accounted cold，非新鲜重测 wall。
分列报告实际新 stage wall、完整 accounted single-query cold、cached solve；同步时间计入，不能只比较迭代数。
计数至少分开 fine K 次数、coarse sparse matvec 次数、P/Pᵀ次数、fine/coarse预条件时间及 GPU 峰值。
沿原规则每25步/候选停点/最终停点计算真实 ||b−Az||/||b||，必须≤1e-8。
两 CAD 均全部原节点恢复；与各旧收敛物理 u 的 relative-L2≤1e-5，零参考用 RMS≤1e-12mm 的既有规则。
保留实际 checkpoint/已花次数与 wall；失败或恢复不重置预算，GT/旧 final/新 sealed test/权重读取均为0。
粗筛通过门：两 CAD 都真收敛、同解、完整成本账本，并且各自完整 cold 至少降低25%。
任一不满足就关闭这个固定32步 P1-derived 两层候选，不救次数/权重/样本/预算；通过后才决定扩到现有五 CAD。

## 明确阻塞与取舍

1. 当前源码没有粗映射/Ac/稀疏应用；角点数、nnz、角色一致性和显存不能从总 N 推定，需首次真实构造记录。
2. COO 临时装配及去重可能比最终 CSR 大；建议实际峰值上限24GiB、无换样本/CPU回退，超限就失败。不能凭 sparse 名称假设便宜。
3. 原依赖锁含 SciPy 1.18.1、Torch 2.14.1；实际 CUDA FP64 CSR/transpose 路径仍未验证。首次真实阶段若不支持即报告具体 API 阻塞，不开独立资源预检链。
4. 同网格 P2→P1 是 p-coarsening，未缩小网格尺度；32次 coarse Jacobi 对极低频可能仍弱，且每次 PCG 增加32个粗 matvec，净收益未知。
5. 支持中点掩码可能削弱粗空间的刚体/低频表示；记录掩码比例及必要守卫，不把 negative 结果外推为所有 AMG/多层方法都无效。
6. 因而现在不值得安装 PETSc/HYPRE/CuPy 或完整求解框架，也不以纯 CPU 稀疏直接分解作 GPU 速度对照。
   真正 h-coarsening/弹性 AMG 是后续独立宏观候选；只有本轮揭示低频仍主导且 GPU sparse 路径成立，再评估依赖和对称固定 V-cycle。

这份提案只给下一有界结构性候选；最终 solver 仍需位移、signed stress、跨 CAD/载荷与冷启动成本共同满足目标。
