# 精确参考点运动边界：独立约束零空间求解

固定两CAD／六解门：PASS；整阶段UTC 65.180811秒／mono 65.180812秒。

真实Hbar约束规定加载RBE参考点的平移／转角，原其他零支撑／材料／几何保持。独立求解使用几何抬升u0=Rᵀ solve(RRᵀ,target)与约束零空间Π，不读取物理基解作为抬升或warmstart；不是大弹簧近似。

| CAD | 原节点 | 三路独立零空间PCG秒 | 紧凑更新＋全N输出秒 | 最大U相对差 | 最大约束相对差 |
|---|---:|---:|---:|---:|---:|
| 50_380 | 212721 | 27.539 | 0.00528 | 5.05e-07 | 4.24e-16 |
| 624_483 | 187217 | 23.804 | 0.00143 | 3.25e-07 | 6.06e-16 |

Π=I−Rᵀ(RRᵀ)⁻¹R、v初值0，独立方程ΠKΠv=Πb−ΠKu0。原固定32步预条件器按ΠMΠ应用；完整投影空间有零方向，SPD只声称在约束零空间。每路alpha／beta／残差停止独立，真残差用实际非零投影RHS归一化，绝不用接近零的Πb归一化。每25步原子保存z／r／p／lift／rhs与真实成本。
六个完整原节点字段、约束、真平衡残差及独立反力先finite、hash、body、Cartesian屏障；之后才读12个原六基解body各一次，构造未改原始柔度的紧凑响应。抬升与独立PCG的基解body读取0。
同一reference全部六运动被指定时，施加于该reference的外载影响约束反力，实体位移由目标运动决定；Πb近零实际见证另列，没有伪构一般实体bodyload。独立lambda=solve(RRᵀ,R(Ku−b))，物理反力Dlambda，平衡含reference约束及原固定支撑反力。
紧凑totalforce= D solve(D rawC D,Dtarget)，reaction=totalforce−appliedforce。能量0.5 totalforce·target必须包含约束反力功，不用appliedforce单独冒总能量。平移mm、转角rad与反力N／Nmm按实际拟合长度规范，分量分别列出。
**验证的是声明的新精确运动边界近似方程，不是新取得的原OptiStruct支撑／转角／离节点GT。**近似curvedP2四Q／weighted-LS RBE3与未知DEBUG／PSOLID控制保留；reference GRID无原H5输出行，物理定义来自Hbar，不生成替代节点。两样本已揭盲，signedS仍未提供。
原K／diag／粗矩阵不重建，旧8fine／粗能量guards不重播；实际新投影K／M与标量及geometry-only抬升资格单列。缓存加载、资格、PCG、保存、屏障、基解读取／API、对照、失败与退出全部计入双时轴；未知工程全冷null。继承源到六基解77.94036／63.43762秒不是本阶段新成本，缓存紧凑响应与独立PCG也不是NN同预算排名。
