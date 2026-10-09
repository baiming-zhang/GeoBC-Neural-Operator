# 物理位移＋学习signed stress读出（proposal_only）

2026-10-10。五CAD物理u已5/5真收敛、原节点误差约0.0162%（root回报）；完整GeoBC的signed scalar stress尚未解决。若冻结FNO初值不过20%成本门，不救初值/Hash参数，而考虑这一整体混合候选。本文仅读既有语义notes/源码，不改当前warm门或源码，不授权实验、GT访问或训练。

**可用但非GT的物理输入。** 用已冻结的近似P2解、原全部TET10节点与缓存physical梯度/四正权Q：
`G=Σ_a u_a⊗∇xN_a; ε=sym(G); σ=2με+λtr(ε)I`。
σ单位MPa，六分量`xx,yy,zz,xy,xz,yz`（tensor shear），不是源scalar公式。最廉价明确恢复：每cell按Q物理权重求mean σ，再按incident cell的`V/10`正权归一到全部十节点。无incident cell的reference节点保留输入validity mask，张量槽置0只表示缺测，不称物理零stress；不丢节点。它是自定义平滑proxy，不是OptiStruct GPSTRESS外推/平均，不能把派生ε/σ称strain/stress GT。

一次真实特征构建复用既有几何/算子guard，不重复解算；新增检查FP64物理mm梯度、partition/affine一致性、tensor对称与finite、正权/归一恢复保持常数、节点覆盖/缺测及exact row身份。公式无GT参与，不用normalized xyz差冒充mm，不clip异常。正式plan再固定数学容差，无单独toy/pilot链。

**无需新下载的粗判别。** 固定这3TRAIN＋2DEV/ver；新fresh scalar读出两臂共享原xyz/25/global32、网络布局/初始化与S loss，一臂输入恢复σ6，另一臂同六槽为0；两侧都保留同validity mask。只学原signed MPa标签，沿既有TRAIN-only asinh S scaler，不abs、改符号或加`C:ε=source_scalar` loss。可固定单seed2026、120完整case updates（3TRAIN×40轮），相同case序、端点比较，不按DEV挑step/宽度/seed；构建与训练总成本预先设硬界。DEV全原节点signed S相对L2、MPa RMS/max、worst/hotspot与完整冷成本并列。只有读出明显优于同新布局直接学习，且物理solve＋恢复＋推理仍有成本机会，才值得扩大到多load/CAD；它只是输入信息粗试，3TRAIN不证明通用求解器。

**两项主要风险。** [已核语义](geobc_stress_semantics.json)：论文称signed von Mises，但exporter的符号、scalar计算先后、节点外推/平均未知；因此不能用选择某个VM/sign公式的巧合拟合当定义确认，自定义恢复可能抹掉hotspot。第二，物理冷solve是每次新增请求的真实费用，小MLP廉价不等于整体廉价；权重、特征准备、恢复和完整solve都需计费。

旧final/newsealed/dia不开，当前原始数组/GT/权重/模型/GPU访问均0；候选须等待warm结果与root独立冻结，不自动启动。
