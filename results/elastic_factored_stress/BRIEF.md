# 固定 q 的 signed-factor 粗筛

固定 TRAIN3/DEV2、seed2026、60 次完整案例更新；原 signed stress 标签不变。实际阶段 4.791s，训练 1.011s。

联合门 **FAIL**：DEV 全节点 signed L2 均值/最差 30.408%/30.826%，热点均值 29.722%；两个精度门均未通过。

| DEV CAD（ver） | 全节点 signed L2 | top1% 热点 L2 | S² 能量加权 sign-error |
|---|---:|---:|---:|
| 102_249 | 30.826% | 32.672% | 2.356% |
| 102_38 | 29.990% | 26.771% | 1.788% |

能量 sign-error 均值/最差为 2.072%/2.356%，通过 ≤5% 门，但未满足各 DEV signed L2<25% 与热点均值<25%；关闭此固定 representation，不追加预算、seed、宽度或损失调整。

DEV 范围下界均值 9.772% 仅描述输出被限制在 [-q,q] 时不可消除的幅值误差；它不是 oracle 预测，也不证明原 signed S 的符号约定。

冷成本沿用原物理端点的 77–101s，加实际 σ 恢复、incidence 与本次必要 head 部分；完整 accounted cold 为 78.911–103.486s（约79–103s）。此 head 缓存旧端点，未测量新的联合 fresh cold，禁止与 coarse 31–42s 合并计算。

缓存增量约 1.193–1.482s，不是完整 cold；训练一次性成本另列。完整分项与未计入的旧 xyz/25/cloud 准备成本保留在 metrics.json。

真实收据：GT 5/S 5，TRAIN 3/DEV 2；σ 5/valid 5，尝试与完成均有逐案例账本。两份 DEV 全节点预测先落盘，之后才读取任一 DEV 缓存 signed S。

仅固定五 CAD 的既有完整 FE 输入接口证据，不作为跨接口 solver 排名或工程验证。无新求解、K guards、scaler fit、原始标签成员/位移/旧 final/DEV dia/外部权重读取；新 test 仍 sealed。
