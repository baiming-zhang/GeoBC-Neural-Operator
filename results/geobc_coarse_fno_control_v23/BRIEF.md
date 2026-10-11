# 紧凑R16表示：FNO与Transolver控制

**同100训练CAD／5遍、knownDEV20个新CAD×三载荷、全部原节点。FNO控制与三TS共享R16输入／连续头，容量未精确匹配。**

| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |
|---|---:|---:|---:|---:|
| FNO R64 | 31.61% | 56.10% | 45.08% | 1.61 |
| FNO R16 control | 30.78% | 64.98% | 56.89% | 1.70 |
| Transolver R16 | 52.06% | 93.20% | 93.36% | 2.28 |

同R16输入下，FNO控制减最佳TS：位移-21.27、应力-28.22个百分点；减FNO64：位移-0.83、应力+8.88个百分点。

完整五方法联合U/S最佳：FNO R64。控制TRAIN位移／应力27.21%／60.54%，TS系列TRAIN应力约94%；输入压缩本身不足以解释全部拟合差距，但算子、容量、优化与表示的相互作用尚未分开。

控制训练整轮4.14分钟（缓存134.16秒），纯训练102.11秒；新开发读出69.57秒。epoch与monotonic原值分别留存，未缩放对齐。

FNO16固定width16/modes8/4blocks、4.244M参数，比紧凑++4.523M少6.17%；同XYZ-first61 pooledR16输入、PointNet1024、native25、76→128→128→4头、真实block-center trilinear查询与全节点物理损失。FNO64 width9/modes12、4.527M，分辨率／频带／宽度不同，所以不是纯分辨率或精确容量匹配消融。

同seed2026、病例顺序／1500更新／5遍、Adam .001fused、oldTRAIN40-only scaler、最高FP32／TF32关闭；原signed U(mm)/S(MPa)与GRID对齐标签保留。未做Hash微调或自动延长40遍。

控制60份finite encoded4／physical4保存重验后才共享knownGT60一次；四个旧候选仅复用完整60例标量，无基线推理或数组重播。rawCSV／FINAL／scalerfit均0。knownDEV已用于迭代、单seed、可用官方TRAIN来源，不能称新盲测或架构上限。

五行完整结果／300TRAIN与60DEV／每载荷20／配对20CAD／最坏及热点／容量与显存／实际梯度资格与成本见metrics.json。连续接口一致不等于独立离节点FEM精度。
