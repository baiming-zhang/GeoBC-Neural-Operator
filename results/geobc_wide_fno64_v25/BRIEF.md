# R64 FNO：同5遍容量控制

**同100训练CAD×三载荷、knownDEV20个新CAD×三载荷、全部原节点物理误差。保持R64/modes12/layers4，仅隐藏宽度9→16；参数4.527M→14.206M。**

| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |
|---|---:|---:|---:|---:|
| FNO R64 width9 | 31.61% | 56.10% | 45.08% | 1.61 |
| FNO R64 width16 | 29.03% | 54.45% | 42.81% | 1.46 |
| Transolver R16 | 52.06% | 93.20% | 93.36% | 2.28 |

大容量减原FNO：位移-2.57、应力-1.64、热点-2.27个百分点；联合U/S差-2.11个百分点。预设推进门槛是否通过：False（联合改善至少3pp、任一场恶化不超过2pp）。

六方法同5遍联合最佳：FNO R64 width16。大容量TRAIN位移／应力25.29%／47.64%。训练整轮3.91分钟，纯训练87.82秒，DEV读出76.85秒；epoch与monotonic分别保留。

新模型只有FNO隐藏width9→16，输出latent16／R64网格／modes12／4blocks／PointNet32／native25／76→128→128→4头／原占据节点stencil／原物理损失保持一致。参数增为3.138倍，初始共享encoder/head与seeded基线逐项匹配；全300GPU缓存和最大325387节点真实全反传资格先于Adam。

同seed2026、固定1500更新／5遍均衡曝光、Adam .001fused、oldTRAIN40-only scaler、最高FP32／TF32关闭；原signed U(mm)/S(MPa)、CSV GRID对齐H5 vertices。独立fresh训练，不自动追加40遍或24000步，不调整Hash小超参。

60份全N finite预测保存重验后才读取已知DEV60份物理GT各一次；五个旧候选仅复用完整60例标量，无baseline推理／数组重播。rawCSV／FINAL／scalerfit均0。knownDEV已用于迭代、单seed、可用官方TRAIN来源，不能称新盲测或架构上限。

完整六方法、每载荷20、20CAD配对、最坏与热点、TRAIN300、参数／显存／真实成本见metrics.json。是否扩大训练需另按证据决定；连续接口一致不等于独立离节点FEM精度。
