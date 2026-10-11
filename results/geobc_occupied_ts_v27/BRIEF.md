# Transolver：空网格token控制

**原版DeepJEB，同100训练CAD×三载荷／5遍，knownDEV20个新CAD×三载荷，全部原节点。原版TS完整初始权重、参数、头、损失和病例顺序一致。**

| 方法 | 位移误差 | 应力误差 | 热点误差 | 纯训练分钟 |
|---|---:|---:|---:|---:|
| FNO R64 width16 | 29.03% | 54.45% | 42.81% | 1.46 |
| Transolver dense R16 | 52.06% | 93.20% | 93.36% | 2.28 |
| Transolver occupied R16 | 52.29% | 93.15% | 93.28% | 2.45 |

排空token减原denseTS：DEV位移+0.23、应力-0.04、热点-0.09个百分点；TRAIN位移+0.09、应力-0.05个百分点。完整七候选联合最佳：R64_FNO_W16_1500。

TRAIN300输入中的R16占据token比例均值10.67%（最少295、最多582个／4096），其余为空。排除空token与其latent设零是耦合的路由改动，不能将结果当作纯attention掩码因果证明，也不能仅凭空token很多宣称它是误差原因。

控制训练整轮5.31分钟，其中纯训练146.94秒；DEV整轮96.57秒。UTC与monotonic原值分列，不缩放、不抹成本。

固定原版TS1,882,172参数，全部初始state SHA与denseTS相等；所有占据token一同进入原官方算子，不抽样，空latent精确零，沿用旧block-center trilinear query与native25头。真实全300GPU缓存、最大325387节点encoder/operator/head及联合leaf有限正梯度、四RNG恢复先于Adam。

同seed2026／1500均衡更新／5遍／Adam .001fused／最高FP32-TF32关闭／旧TRAIN40-only scaler／30.5GB SI。signed U(mm)/S(MPa)与CSV GRID对齐H5 vertices保留；物理原节点损失与物理指标一致。

新60份finite全N预测保存重验后才共享已知DEV60份GT一次；六个旧候选仅复用完整60例标量，无baseline推理或数组重播。rawCSV／FINAL／scalerfit0。knownDEV已用于迭代、单seed、可用官方TRAIN来源，不是新盲测，也不能宣称架构上限。

完整七行、每载荷20、配对20CAD、最坏／热点、TRAIN300、容量／显存／token数／资格与成本见metrics.json。若无明显改善停止此路线，下一阶段需解决宏观训练或条件表示问题，不默认40遍或24000步。
