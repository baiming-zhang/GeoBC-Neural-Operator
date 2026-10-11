# 冻结全局解后的局部残差：5遍筛选

**原版DeepJEB，100 TRAIN CAD×3载荷；knownDEV20 CAD×3载荷，全部原节点。**

| 方法 | 位移误差 | 应力误差 | 热点误差 | 累计更新窗口分钟 |
|---|---:|---:|---:|---:|
| Intact FNO 10 | 25.60% | 52.14% | 41.93% | 3.23 |
| FNO5 + Hash5 | 31.52% | 53.75% | 39.02% | 2.50 |
| FNO teacher 5 | 31.61% | 56.10% | 45.08% | 1.61 |

另一残差MLP的位移／应力／热点为31.14%／55.87%／44.75%。四方法完整指标、每载荷20例、配对20CAD及最坏病例保存在metrics.json。

最佳残差相对原FNO5：位移-0.09、应力-2.34、热点-6.06个百分点；相对完整FNO10为+5.91／+1.61／-2.91个百分点。

本轮有限方法池综合领先：Intact FNO 10。两残差未通过既定联合／应力各改善3个百分点的推进门槛；不自动追加40遍或24,000步，不宣布最终架构上限。

两残差共同TRAIN整轮4.07分钟，包含共享缓存、一次300例冻结全局特征、初始化、更新、原节点读出、保存和退出。knownDEV整轮119.13秒。MLP／Hash新增更新窗口分别39.71／53.26秒。表中的累计窗口计入已有teacher5一次；共享新缓存和旧teacher整轮IO另计，不重复归零。UTC／monotonic跨度不同，两轴均保留，不冒充精确纯CUDA时间。

真实最大325,387节点在训练autograd开启时初始预测逐点等于teacher；mutable梯度完整有限、冻结teacher/reference无梯度且前后hash不变，随后两arm各1500次更新。原R64输入网格只供一次冻结特征提取，消费后释放；原节点/stencil保留。采样GPU总峰值约22.58GB，上限30.5GB；外部workspace精确峰值未知。

teacher此前5遍，mutable decoder额外5遍，reference0；不是fresh5或所有参数10。完整FNO10继承Adam并采用第6–10遍顺序；残差重新初始化Adam及5遍顺序，属于效果／成本比较。MLP 26,884 trainable，官方TCNN实际7,745,920 trainable，另有同尺寸冻结reference；FP32对原生FP16/FP32 master，容量／精度不是单因素因果对照。残差加在encoded4后经旧TRAIN40-only scaler物理解码，signed U(mm)/S(MPa)直接监督，非teacher蒸馏。

先提交两arm×60=120份finite全N encoded/physical不可变预测并验hash，随后共享knownGT60一次。旧FNO5/10只复用已评分标量，不重读PT/GT或重跑推理。knownDEV已用于探索，不是新盲测；FINAL本轮0且先前已揭盲，不用于迭代结论。连续接口的C0取决于显式native25延伸，尚未测独立离节点FEM精度。
