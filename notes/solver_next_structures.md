# 下一求解器结构：待响应基结果后择一（proposal_only）

2026-10-10，仅源码/小JSON/原始研究，无数组、权重、代码或实验执行。7家族并非都已收敛：FNO600秒DEV端点U约36%/S56%；GINO480需740秒、U54.4%/S72.5%，尚无成本优势。组合载荷与梯度仍有缺口。

**优先1：几何/BC DeepONet，低成本低秩场生成。** 固定rank64，`z_c(q)=Σₖb_ck(a)t_k(q)+β_c`；branch用现1024固定点的xyz+非force原19槽及load6，trunk用xyz+同19槽。变几何set branch是适配，不是原文固定传感器复现。新cache需保留cloud原节点索引/xyz身份/角色覆盖；现cache仅存xyz，禁止猜ID−1、最近节点复制或补采样。最小配对：共同新branch与query输入，concat-MLP vs branch×trunk，fresh同seed/signed4/scaler，实际参数差单列。若用u9，两边共同采用alpha/一次均值偏移、S独立非线性，不扫rank/head。它不走FNO体素/FFT或GINO积分边，与现MLP的区别是显式场因子化；低秩可能丢尖峰，BC flags的off-node语义仍缺失，可查询不代表梯度/BC合格。[原论文](https://doi.org/10.1038/s42256-021-00302-5)、[作者代码](https://github.com/lululxvi/deeponet)。

**优先2：完整多尺度mesh operator，成本更高。** 原节点latent32在真实角—中点边更新，经2048/128连通粗层固定4次learned V-cycle，直接输出全节点signed4；边核用相对xyz/L、长度、材料/BC，保留L/原25。必须有细层交互、学习跨层传输，区别于旧4CAD浅层“粗tokens→固定P→Hash”。需将目前仅4CAD的真实边/角色ownership/硬mass R/明确P扩到45CAD。机制配对只启用vs旁路跨层通路，共同细层核/宽度/迭代；旁路有效容量较低必须说明。跨家族筛选可直接对共同MLP/FNO，不追加机制实验。N约18–27万，单N×32 FP32约23–35MB；既有四CAD有向edge×64激活约157–236MB/层，均算术估算，反传另计。分块/checkpoint及通道分解避免edge×32×32存储，属于实现适配；图不是刚度、mass非精确FE积分，不继承mesh不变性或弹性保证。[MGKN原论文§3](https://proceedings.neurips.cc/paper_files/paper/2020/file/4b21cf96d4cf612f239a6c322b10c8fe-Paper.pdf)、[官方旧实现（deprecated）](https://github.com/NeuralOperator/graph-pde)。

**试验/结果分支。** Root只择一：40TRAIN/5DEV×known3、共享TRAIN-only scaler、全N signed4 MSE、同case顺序/权重、480更新/四pass/seed2026，报告物理U/S均值/最差/应力热点与实际训练/全N推断/显存。慢收敛不淘汰，有成本机会才冻结600秒+第二seed。若force-free u9改善known3与DEV-dia且S不明显退化，优先DeepONet、两臂共同保留响应结构，检验省去grid的代价；若两臂仍迁移弱或值改善但梯度差，优先完整mesh，**不是缺拓扑的因果证明**。FNO若已有足够收益应先确认，不自动派两家。DEV-dia已探索，旧final不开发、新sealed test不解封；连续/BC能力仍需独立冻结检查。
