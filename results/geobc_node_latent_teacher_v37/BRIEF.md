# 固定teacher中间场：TS能否学会该算子映射？

**同一TRAIN teacher的编码器／输出头冻结，只训练新的TS或FNO算子各5遍；这不是10遍FEM监督，也不是fresh5公平选型。**

下表均为20个knownDEV新CAD×三载荷、全原节点；teacher节点latent误差属于固定特征坐标，不是物理场误差。

| 方法 | teacher节点latent误差 | FEM位移误差 | FEM应力误差 | 新增纯训练分钟 |
|---|---:|---:|---:|---:|
| FNO16 teacher physical5 | 0.00% | 30.78% | 64.98% | 0.00 |
| TS frozen-head latent5 | 98.66% | 90.06% | 98.37% | 2.23 |
| FNO16 frozen-head latent5 | 8.73% | 31.42% | 65.90% | 1.62 |

TRAIN300上的节点latent拟合（初始→5遍端点）：
- TS frozen-head latent5：107.32%→98.62%，固定诊断门槛未通过。
- FNO16 frozen-head latent5：100.05%→7.56%，固定诊断门槛通过。

新增训练整轮7.49分钟；DEV整轮3.39分钟；teacher旧整轮4.19分钟／旧纯训练102.11秒另列。表中teacher新增成本为0，指复用已有训练，绝非teacher免费。推理计时包含额外latent诊断，不与历史纯推理速度比较。

teacher→学生的物理U/S误差另存metrics.json；teacher自身也有原FEM误差。学会teacher映射不意味着FEM达到高精度，更不等价新的离节点FEM验证。

完整TRAIN300、首真实最大325387节点算子梯度通过；冻结encoder/head无梯度且hash不变，onlyoperator freshAdam1500步。CPU合成33337节点检查只检验完整链式反传，不替代真实GPU。共同teacher-R16节点latent目标按真实blockcenter三线性插值，完整16通道范数、equalcase，无epsilon／裁剪／节点采样；空网格体积不作为损失样本。

180个teacher／学生全节点encoded4／physical4不可变有限预测与完整latent标量先保存并重验，再共享knownDEV60 FEM标签一次；14旧候选仅复用物理标量。原物理TRAIN300缓存仍实际读取一次用于FEM读出，不用来更新学生；teacher目标不冒充FEM。原signed U(mm)/S(MPa)、GRID对齐、旧TRAIN40 scaler保留。

TRAIN学习门槛是端点后冻结的描述性判断，不冒称训练前注册，未用于筛掉模型或改变DEV分母。knownDEV已迭代、单seed／availabilityconditional来源；原FINAL已揭盲，本阶段不读取。TS1.882M、FNO16 4.244M、teacher同FNO架构及目标坐标均是局限，不能宣布TS架构上限。

完整17行、每载荷20、配对20CAD、热点、成本／曝光／冻结参数及未定义teacher零范数见metrics.json；不将本诊断的最低误差称全部历史最好，不自动延长40遍或24000步。
