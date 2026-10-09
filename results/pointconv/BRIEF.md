# PointConv点卷积求解器：固定480初筛

一个新PointConv家族，原25含实际载荷、direct4、k32及FPS2048/128点层级。全部原节点fine/identity skip/输出完整；stride仅局部邻居，不声称每点上粗层。

| 固定端点 | 位移 mean / worst % | signed stress mean / worst / hotspot % |
|---|---:|---:|
| known15 | 85.09 / 139.74 | 98.04 / 101.15 / 99.50 |
| DEV5 dia | 73.77 / 78.91 | 89.66 / 91.69 / 96.60 |

实际训练277.93秒，raw wall 304.44秒；参数712131。缓存准备、读入/H2D、推断与peak分列于JSON，重叠计时不相加。

两场均值未优于历史FNO480，本配置当前没有精度选型优势。 四pass不是收敛或整个家族有效性的结论。

历史同480 known15，仅作上下文：

| 方法 | U / S mean % | 实际训练秒 |
|---|---:|---:|
| FNO direct4 | 47.51 / 66.71 | 99.78 |
| MLP direct4 | 71.59 / 97.60 | 43.70 |
| Multiscale mesh | 106.48 / 99.30 | 193.91 |
| GINO | 54.42 / 72.46 | 740.12 |
| FNO force-free u9 | 35.03 / 76.97 | 104.18 |
| BC DeepONet u9 | 68.86 / 95.36 | 52.29 |

这些结构的参数、局部交互与输入图不同，不是同容量或拓扑单因素因果实验。完整曲线、最差ID、首正式全案例梯度收据、实际层级节点数/边数/P-R支持与物理单位见[metrics.json](metrics.json)。

support-reference角色不等于节点Dirichlet，密度补偿不是FE积分/刚度。DEVdia已探索；旧final和新sealed test保持关闭。本初筛不新增P2监督或应变GT声明。

![实际固定端点精度与成本](comparison.png)
