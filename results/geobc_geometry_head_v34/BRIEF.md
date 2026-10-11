# 学得的几何编码器／查询头：TS与FNO迁移对照

**三控制均继承FNO16在TRAIN上的5遍状态，再新增5遍。下表累计纯训练包含一次teacher5，不当fresh5公平比较。**

| 方法 | 位移误差 | 应力误差 | teacher+新增纯训练分钟 | 推理秒／60例 |
|---|---:|---:|---:|---:|
| TS learned-head +5 | 53.01% | 94.37% | 4.08 | 1.75 |
| FNO16 learned-head reset +5 | 29.61% | 61.86% | 3.50 | 1.27 |
| FNO16 intact 5 to10 | 27.65% | 58.96% | 3.53 | 4.13 |

原TS fresh5 U/S52.06%／93.20%；原FNO16 teacher5 30.78%／64.98%。
本报告14个混合曝光候选中描述性最佳：R64_FNO_W16_1500；TS系列描述性最佳：R16_TRANSOLVER_1500。该排序仅决定后续优先级，不能抹平容量、表示、继承状态与曝光差异。

新几何差值（负值更好）：
- TS_warm_minus_old_fresh5：位移+0.95、应力+1.18、热点+0.14个百分点。
- FNO_reset_minus_teacher5：位移-1.17、应力-3.11、热点-3.31个百分点。
- FNO_continue_minus_teacher5：位移-3.14、应力-6.02、热点-7.78个百分点。
- TS_warm_minus_FNO_reset：位移+23.40、应力+32.51、热点+39.93个百分点。
- TS_warm_minus_FNO_continue：位移+25.36、应力+35.42、热点+44.39个百分点。

本轮三控制整轮9.03分钟，DEV167.87秒；teacher旧整轮4.19分钟／旧纯训练102.11秒另列。三控制共享一次新GPU缓存和一次teacher PT读取，旧teacher不重播三次。表中纯训练／推理不包括缓存与检查点等整轮成本；UTC与monotonic保留原值。

TS与FNO重置控制：各自seed2026完整fresh state先与原控制SHA匹配，再严格复制同一TRAIN teacher的encoder/head；各自operator前后hash保持，Adam全新。完整FNO续训控制：全state、Adam1500步、四RNG、cursor和每case5次曝光保留，继续到3000／10遍。所有组仍可训练，原R16输入、native25、query、参数规模、旧scaler、原物理损失不变。

新增1500次更新都使用固定10遍均衡顺序的后5遍，前5遍hash与teacher旧顺序一致；重置控制仅记录新增5遍算子曝光，encoder/head继承teacher5；完整FNO全部参数10遍。旧fresh5使用前5遍顺序，因此与旧fresh5的差同时含初始化、信息曝光和顺序变化；不是纯架构或纯初始化因果消融。TS1.882M与FNO16 4.244M也不是容量匹配。

teacher只从原100TRAIN×三载荷得到，不用DEV更新、不产生teacher预测伪GT。所有监督始终是原CSV GRID对齐H5 vertices的signed U(mm)/S(MPa)，旧TRAIN40 scaler无重拟合。FP32最高精度／TF32关闭、30.5GB SI。

实际共享全300TRAIN GPU缓存，三arm最大325387节点encoder/operator/head及jointleaf正有限梯度先于任何新增更新，并恢复相应RNG；全部300物理TRAIN读出复用缓存共900次。CPU合成扰动teacher的33337节点含569尾块只检验迁移数学与完整链式梯度，不是真实训练teacher或GPU资格替代。

180新wholeN finite immutable预测保存并重验后才共享knownDEV60目标一次；11旧候选只复用物理标量，不重跑旧推理或旧数组，rawCSV／FINAL／scalerfit0。knownDEV20CAD×三载荷已用于迭代，单seed、availabilityconditional官方TRAIN来源，不是新盲测或架构上限。

完整14候选、全部ties、每载荷20、配对20CAD、TRAIN300、初始化／teacher／Adam／参数／显存／成本见metrics.json。根据物理误差及实际成本决定下一宏观阶段，不自动40遍或24000步。

本表未纳入既有R64 FNO10遍／40遍端点；完整求解器判断仍需参考这些较长训练结果，不将这里的14候选最低误差称为全部历史最好。
