# GeoBC：一分钟结果

**目前继续以FNO为主候选；Hash保留应力热点对照，三版紧凑Transolver短筛已完成。最终求解器尚未选定。**

同原版DeepJEB、100训练CAD×三载荷、物理损失，在20个knownDEV新CAD×三载荷、全原节点上的结果：

| 方法 | 训练遍数 | 位移误差 | 应力误差 |
|---|---:|---:|---:|
| FNO | 5 | 31.61% | 56.10% |
| 全局Hash | 5 | 31.37% | 54.45% |
| FNO | 10 | 25.60% | 52.14% |

同5遍Hash联合优势只有0.94个百分点，未达推进门槛，停止追加；FNO追加5遍纯训练97秒，改善明显，但不能当同曝光选型结论。原Hash<1%是已知单样本拟合，不能与新CAD误差混比。

原全节点++物理5遍训练约32分钟，TRAIN应力仍93.81%。新三版紧凑TS同5遍已完成，原版最好：新几何U/S52.06%／93.20%，仍落后FNO31.61%／56.10%；三者TRAIN应力也约94%，当前配方拟合不足。相同R16输入FNO控制已完成，新几何应力64.98%，明显优于TS但差于原R64 FNO56.10%；压缩本身不足解释TS差距。R64大容量同5遍已完成，新几何U/S29.03%／54.45%；参数增3.138倍仅改善联合2.11pp，未达推进门槛，不追加长训。空token控制已完成，应力93.15% vs原版93.20%，没有明显改善。原TS／FNO双模型训练目标控制已完成，同5遍标准化MSE使新几何U/S分别变差到65.62%／94.60%和48.71%／72.14%，停止此路线，不追加长训。先5遍、有学习再10遍，不统一40遍／24000步。40遍耗时是整轮总计：已有FNO整轮16.61分钟，不再乘40。

[同5遍Hash对照](geobc_global_hash_physical_v18/BRIEF.md) · [FNO5→10](geobc_physical10_v16/BRIEF.md) · [++训练负结果](geobc_native_physical_TRAIN_v13/BRIEF.md) · [三版精度比较](geobc_coarse_three_physical5_v21/BRIEF.md) · [R16 FNO控制](geobc_coarse_fno_control_v23/BRIEF.md) · [R64容量控制](geobc_wide_fno64_v25/BRIEF.md) · [空token控制](geobc_occupied_ts_v27/BRIEF.md)。

保留signed U(mm)/S(MPa)，按GRID对齐CSV与H5原节点；旧H5-only结果撤回。knownDEV已用于迭代、单seed／可用官方TRAIN来源；先前FINAL已揭盲，未来不再用来调参。模型容量和表示不同，不能宣布架构上限；连续接口一致不等于独立离节点FEM精度。[冻结测试](geobc_FINAL_v9/BRIEF.md)。
