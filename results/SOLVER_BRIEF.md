# GeoBC：一分钟结果

**目前优先推进FNO；Transolver当前配方拟合不足，最终求解器仍未确定。**

原版DeepJEB，100训练CAD×三载荷；20个knownDEV新CAD×三载荷，全原节点物理误差：

| 方法 | 训练遍数 | 位移误差 | 应力误差 |
|---|---:|---:|---:|
| FNO | 5 | 31.61% | 56.10% |
| FNO | 10 | 25.60% | 52.14% |
| 原版紧凑Transolver | 5 | 52.06% | 93.20% |

同5遍全局Hash为31.37%／54.45%，只比FNO联合改善0.94个百分点，停止小幅微调。Hash单样本<1%说明能拟合已知场，不能当新CAD精度。

TS系列原版优于++／3，但三者TRAIN应力也约94%，首先是拟合不足。相同R16表示的FNO应力64.98%，所以输入压缩不足以解释差距。剔除空token、标准化MSE、固定幅值先验及迁移TRAIN学得的编码器／输出头均未解决TS问题；这些负结果不追加40遍。

最新迁移对照已完成：TS应力94.37%，没有改善；R16 FNO从自身5→10遍应力64.98%→58.96%。三控制新训练共9.03分钟、DEV2.80分钟，继承teacher旧4.19分钟单列，不当fresh5公平比较。[迁移对照](geobc_geometry_head_v34/BRIEF.md)。

中间场诊断已完成：TS新CAD teacher-latent误差98.66%，FNO8.73%；但FNO学生原FEM应力仍65.90%，没有超越物理基线。停止该路线长训。下一检查官方structured3D TS的邻域卷积是否更适合规则R16输入；固定一个容量近FNO的候选，CPU反传一致已通过，真实物理5遍训练尚未启动。[诊断报告](geobc_node_latent_teacher_v37/BRIEF.md)。

5遍初筛，有证据再10遍；不统一40遍或24000步。旧FNO40遍整轮仅16.61分钟，knownDEV U/S23.17%／50.85%；全节点++40遍约4.3小时是整段外推总时间，不能再乘40。较长训练结果不与5遍当等曝光公平比较。

保留signed U(mm)/S(MPa)、GRID对齐CSV与H5原节点；旧H5-only撤回。knownDEV已用于迭代、单seed／可用官方TRAIN来源，不能宣布架构上限；旧FINAL已揭盲，未来不用调参。连续接口一致不等于独立离节点FEM精度。

[同5遍Hash对照](geobc_global_hash_physical_v18/BRIEF.md) · [FNO5→10](geobc_physical10_v16/BRIEF.md) · [三版TS](geobc_coarse_three_physical5_v21/BRIEF.md) · [既有40遍对照](geobc_operator_hash_v8/BRIEF.md)
