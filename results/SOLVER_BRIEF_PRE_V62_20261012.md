# 一分钟结果

**少量场共享后，条件Hash的应力拟合优势仍成立；新CAD验证目前仍优先FNO20，最终solver未定。**

| 同3个困难场／共享一个模型／每场1000曝光 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| 普通FNO解码器 | 6.83% | 19.62% | 23.16% |
| FNO条件Hash解码器 | 4.85% | 0.76% | 0.43% |

Hash三个场都低于1%；此前独立应力0.60%，共享仅增加0.15个百分点。整轮6.39分钟、CPU报告0.98秒。初值、输入和物理损失相同，但Hash约12.25M参数／混合精度、FNO约4.53M／FP32，不能单独归因Hash编码。已知3场拟合不等于新CAD迁移，位移仍约5%。[共享结果与云图](geobc_shared_decoder_v61/BRIEF.md) · [独立对照](geobc_single_decoder_v60/BRIEF.md)。

| 现有新CAD knownDEV60 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO 20遍 | 20.65% | 48.83% | 36.84% |
| FNO 10遍 | 25.60% | 52.14% | 41.93% |
| 紧凑原版Transolver 10遍 | 40.33% | 91.50% | 91.33% |

FNO20训练集应力35.74%，共享拟合仍不足；训练累计6.17分钟、整个训练阶段14.32分钟，新CAD读出71.09秒。宽FNO、++、3、全局Hash、GINO与各消融没有更优的现有验证结果，但不同容量、表示和成本不能当架构上限。[当前验证](geobc_fno20_readout_v57/BRIEF.md) · [最佳原版TS对比](geobc_original_TS_v51/BRIEF.md)。

**下一步回到100CAD规模。** 复用现有Hash5遍检查点，追加5遍／1500步，对比同10遍FNO；TRAIN改善达到新门槛后再验证knownDEV60。目前仅提案、未运行，旧5遍负结果保留。随后可考虑同连续解码器的Transolver骨干对照。[下一阶段](../notes/geobc_global_hash_learning10_v62_proposal.json)。

原版DeepJEB、CSV GRID对齐、全部原FEM节点、signed U(mm)/S(MPa)。100TRAIN／20knownDEV／20FINAL按CAD隔离，scaler仅TRAIN；knownDEV已迭代、单seed、可用来源条件选择，旧FINAL不再调参。旧H5-only撤回。Slides单/双场<1%，共享50场15.11%；FNO grid2.46%／FEM14.38%，不能混比。原配置不可完整恢复，不能声称精确数值复现。

历史诊断、失败与局限保留在[此前简报](SOLVER_BRIEF_PRE_V60_20261012.md)、[阶段历史](README_STATE_PRE_V60_20261012.md)及各实验目录；完整目标和研究持续推进。
