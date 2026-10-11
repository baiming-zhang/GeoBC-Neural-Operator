# 一分钟结果

**单场应力拟合：FNO条件Hash明显更好。新CAD验证：当前仍优先FNO20，最终求解器未定。**

| 同3个困难场／fresh1000步独立拟合 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| 普通FNO解码器 | 5.12% | 18.27% | 20.85% |
| FNO条件Hash解码器 | 4.95% | 0.60% | 0.10% |

两个方案相同骨干初值、几何载荷输入与物理U/S损失；Hash约12.25M参数，FNO约4.53M，容量和精度不同。三个Hash场都低于1%应力，但位移仍约5%。整轮7.14分钟，CPU报告0.95秒；这是已知场拟合，不是新CAD预测。[两行诊断与云图](geobc_single_decoder_v60/BRIEF.md)。

| 现有新CAD knownDEV60 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO 20遍 | 20.65% | 48.83% | 36.84% |
| FNO 10遍 | 25.60% | 52.14% | 41.93% |
| 紧凑原版Transolver 10遍 | 40.33% | 91.50% | 91.33% |

FNO20训练集应力35.74%，共享拟合仍不足；训练累计6.17分钟、整个训练阶段14.32分钟，新CAD读出71.09秒。宽FNO、++、3、全局Hash、GINO与各消融没有更优的现有验证结果，但不同容量、表示和成本不能当架构上限。[当前验证](geobc_fno20_readout_v57/BRIEF.md) · [最佳原版TS对比](geobc_original_TS_v51/BRIEF.md)。

**接下来只推进能改变选型的问题：共享模型能否保留Hash的单场应力优势。** 同3个场、相同每场1000曝光，比较共享FNO与条件Hash；目前仅提案、未运行，不扫Hash小超参。[下一阶段](../notes/geobc_shared_decoder_v61_proposal.json)。

原版DeepJEB、CSV GRID对齐、全部原FEM节点、signed U(mm)/S(MPa)。100TRAIN／20knownDEV／20FINAL按CAD隔离，scaler仅TRAIN；knownDEV已迭代、单seed、可用来源条件选择，旧FINAL不再调参。旧H5-only撤回。Slides单/双场<1%，共享50场15.11%；FNO grid2.46%／FEM14.38%，不能混比。原配置不可完整恢复，不能声称精确数值复现。

历史诊断、失败与局限保留在[此前简报](SOLVER_BRIEF_PRE_V60_20261012.md)、[阶段历史](README_STATE_PRE_V60_20261012.md)及各实验目录；完整目标和研究持续推进。
