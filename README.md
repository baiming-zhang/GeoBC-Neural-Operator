# GeoBC：原版DeepJEB求解器筛选

从整体几何、材料、载荷与约束预测位移和signed应力，目标是共享模型迁移到新CAD，并支持连续空间查询。研究仍在进行。

## 当前最重要的结论

**共享一个模型学习3个困难场，条件Hash的平均应力误差仍为0.76%，三个场都低于1%；普通FNO为19.62%。** 前一轮独立拟合分别0.60%/18.27%。少量场共享没有消除Hash解码器包的应力拟合优势；当前高误差不能只归因于共享或新CAD泛化。

| 同3场／每场1000曝光／一个共享模型 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| 普通FNO解码器 | 6.83% | 19.62% | 23.16% |
| FNO条件Hash解码器 | 4.85% | 0.76% | 0.43% |

整轮6.39分钟，CPU报告0.98秒；两方案相同自身fresh初值、几何载荷输入、scaler与物理U/S联合损失，每模型Adam3000步。Hash从独立到共享应力增加0.15个百分点。FNO约4.53M参数/FP32，Hash约12.25M参数/混合TCNN精度，容量和精度不同；位移仍约5%。不是未见几何排名。[共享两行简报](results/geobc_shared_decoder_v61/BRIEF.md) · [共享应力云图](results/geobc_shared_decoder_v61/representative_stress.png) · [此前独立拟合](results/geobc_single_decoder_v60/BRIEF.md)。

**新CAD验证目前仍以FNO20遍为优先候选，最终solver未定。** 单场模型没有未见几何预测资格。

| 现有knownDEV60结果 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO 20遍 | 20.65% | 48.83% | 36.84% |
| FNO 10遍 | 25.60% | 52.14% | 41.93% |
| 紧凑原版Transolver 10遍 | 40.33% | 91.50% | 91.33% |

FNO20训练集应力仍35.74%；共享拟合也不足。容量、表示和训练成本不同，此表是实际候选状态，不是架构上限或等预算排名。++/3、全局Hash、GINO及消融完整结果见[一分钟结果](results/SOLVER_BRIEF.md)与[阶段历史](results/README_STATE_PRE_V60_20261012.md)。

下一步回到100CAD×三载荷，复用现有Hash5遍的model/Adam/RNG，追加5遍／1500步短验证，对比同10遍FNO。当前仅[新提案](notes/geobc_global_hash_learning10_v62_proposal.json)，未运行；此前Hash5遍的负推进门槛和成本保留，新3场证据支持单独重新评估。TRAIN应力至少改善3个百分点且位移不恶化超过2个百分点，才另行准备knownDEV60读出，不自动长训练。后续可根据结果设计同连续解码器的Transolver骨干对照。

## 数据、指标与复现范围

固定100 TRAIN/20 knownDEV/20 FINAL CAD，每CAD三载荷，CAD隔离；scaler仅旧TRAIN40拟合。原版DeepJEB CSV按GRID坐标对齐H5 vertices，全部原节点、signed U(mm)/S(MPa)，不造符号。teacher/grid/FEM指标分开，旧H5-only结果撤回。knownDEV已用于迭代、单seed、可用来源条件选择，不能称独立最终盲测；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

原slides单/双已知场应力0.077%/0.088%、0.108%；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。30项可恢复实验已完成，但原目录、部分25特征schema、网格原点与50样本集合不可恢复，不能称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

整体FNO条件连续TCNN方案已经实现并通过节点/分块/重排/查询梯度检查；接口连续不等于独立离节点FEM精度。RBE2 CM为耦合成员，不假定grounded SPC。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

## 运行与私人交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存30.5GB SI，公共非阻塞GPU锁。用 `python -m tools.<模块名>` 按冻结计划与真实终态交接运行，不因旧status重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复327个冻结源文件字节，未知内容拒绝覆盖。半小时监视器继续推进后续任务。代码、配置、标量结果、图和检查点索引发布到[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)，原数据及检查点体积文件留本地。

最新已验证远端提交`1d4fe57`，包含本轮v60实际同场拟合、两图、源码与终态证据，以及v61共享对照提案。发布47.52秒，32份变化文件，前中后private身份及确切远端commit/tree均核实；研究持续推进。[发布状态](notes/github_publish_plan.json)。
