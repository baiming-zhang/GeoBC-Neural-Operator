# GeoBC：原版DeepJEB求解器筛选

从整体几何、材料、载荷与约束预测位移和signed应力，目标是共享模型迁移到新CAD，并支持连续空间查询。研究仍在进行。

## 当前最重要的结论

**FNO仍是当前领先的基线，但延长20→40遍没有明显改善新CAD应力，停止自动加长训练。** 当前最低联合误差为FNO40，最低应力为FNO30；两者联合分只差0.06个百分点，不作稳健赢家或最终solver结论。

| 同20个新CAD×三载荷／全部原节点 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO 20遍 | 20.65% | 48.83% | 36.84% |
| FNO 30遍 | 20.63% | 48.07% | 36.83% |
| FNO 40遍 | 19.10% | 49.48% | 34.86% |

训练应力35.74%→30.18%，但40遍新CAD应力反而增加0.65个百分点，20个CAD中15个应力变差。新增训练区间5.89分钟、完整训练阶段8.39分钟，120份预测先提交再共享GT60一次的配对验证2.08分钟。两个DEV推进门均未通过；不自动80遍。40遍累计训练区间12.06分钟，包含中间读出与保存，30遍前缀时间未单独记录。[最新配对简报](results/geobc_fno40_pair_readout_v73/BRIEF.md) · [训练趋势](results/geobc_fno_learning40_v72/BRIEF.md)。

**当前Transolver方案未胜过FNO，仍不能宣布整个架构的上限。** 紧凑原TS10的联合分在已有TS候选中最低（位移40.33%／应力91.50%）；应力最低的TS是同连续解码器短五遍版（61.14%／72.72%）。该短五遍训练完整613.46秒、验证82.59秒；原网格TS三场应力1.52%与R64 TS1.54%基本相同，停止扩大这条路线。[TS比较](results/geobc_transolver_macro100_readout_v67/BRIEF.md) · [原网格诊断](results/geobc_transolver_native_shared3_v70/BRIEF.md)。

**小场拟合和新几何预测的误差不能混比。** 共享三个已知困难场，条件Hash应力0.76%、TS1.54%；扩大到100CAD时Hash10新CAD应力50.85%。slides单／双场约0.1%，但共享50场平均15.11%、没有一例低于1%。当前训练误差也高，缺口同时包含共享拟合和迁移，不能全部归因泛化。[共享拟合对照](results/geobc_shared_decoder_v61/BRIEF.md)。

幅值诊断已完成（16.81秒）：即使用真值给每场最优缩放，应力仍约47.8%。下一步匹配比较局部query几何token与额外全局池化token，先固定3CAD×3载荷共享拟合，不直接扩大100CAD。[诊断](results/geobc_field_amplitude_shape_v75/BRIEF.md)。不继续Hash小超参或延长训练救某一种方法。完整旧结果、失败和成本留在各阶段JSON／CSV，不在首页堆叠。

## 数据、指标与复现范围

固定100 TRAIN/20 knownDEV/20 FINAL CAD，每CAD三载荷，CAD隔离；scaler仅旧TRAIN40拟合。原版DeepJEB CSV按GRID坐标对齐H5 vertices，全部原节点、signed U(mm)/S(MPa)，不造符号。teacher/grid/FEM指标分开，旧H5-only结果撤回。knownDEV已用于迭代、单seed、可用来源条件选择，不能称独立最终盲测；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

原slides单/双已知场应力0.077%/0.088%、0.108%；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。30项可恢复实验已完成，但原目录、部分25特征schema、网格原点与50样本集合不可恢复，不能称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

整体FNO条件连续TCNN方案已经实现并通过节点/分块/重排/查询梯度检查；接口连续不等于独立离节点FEM精度。RBE2 CM为耦合成员，不假定grounded SPC。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

## 运行与私人交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存30.5GB SI，公共非阻塞GPU锁。用 `python -m tools.<模块名>` 按冻结计划与真实终态交接运行，不因旧status重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复冻结源文件字节，未知内容拒绝覆盖。短实验临时使用5分钟监视，后续长阶段恢复半小时监视。代码、配置、标量结果、图和检查点索引发布到[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)，原数据及检查点体积文件留本地。

最新已验证远端提交`1469813`，包含FNO30／40遍结果及幅值／空间分布诊断的实际结果、图与下一局部几何查询对照的设计（尚未实施该对照）。发布41.17秒、25份变化文件，前中后private身份及确切远端commit/tree均核实；研究继续推进。[发布状态](notes/github_publish_plan.json)。

局部几何短对照已完成：同3CAD×三载荷、相同参数和初值，局部聚合只降低应力0.98个百分点，却使位移增加2.53个百分点，未通过推进门；完整2分13秒。停止扩大该方案。下一步比较普通、固定Fourier坐标及既有TCNN三种解码器包，先做同九场短拟合，再决定是否值得新CAD验证。 [短对照](results/geobc_local_query_geometry_v76/BRIEF.md)。
