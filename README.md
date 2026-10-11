# GeoBC：原版DeepJEB求解器筛选

从整体几何、材料、载荷与约束预测位移和signed应力，目标是共享模型迁移到新CAD，并支持连续空间查询。研究仍在进行。

## 当前最重要的结论

**新CAD综合误差目前仍以FNO20为优先；条件Hash在同10遍曝光下略好于FNO，热点优势更明显，最终solver未定。**

| 同20个新CAD×三载荷／全部原节点 | 位移误差 | 应力误差 | 热点误差 | 累计純训练分钟 |
|---|---:|---:|---:|---:|
| 条件Hash 10遍 | 25.26% | 50.85% | 35.07% | 3.29 |
| FNO 10遍 | 25.60% | 52.14% | 41.93% | 3.23 |
| FNO 20遍 | 20.65% | 48.83% | 36.84% | 6.17 |

Hash追加5遍后，训练集应力47.28%→39.32%，新CAD54.45%→50.85%。新增训练92.37秒、完整训练阶段248.03秒，验证72.56秒；60份预测全部提交、重验后才读GT各一次。纯训练成本不包含加载、失败和分析等待，完整成本另外保留。[本轮三行简报](results/geobc_global_hash10_readout_v63/BRIEF.md) · [训练趋势](results/geobc_global_hash_learning10_v62/BRIEF.md)。

**Hash的小样本拟合优势可以观察到，但扩大到100CAD后明显缩小。** 同3个困难场共享一个模型、每场1000曝光，Hash应力0.76%／FNO19.62%；Hash位移仍4.85%。原slides单／双场<1%属于已知场拟合，不能直接与新CAD比较。当前Hash10和FNO20的训练集应力仍39.32%／35.74%，共享拟合仍不足，不只泛化问题。[小样本共享对照](results/geobc_shared_decoder_v61/BRIEF.md)。

相同10遍新CAD上，Hash比FNO位移／应力／热点低0.34／1.30／6.85个百分点；FNO20整体位移和应力更低，Hash10热点低1.77个百分点。容量和精度不同、20遍曝光不同，不能宣布架构上限。此前紧凑原版Transolver10新CAD位移40.33%／应力91.50%；该输出表示与配置尚不能代表Transolver所有方案。[原TS结果](results/geobc_original_TS_v51/BRIEF.md)。

**下一步优先Transolver骨干＋同连续解码器对照，暂不继续延长Hash训练。** 用相同整体几何／边界条件输入和TCNN解码器，先验证固定共享3场的拟合与真实成本，再决定是否推进100CAD比较。当前仅[下一阶段提案](notes/geobc_transolver_continuous_decoder_v64_proposal.json)，未实施；不自动40遍／24000步或扫描Hash小超参。研究持续推进，历史负结果和成本保留。

## 数据、指标与复现范围

固定100 TRAIN/20 knownDEV/20 FINAL CAD，每CAD三载荷，CAD隔离；scaler仅旧TRAIN40拟合。原版DeepJEB CSV按GRID坐标对齐H5 vertices，全部原节点、signed U(mm)/S(MPa)，不造符号。teacher/grid/FEM指标分开，旧H5-only结果撤回。knownDEV已用于迭代、单seed、可用来源条件选择，不能称独立最终盲测；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

原slides单/双已知场应力0.077%/0.088%、0.108%；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。30项可恢复实验已完成，但原目录、部分25特征schema、网格原点与50样本集合不可恢复，不能称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

整体FNO条件连续TCNN方案已经实现并通过节点/分块/重排/查询梯度检查；接口连续不等于独立离节点FEM精度。RBE2 CM为耦合成员，不假定grounded SPC。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

## 运行与私人交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存30.5GB SI，公共非阻塞GPU锁。用 `python -m tools.<模块名>` 按冻结计划与真实终态交接运行，不因旧status重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复338个冻结源文件字节，未知内容拒绝覆盖。半小时监视器继续推进后续任务。代码、配置、标量结果、图和检查点索引发布到[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)，原数据及检查点体积文件留本地。

最新已验证远端提交`0bae695`，包含本轮v61共享拟合、两图、源码与真实终态，以及v62短验证提案和父检查点元数据证据。发布48.80秒，33份变化文件，前中后private身份和确切远端commit/tree均核实；研究继续推进。[发布状态](notes/github_publish_plan.json)。
