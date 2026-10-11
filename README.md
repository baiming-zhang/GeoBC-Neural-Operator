# GeoBC：原版DeepJEB求解器筛选

从整体几何、材料、载荷与约束预测位移和signed应力，目标是共享模型迁移到新CAD，并支持连续空间查询。研究仍在进行。

## 当前最重要的结论

**新CAD综合误差目前仍以FNO20为优先；条件Hash在同10遍曝光下略好于FNO，热点优势更明显，最终solver未定。**

| 同20个新CAD×三载荷／全部原节点 | 位移误差 | 应力误差 | 热点误差 | 累计纯训练分钟 |
|---|---:|---:|---:|---:|
| 条件Hash 10遍 | 25.26% | 50.85% | 35.07% | 3.29 |
| FNO 10遍 | 25.60% | 52.14% | 41.93% | 3.23 |
| FNO 20遍 | 20.65% | 48.83% | 36.84% | 6.17 |

Hash追加5遍后，训练集应力47.28%→39.32%，新CAD54.45%→50.85%。新增训练92.37秒、完整训练阶段248.03秒，验证72.56秒；60份预测全部提交、重验后才读GT各一次。纯训练成本不包含加载、失败和分析等待，完整成本另外保留。[本轮三行简报](results/geobc_global_hash10_readout_v63/BRIEF.md) · [训练趋势](results/geobc_global_hash_learning10_v62/BRIEF.md)。

**Hash的小样本拟合优势可以观察到，但扩大到100CAD后明显缩小。** 同3个困难场共享一个模型、每场1000曝光，Hash应力0.76%／FNO19.62%；Hash位移仍4.85%。原slides单／双场<1%属于已知场拟合，不能直接与新CAD比较。当前Hash10和FNO20的训练集应力仍39.32%／35.74%，共享拟合仍不足，不只泛化问题。[小样本共享对照](results/geobc_shared_decoder_v61/BRIEF.md)。

相同10遍新CAD上，Hash比FNO位移／应力／热点低0.34／1.30／6.85个百分点；FNO20整体位移和应力更低，Hash10热点低1.77个百分点。容量和精度不同、20遍曝光不同，不能宣布架构上限。此前紧凑原版Transolver10新CAD位移40.33%／应力91.50%；该输出表示与配置尚不能代表Transolver所有方案。[原TS结果](results/geobc_original_TS_v51/BRIEF.md)。

**原版Transolver＋相同连续解码器已完成共享三场对照。** 每场1000次曝光，TS BF16位移9.33%／应力1.54%／热点2.46%，FNO条件Hash为4.85%／0.76%／0.43%；TS学习区间792.3秒，约为FNO188.6秒的4.20倍。原FP32因900秒学习预算结束于2285次更新，完整909.41秒的失败成本和检查点保留，没有最终1000端点。[TS真实对照](results/geobc_transolver_bf16_v65/BRIEF.md) · [FP32成本](results/geobc_transolver_decoder_v64_budget/BRIEF.md)。

**原版TS同方案的fresh100CAD×三载荷短五遍及knownDEV60已完成。** 全TRAIN位移62.85%／应力71.12%，knownDEV位移61.14%／应力72.72%；完整训练613.46秒、验证82.59秒，推进门槛未通过。当前FNO20仍领先，不自动加长TS训练。旧去空体素TS与Native++负结果保留，不预设native表示能解决误差。[最新TS比较](results/geobc_transolver_macro100_readout_v67/BRIEF.md)。

**原网格TS三场对照也已完成：应力1.52%／位移9.59%，与R64 TS的1.54%／9.33%基本相同，学习时间只省约6%。** 该路线暂停扩大100CAD，能力门通过不代表实际优势。原启动的元数据错误发生在0更新；恢复保留原绝对截止与失败成本，完整结果及两张图已验收。[原网格比较](results/geobc_transolver_native_shared3_v70/BRIEF.md)。当前实际进行同一FNO20检查点的固定30／40遍趋势诊断，预计新增约6分钟学习区间，检查更多曝光是否同时改善TRAIN和新CAD；不自动80遍或其他方法长训，目前没有新端点结果。

## 数据、指标与复现范围

固定100 TRAIN/20 knownDEV/20 FINAL CAD，每CAD三载荷，CAD隔离；scaler仅旧TRAIN40拟合。原版DeepJEB CSV按GRID坐标对齐H5 vertices，全部原节点、signed U(mm)/S(MPa)，不造符号。teacher/grid/FEM指标分开，旧H5-only结果撤回。knownDEV已用于迭代、单seed、可用来源条件选择，不能称独立最终盲测；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

原slides单/双已知场应力0.077%/0.088%、0.108%；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。30项可恢复实验已完成，但原目录、部分25特征schema、网格原点与50样本集合不可恢复，不能称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

整体FNO条件连续TCNN方案已经实现并通过节点/分块/重排/查询梯度检查；接口连续不等于独立离节点FEM精度。RBE2 CM为耦合成员，不假定grounded SPC。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

## 运行与私人交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存30.5GB SI，公共非阻塞GPU锁。用 `python -m tools.<模块名>` 按冻结计划与真实终态交接运行，不因旧status重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复冻结源文件字节，未知内容拒绝覆盖。短实验临时使用5分钟监视，后续长阶段恢复半小时监视。代码、配置、标量结果、图和检查点索引发布到[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)，原数据及检查点体积文件留本地。

最新已验证远端提交`bb16d95`，包含已完成的TS100CAD短五遍与knownDEV60物理报告及图、原网格资格证据和技术失败记录，以及保持原截止时间的三场拟合恢复源码与启动状态（尚无新拟合结果）。发布82.65秒、63份变化文件，前中后private身份及确切远端commit/tree均核实；研究继续推进。[发布状态](notes/github_publish_plan.json)。
