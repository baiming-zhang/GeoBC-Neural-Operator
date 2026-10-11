# GeoBC：原版DeepJEB求解器筛选

从整体几何、材料、载荷与约束预测位移和原signed应力，目标是共享模型迁移到新CAD，并支持连续空间查询。研究继续进行。

## 当前结果

**FNO路线目前优先；加条件Hash在同20遍比较中只略好，Transolver仍落后。尚未确定最终solver。**

| 同20遍／同20个开发CAD×三载荷 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO + 条件Hash | 20.37% | 48.34% | 33.14% |
| FNO | 20.65% | 48.83% | 36.84% |
| 原Transolver + 连续解码器 | 35.79% | 59.38% | 38.68% |

Hash只比FNO好0.27／0.49个百分点，未通过3个百分点DEV推进门；训练应力29.21%对35.74%的优势没有充分迁移。Transolver从5遍补到20遍有所改善，仍比FNO差15.15／10.55个百分点；同曝光仍有容量、精度和成本差异，不能据此宣布架构上限。

本轮训练28.02分钟（Hash追加4.33分钟、TS追加20.67分钟），验证2.15分钟。先提交全部120份全节点预测，再共享读取GT60一次；旧方法仅冻结标量复用。[一分钟简报](results/SOLVER_BRIEF.md) · [同20遍比较](results/geobc_matched20_readout_v79/BRIEF.md) · [训练曲线](results/geobc_matched20_solver_v78/BRIEF.md)。

已有FNO30／40开发集应力48.07%／49.48%，与本轮接近，停止自动增加遍数。真值oracle整体缩放仍留下约47.8%误差，空间／signed模式问题更值得探索。局部几何短对照负结果已关闭；三解码器九场短拟合的TCNN优点只作为拟合证据，不替代迁移。

原网格Transolver短比较已结束：同10遍训练应力55.23%，R64版54.77%；5遍时约10个百分点的优势没有保留，关闭这条表示分支，不自动扩大或用开发集小调。完整13.40分钟。[原网格比较](results/geobc_nativeTS100_matched10_v80/BRIEF.md)。

下一步设计为FNO与FNO加Hash的配对随机种子确认：另加2027／2028两个seed，各方案固定20遍，连同已有2026共三组；检验0.49个百分点的应力差是否可重复，并比较容量／实际成本。保留原配方，不换超参数、不给每个模型跑24000步。目前仅设计，未执行。[种子确认设计](notes/geobc_solver_seed_confirmation_v81_design.json)。

## 数据、指标与复现范围

固定100 TRAIN/20 knownDEV/20 FINAL CAD，每CAD三载荷，CAD隔离；scaler仅旧TRAIN40拟合。原版DeepJEB CSV按GRID坐标对齐H5 vertices，全部原节点、signed U(mm)/S(MPa)，不造符号。teacher/grid/FEM指标分开，旧H5-only结果撤回。knownDEV已用于迭代、单seed、可用来源条件选择，不能称独立最终盲测；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

原slides单/双已知场应力0.077%/0.088%、0.108%；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。30项可恢复实验已完成，但原目录、部分25特征schema、网格原点与50样本集合不可恢复，不能称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

整体FNO条件连续TCNN方案已经实现并通过节点/分块/重排/查询梯度检查；接口连续不等于独立离节点FEM精度。RBE2 CM为耦合成员，不假定grounded SPC。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

## 运行与私人交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存30.5GB SI，公共非阻塞GPU锁。用 `python -m tools.<模块名>` 按冻结计划与真实终态交接运行，不因旧status重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复冻结源文件字节，未知内容拒绝覆盖。短实验临时使用5分钟监视，后续长阶段恢复半小时监视。代码、配置、标量结果、图和检查点索引发布到[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)，原数据及检查点体积文件留本地。

最新已验证私人仓库提交`30ae940`，包含同20遍的实际训练／验证结果、代码、曲线和已检查的图，以及下一原网格Transolver短比较的设计（未执行）。发布70.83秒、48份变化文件，前中后private身份与确切远端commit/tree核实。[发布状态](notes/github_publish_plan.json)。
