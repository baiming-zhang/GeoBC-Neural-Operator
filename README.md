# GeoBC：原版DeepJEB求解器筛选

从整体几何、材料、载荷与约束预测位移和原signed应力，目标是共享模型迁移到新CAD，并支持连续空间查询。研究继续进行。

## 当前结果

**三种子确认后，默认保留普通FNO。Hash整体优势很小，但热点预测稳定更好，保留作对照；最终求解器仍未确定。**

| 同20遍／三seed／20个开发CAD×三载荷 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO | 21.72% | 49.33% | 37.53% |
| FNO + 条件Hash | 21.97% | 48.71% | 33.24% |

Hash应力只改善0.62个百分点，位移差0.25个百分点，联合误差仅好0.19个百分点，未达到预声明3个百分点推进门；热点改善4.30个百分点在三seed中均保留。参数为1225万对453万，新增seed每模型学习6.39对5.90分钟；60场纯推理0.82对0.96秒，整体验证含IO为3.92分钟。不能只因Hash参数大就说它推理更慢。

当前最好的Transolver方案同20遍／单seed位移35.79%、应力59.38%，仍落后；不同seed数量、容量与精度不能当架构上限。原网格TS短对照和继续加长FNO均未带来实际改善，已关闭，不重跑。

本轮四个模型各20遍已完成：恢复执行27.01分钟，原任务含零更新字段错误与定位等待37.22分钟。验证240份预测先全部保存重验，再共享读GT60一次；原2026结果只复用标量，不重推理。下一步定位应力误差中符号与空间幅值的贡献，再决定结构性改动，不在Hash上做小调。

[三seed完整误差范围与成本](results/geobc_solver_seed_readout_v83/BRIEF.md) · [训练拟合与曲线](results/geobc_solver_seeds_training_v82/BRIEF.md)。原版DeepJEB、GRID对齐CSV、signed U(mm)/S(MPa)、全原节点、CAD隔离、scaler仅旧TRAIN40。三seed共用同20CAD，不是180独立病例；knownDEV已迭代、来源受可用性影响，不是最终盲测。


## 数据、指标与复现范围

固定100 TRAIN/20 knownDEV/20 FINAL CAD，每CAD三载荷，CAD隔离；scaler仅旧TRAIN40拟合。原版DeepJEB CSV按GRID坐标对齐H5 vertices，全部原节点、signed U(mm)/S(MPa)，不造符号。teacher/grid/FEM指标分开，旧H5-only结果撤回。knownDEV已用于迭代、单seed、可用来源条件选择，不能称独立最终盲测；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

原slides单/双已知场应力0.077%/0.088%、0.108%；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。30项可恢复实验已完成，但原目录、部分25特征schema、网格原点与50样本集合不可恢复，不能称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

整体FNO条件连续TCNN方案已经实现并通过节点/分块/重排/查询梯度检查；接口连续不等于独立离节点FEM精度。RBE2 CM为耦合成员，不假定grounded SPC。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

## 运行与私人交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存30.5GB SI，公共非阻塞GPU锁。用 `python -m tools.<模块名>` 按冻结计划与真实终态交接运行，不因旧status重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复冻结源文件字节，未知内容拒绝覆盖。短实验临时使用5分钟监视，后续长阶段恢复半小时监视。代码、配置、标量结果、图和检查点索引发布到[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)，原数据及检查点体积文件留本地。

最新已验证私人仓库提交`0277577`，包含原网格TS固定10遍的实际负结果、代码、曲线和已检查的图，以及下一配对种子确认设计（未执行）。发布52.60秒、33份变化文件，前中后private身份与确切远端commit/tree核实。[发布状态](notes/github_publish_plan.json)。


