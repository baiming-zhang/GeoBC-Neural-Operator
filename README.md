# GeoBC：DeepJEB 场预测与求解器筛选

根据几何、材料、载荷与约束，预测原节点及连续空间的位移和 signed 应力。使用**原版 DeepJEB**；最终求解器尚未选定。

## 目前结论

**最新学习候选优先FNO：100个训练CAD×三载荷、12000更新，在knownDEV20 CAD×三载荷上位移／signed应力误差23.17%／50.85%。最终求解器尚未选定。**

同40CAD／12000更新、R64/native25、约4.53M参数的历史公平对照：FNO26.50%／55.55%，Transolver++52.35%／83.99%。目前TS系列以++为主，FNO优先；不宣布架构上限。[同配置比较](results/geobc_short_screen/BRIEF.md)。

Hash低于1%来自单／双已知病例的应力拟合，不能直接比较新CAD的联合位移／应力。几何条件Hash在同40CAD／4800更新的新60病例上49.19%／64.96%，FNO27.93%／56.06%；总训练阶段3.43／4.93分钟。[40遍确认](results/geobc_two_warm_screens_v3/BRIEF.md)。

## 最新进展

FNO扩大TRAIN40→100：同4800更新没有改善（28.59%／56.49%）；给予相同40遍曝光、12000更新后改善至23.17%／50.85%，位移19/20、应力20/20 CAD更低。整轮训练996.53秒（16.61分钟），开发读出98.23秒，CPU导出4.38秒，图已验看；120份完整有限预测先验证再共享knownGT60一次。未读FINAL／原CSV、新拟合scaler0。同更新数与同曝光数分别解释，不能声称纯数据单因素。[三行覆盖报告](results/geobc_fno_coverage_v4/BRIEF.md)。

同100CAD／4800更新宏观比较已完成：FNO28.59%／56.49%，几何条件Hash41.27%／61.61%，全局算子Hash29.86%／54.01%。全局Hash20/20开发CAD应力与热点更低，但位移较差、参数及混合精度不同；未全面胜出。全局latent置零导致83.73%／100.00%，仅为推理依赖诊断。240完整有限预测先提交／重验再knownGT60共享一次，开发整轮240.66秒，CPU导出18.22秒，实际图已验看。[三行比较](results/geobc_operator_hash_v5/BRIEF.md)。

真实TCNN全节点梯度／FNO初始化字节检查通过；训练后真实CUDA连续查询整轮4.83秒通过节点／chunk／perm一致性和输入梯度，GT／更新0。原全局阶段元数据getter技术失败126秒与修复间隔469秒均保留，原20分钟绝对budget未重置；恢复训练287.95秒、原完整campaign1011.26秒。只将有竞争力的全局Hash从自身4800恢复model／Adam／四RNG／cursor，追加7200到与FNO相同12000／40遍：训练最多10分钟、整体20分钟、30.5GB。不延长几何Hash、不调微小参数；新端点尚未完成，不声称最终选型。[CUDA接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

此前原生++10遍／1200更新阶段26.32分钟，开发69.68%／94.05%，当前配方贵且欠拟合，暂降低投入优先级。原双24000计划已主动终止，真实非零退出、日志和成本保留；技术交接空argv快照失败已用实际完整进程身份核对，未重跑预测／GT、不冒称完成。[三候选初筛](results/geobc_three_short_screens/BRIEF.md)。

## 数据与资源

前瞻协议固定 **100 TRAIN／20 known DEV／20 reserved FINAL CAD**：TRAIN为旧40加新60，每CAD三载荷；scaler沿用仅旧TRAIN40拟合的版本。FINAL数值标签未读，不用于调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

420病例接口已整合：TRAIN300有已有目标，DEV60／FINAL60仅输入。两个实际TRAIN加载检查通过；旧int32／新int64索引兼容故障及原预算内修复保留，未验证全部300数值正文，本次100CAD缓存18.39GB、最大全节点梯度已通过，设备使用峰值21.83GB。[接口简报](results/geobc_prospective_dataset_interface/BRIEF.md)。后续入口为 `geobc.prospective_dataset_v2.load_case`，明确选择40或100范围。

用户显存预算 **30.5 GB**（30,500,000,000 bytes）。100CAD R64全缓存加当前++工作空间估计28.64GB，后续优先GPU常驻；必须以实际全节点allocated／reserved各自峰值确认。[资源预算](notes/geobc_user_resource_budget.json) · [缓存估算](notes/geobc_prospective_cache_memory_plan.json)。历史实验保留其原24GiB上限；当前v4采用用户30.5GB预算。 共用TRAIN缓存接口已通过元数据与合成CPU检查（1.074秒），100CAD原生方法仅需约7.02GiB缓存；真实GPU载入仍未执行。[缓存接口](results/geobc_train_cache_interface/BRIEF.md)。

## 复现与证据

30项可恢复slides实验已完成，涵盖MLP、NGP、Hash、Transolver／++／3、残差、FNO和蒸馏；单、双、五样本拟合与共享50样本分开报告。部分历史概述缺失配置，明确列为不可恢复。原实验目录及部分25特征schema、网格原点、50样本集合缺失，不能称精确数值重放。

原CSV按GRID坐标对齐H5 vertices，保留signed U(mm)和S(MPa)；撤回的H5-only结果不引用。全部原节点评估，病例均值、最差及最高绝对应力1%热点分开；teacher、grid往返和FEM误差分开。连续查询一致性检查不等于独立离节点FEM精度。

[一分钟结果](results/SOLVER_BRIEF.md) · [Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md) · [Transolver比较](results/TRANSOLVER_COMPARISON.md) · [研究历史](results/RESEARCH_HISTORY.md) · [阶段详细记录](results/geobc_exploration/REPORT.md)

原40／200CAD下载协议已在固定预算内input-negative，缺223／540源；不替换ID或重复失败下载。旧++作者制度停于23166／24000资源失败，最终物理端点缺失，不能冒充完成。[输入负结果](results/geobc_coverage_input_negative/BRIEF.md) · [制度负结果](results/geobc_pp_regime_confirmation/BRIEF.md)。

## 运行与私人仓库

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。以下在WSL项目目录运行：

```bash
python tools/run_slides_suite.py --dry-run
python tools/summarize_suite.py
python tools/plot_suite_fields.py
```

研究阶段必须使用各自冻结计划和真实进程交接，不能因旧status、锁文件或SOURCE-ready重复启动GPU任务。30分钟监视器持续推进；活训练期间只作轻量检查。

[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；最近已验收提交 `5c89902` 包含FNO100CAD实际完成结果。全局Hash宏观对照、必要依赖诊断、CUDA连续接口和续训启动状态正在整理为下一快照；仅在远端精确commit、owner/repo ID及private状态核验后声明发布。[发布计划](notes/github_publish_plan.json)。

更早逐阶段运行叙述存于[README历史快照](results/README_STATE_HISTORY_20261011.md)，状态以当前真实报告和运行收据为准。
