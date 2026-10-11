# GeoBC：DeepJEB 场预测与求解器筛选

根据几何、材料、载荷与约束，预测原节点及连续空间的位移和 signed 应力。使用**原版 DeepJEB**；最终求解器尚未选定。

## 目前结论

**当前配方 FNO 领先，Transolver 系列优先保留 ++；Hash 仍是连续场候选。** 最新共同对照使用相同 R64 输入、native25 查询头、约4.53M参数、TRAIN40 CAD×三载荷和4800次更新。

| 方法 | TRAIN 位移／应力误差 | 新20 CAD×三载荷位移／应力误差 | 训练 UTC 分钟 |
|---|---:|---:|---:|
| FNO | 19.5%／39.0% | 27.9%／56.1% | 26.97 |
| Transolver++ | 40.6%／85.9% | 41.4%／86.4% | 144.36 |

以上为全原节点、物理单位 relative-L2 的病例均值；新20 CAD是已知开发池，不能当作最终测试。这说明当前配方的差距，不能宣布某个架构的上限。见[共同配置比较](results/geobc_common_grid_comparison/BRIEF.md)。

按CAD配对分析，当前FNO联合评分领先20/20 CAD，平均差21.9个百分点；逐一移除一个CAD后仍领先21.0–22.5个百分点。这是已知开发池的单seed结果。[CAD配对证据](results/geobc_common_grid_CAD_pair_analysis/BRIEF.md)。

Hash 的约0.4%来自单病例已知应力拟合，五病例约1.45%；它与跨CAD、联合位移／应力任务不同。FNO 在独立单病例4800更新可达到位移3.9%／应力11.5%，表明共享训练误差包含拟合不足。[单例诊断](results/geobc_single_case_fit/BRIEF.md) · [Hash口径](notes/geobc_hash_error_comparability.json)。

## 正在推进

初筛预算已缩短：后续候选先跑 **10遍／1200更新，每方法训练上限30分钟**，在同CAD、同标签和同指标下比较精度及成本；仅有竞争力的候选进入长训练。40遍指4800次更新，并非重复40次四小时实验。

- FNO 已完成24000：TRAIN120位移／应力为 **10.5%／24.2%**，新增训练UTC10.23分钟。新的开发读出尚未执行。
- ++12000的TRAIN位移／应力为 **52.7%／81.9%**；同曝光FNO为17.4%／30.7%。++应力略降，但位移较自身4800的40.6%恶化，延长训练尚未追上。继续原固定24000端点。[中间物理比较](results/geobc_exposure_comparison/12000_INTERIM.md) · [真实运行状态](notes/geobc_exposure_live_milestone.json)。
- ++位移恶化集中在小／中位移病例（74/120病例、24/40 CAD）；节点能量合并误差反而44.4%→40.7%，实际平均RMS近乎不变。正式病例等权主指标不变，目标权重的因果作用未验证。[病例分布诊断](results/geobc_exposure_comparison/MIDPOINT_DIAGNOSTIC.md)。
- ++延长训练已主动终止，真实非零退出、闭合时钟、检查点及全部日志保留；原双24k实验不宣称完成。同12000检查点的完整开发60例读出已完成（97.15秒）：FNO **26.5%／55.5%**，++ **52.4%／84.0%**；FNO联合评分在20/20 CAD更低。[同曝光结果](results/geobc_short_screen/BRIEF.md)。独立v2入口已启动FNO／原生Transolver++／条件化连续Hash各10遍短筛（先FNO，后++、Hash），使用30.5GB上限。[计划](notes/geobc_exposure_extension_plan.json) · [宏观问题](notes/geobc_macro_conditioning_source_review.json)。

当前模型通过网格和查询特征接收边界条件；全局PointNet只编码几何。历史粗粒度约束／加载区点云消融未带来明确收益，下一轮避免重复该配方。 原生节点++接口也已实现，全部初始参数与R64对照一致，CPU合成接口检查1.106秒；真实CUDA和精度未验证。[原生接口准备](results/geobc_native_context_interface/BRIEF.md)。

后续宏观方法已共用全节点训练与开发集评估入口；开发标签仅在所有候选预测完整提交后读取。CPU合成协议检查通过，真实CUDA和新精度仍未执行。[评估接口](results/geobc_macro_readout_interface/BRIEF.md)。

## 数据与资源

前瞻协议固定 **100 TRAIN／20 known DEV／20 reserved FINAL CAD**：TRAIN为旧40加新60，每CAD三载荷；scaler沿用仅旧TRAIN40拟合的版本。FINAL数值标签未读，不用于调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

420病例接口已整合：TRAIN300有已有目标，DEV60／FINAL60仅输入。两个实际TRAIN加载检查通过；旧int32／新int64索引兼容故障及原预算内修复保留，未验证全部300数值正文，也未启动新100CAD训练。[接口简报](results/geobc_prospective_dataset_interface/BRIEF.md)。后续入口为 `geobc.prospective_dataset_v2.load_case`，明确选择40或100范围。

用户显存预算 **30.5 GB**（30,500,000,000 bytes）。100CAD R64全缓存加当前++工作空间估计28.64GB，后续优先GPU常驻；必须以实际全节点allocated／reserved各自峰值确认。[资源预算](notes/geobc_user_resource_budget.json) · [缓存估算](notes/geobc_prospective_cache_memory_plan.json)。当前运行实验保留自身较小的原24GiB上限。 共用TRAIN缓存接口已通过元数据与合成CPU检查（1.074秒），100CAD原生方法仅需约7.02GiB缓存；真实GPU载入仍未执行。[缓存接口](results/geobc_train_cache_interface/BRIEF.md)。

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

[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；最近已验收提交 `2096c25` 包含同100次曝光的TRAIN120中间物理比较及++位移恶化的负结果；原双24000阶段现已主动终止；同12000开发读出已完成，10遍宏观短筛已启动；最新阶段结果正在整理发布。宏观接口与CAD配对证据在先前`1f8c127`保留。发布29.31秒，精确commit和private状态通过远端核验。[发布计划](notes/github_publish_plan.json)。

更早逐阶段运行叙述存于[README历史快照](results/README_STATE_HISTORY_20261011.md)，状态以当前真实报告和运行收据为准。
