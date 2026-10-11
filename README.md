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

## 最新短筛与下一步

初筛使用同TRAIN40 CAD×三载荷、seed2026、10遍／1200更新、每方法训练上限30分钟、30.5GB。完整开发60例：FNO U/S50.7%/73.2%，几何条件Hash67.5%/74.3%，原生++69.7%/94.0%；阶段1.44/1.32/26.32分钟。当前原生++拟合差且慢，降低投入优先级，不宣布架构上限。[三行初筛](results/geobc_three_short_screens/BRIEF.md)。

两个低成本方法从自身1200检查点保留model、Adam、四RNG和病例顺序，继续至总4800／40遍：

| 方法 | 开发位移／应力误差 | 总训练阶段分钟 |
|---|---:|---:|
| FNO | **27.93%／56.06%** | 4.93 |
| 几何条件化Hash | 49.19%／64.96% | 3.43 |

FNO在20/20个开发CAD的位移、应力更低，热点13/20更低。TRAIN120分别19.5%/39.0%与39.3%/52.1%，拟合及泛化问题同时存在。新增训练分别157.94／100.51秒，两阶段总费用完整保留；开发读出81.89秒，120完整预测先验证再共享GT60一次，图已验看。[两行确认](results/geobc_two_warm_screens_v3/BRIEF.md)。

原双24000计划已主动终止，不冒充完成；同12000开发FNO26.5%/55.5%、R64++52.4%/84.0%，继续盲加步数收益不足。[同12000对照](results/geobc_short_screen/BRIEF.md)。初筛自动交接因初始argv空快照失败，实际评估完整且正常退出；根核对实际guard的相同PID/ticks完整身份后CPU导出，不重跑预测/GT，原失败保留，新guard已等待完整argv。[技术处理](notes/geobc_short_screen_handoff_failure_review.json)。

下一宏观问题是**扩大FNO训练CAD覆盖40→100**：候选预算内检查同4800更新及同40遍12000更新，沿用old40-only scaler、knownDEV60及reservedFINAL；独立v4科学计划和源码已冻结并通过真实上游终态放行，100CAD任务已于北京时间09:46启动，exec99080；4800及12000检查点，训练最多30分钟、整体40分钟，实际结果待完成。不进行Hash微小调参；连续全局几何/BC字段验证仍是完整GeoBC目标，原节点误差不能替代独立离节点FEM精度。

## 数据与资源

前瞻协议固定 **100 TRAIN／20 known DEV／20 reserved FINAL CAD**：TRAIN为旧40加新60，每CAD三载荷；scaler沿用仅旧TRAIN40拟合的版本。FINAL数值标签未读，不用于调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

420病例接口已整合：TRAIN300有已有目标，DEV60／FINAL60仅输入。两个实际TRAIN加载检查通过；旧int32／新int64索引兼容故障及原预算内修复保留，未验证全部300数值正文，100CAD实际载入与最大全节点梯度正在本次预算内验证。[接口简报](results/geobc_prospective_dataset_interface/BRIEF.md)。后续入口为 `geobc.prospective_dataset_v2.load_case`，明确选择40或100范围。

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

[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；最近已验收提交 `7d8d556` 包含同12000更新完整开发60例比较、原24k主动终止和三候选10遍短筛代码／计划。新10遍全部终态、warm40确认及100CAD新阶段现已在本地整理，待本次具体快照发布；不将本地文件冒充远端。上一发布50.63秒，精确commit和private状态均已核验。[发布计划](notes/github_publish_plan.json)。

更早逐阶段运行叙述存于[README历史快照](results/README_STATE_HISTORY_20261011.md)，状态以当前真实报告和运行收据为准。
