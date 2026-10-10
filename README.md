# GeoBC：DeepJEB 场预测与求解器筛选

根据几何、材料、载荷与约束，预测原节点及连续空间的位移和 signed 应力。使用**原版 DeepJEB**；最终求解器尚未选定。

## 目前结论

**当前配方 FNO 领先，Transolver 系列优先保留 ++；Hash 仍是连续场候选。** 最新共同对照使用相同 R64 输入、native25 查询头、约4.53M参数、TRAIN40 CAD×三载荷和4800次更新。

| 方法 | TRAIN 位移／应力误差 | 新20 CAD×三载荷位移／应力误差 | 训练 UTC 分钟 |
|---|---:|---:|---:|
| FNO | 19.5%／39.0% | 27.9%／56.1% | 26.97 |
| Transolver++ | 40.6%／85.9% | 41.4%／86.4% | 144.36 |

以上为全原节点、物理单位 relative-L2 的病例均值；新20 CAD是已知开发池，不能当作最终测试。这说明当前配方的差距，不能宣布某个架构的上限。见[共同配置比较](results/geobc_common_grid_comparison/BRIEF.md)。

Hash 的约0.4%来自单病例已知应力拟合，五病例约1.45%；它与跨CAD、联合位移／应力任务不同。FNO 在独立单病例4800更新可达到位移3.9%／应力11.5%，表明共享训练误差包含拟合不足。[单例诊断](results/geobc_single_case_fit/BRIEF.md) · [Hash口径](notes/geobc_hash_error_comparability.json)。

## 正在推进

两臂从各自完整4800检查点继续至12000／24000更新，保留Adam、RNG和病例顺序；比较每病例40／100／200次曝光，不改变损失或scaler。

- FNO 已完成24000：TRAIN120位移／应力为 **10.5%／24.2%**，新增训练UTC10.23分钟。新的开发读出尚未执行。
- ++仍在同端点训练；真实运行状态见[轻量监视收据](notes/geobc_exposure_live_milestone.json)。整个配对尚未完成。
- 两臂结束后固定读取相同开发60例，再决定推进原生网格Transolver、条件化连续Hash或数据覆盖度对照。[计划](notes/geobc_exposure_extension_plan.json) · [宏观问题](notes/geobc_macro_conditioning_source_review.json)。

当前模型通过网格和查询特征接收边界条件；全局PointNet只编码几何。历史粗粒度约束／加载区点云消融未带来明确收益，下一轮避免重复该配方。

## 数据与资源

前瞻协议固定 **100 TRAIN／20 known DEV／20 reserved FINAL CAD**：TRAIN为旧40加新60，每CAD三载荷；scaler沿用仅旧TRAIN40拟合的版本。FINAL数值标签未读，不用于调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

420病例接口已整合：TRAIN300有已有目标，DEV60／FINAL60仅输入。两个实际TRAIN加载检查通过；旧int32／新int64索引兼容故障及原预算内修复保留，未验证全部300数值正文，也未启动新100CAD训练。[接口简报](results/geobc_prospective_dataset_interface/BRIEF.md)。后续入口为 `geobc.prospective_dataset_v2.load_case`，明确选择40或100范围。

用户显存预算 **30.5 GB**（30,500,000,000 bytes）。100CAD R64全缓存加当前++工作空间估计28.64GB，后续优先GPU常驻；必须以实际全节点allocated／reserved各自峰值确认。[资源预算](notes/geobc_user_resource_budget.json) · [缓存估算](notes/geobc_prospective_cache_memory_plan.json)。当前运行实验保留自身较小的原24GiB上限。

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

[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；最近已验收提交 `49f137b` 包含CPU80输入准备与FNO24000训练端点。当前420数据接口、兼容修复及新资源预算正在形成下一次快照，尚不声称已上传。[发布计划](notes/github_publish_plan.json)。

更早逐阶段运行叙述存于[README历史快照](results/README_STATE_HISTORY_20261011.md)，状态以当前真实报告和运行收据为准。
