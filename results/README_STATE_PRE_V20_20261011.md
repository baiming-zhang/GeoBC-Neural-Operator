# GeoBC：DeepJEB 场预测与求解器筛选

根据几何、材料、载荷与约束，预测原节点及连续空间的位移和 signed 应力。使用**原版 DeepJEB**；最终求解器尚未选定。

## 目前结论

**冻结后20个独立CAD×三载荷测试已完成：FNO位移／应力22.39%／48.54%，全局Hash24.29%／48.44%。继续以FNO为主模型；Hash热点34.05% vs FNO41.69%，20/20 CAD改善，保留热点对照。绝对误差仍高，求解器筛选继续。**

[一分钟结果](results/SOLVER_BRIEF.md) · [冻结后独立测试](results/geobc_FINAL_v9/BRIEF.md) · [同训练量开发比较](results/geobc_global_hash_matched_v8/BRIEF.md)。历史同40CAD／12000／约4.53M公平对照：FNO26.50%／55.55%，Transolver++52.35%／83.99%；目前TS系列以++为主，当前配方FNO优先，不宣布架构上限。Hash<1%是已知少样本应力拟合，不等于新CAD联合U/S精度。

## 最新进展

全局Hash宏观上下文使几何Hash新CAD误差明显降低，但最终同40遍曝光未全面胜出，因此停止Hash追加或微调。headlatentzero和geomzero只作推理依赖诊断；真实官方TCNN训练后连续query通过节点／分块／重排一致性及输入梯度，未测独立离节点FEM精度。[4800宏观对照](results/geobc_operator_hash_v5/BRIEF.md) · [CUDA连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

从自身4800 checkpoint恢复model／Adam／四RNG／cursor，仅追加7200步到12000，训练444.68秒、完整阶段605.68秒；新开发完整60份预测先提交／重验再共享knownGT60一次，74.38秒，CPU导出5.65秒，图已验看。初始元数据失败126秒、修复gap及原stage1011秒保留，无预算重置，无GT或预测重播。

FNO主模型由knownDEV联合误差在FINAL标签读取前选定。FINAL两端点120份全节点预测先保存／重验，再20个CSV各解析一次并共享60例场，完整阶段155.38秒；CPU报告13.27秒。云图色条排版另用已有数组修复13.42秒，原图／评分保留，修复图已验看，无CSV重读或新推理。[位移云图](results/geobc_FINAL_v9/plots_v2/field_U.png) · [应力云图](results/geobc_FINAL_v9/plots_v2/field_S.png)。

这一批FINAL现已揭盲，不能再当作未来选型的盲测。停止Hash追加；下一项宏观候选是100CAD上的Transolver++短筛，与FNO配对，先检验GPU常驻缓存、实际显存与训练速度，再决定短端点预算。资源验证已实际完成70秒，最大节点全反传通过，allocated／reserved峰值18.91／20.09GB；native++100CAD的5遍／1500步短筛已正常完成，总耗时34.68分钟，训练集全300例位移／应力误差65.81%／95.02%，仍明显未拟合；不能据此宣称架构上限。FNO的同样5遍对照及knownDEV20个新CAD×三载荷评估均已完成：FNO位移／应力60.56%／74.58%，native++67.43%／94.86%；纯训练1.45／32.08分钟。两者均未充分拟合，当前FNO初筛更优；[5遍简报](results/geobc_macro100_short5_v11/BRIEF.md)。物理相对误差损失的FNO100CAD／5遍验证已完成：新几何knownDEV位移／应力从60.56%／74.58%降至31.61%／56.10%，热点从70.09%降至45.08%；整轮4.27分钟，DEV读出68.07秒，CPU简报3.12秒，图已验看。[物理损失简报](results/geobc_shared_physical_v12/BRIEF.md)。native++同模型／同5遍／同物理损失对照已正常完成：TRAIN位移／应力55.72%／93.81%，应力仅改善1.21个百分点，未通过预设门槛；完整阶段34.64分钟、纯训练31.81分钟。新++DEV未执行，结果不能当作新几何误差。[训练负结果](results/geobc_native_physical_TRAIN_v13/BRIEF.md)。FNO从自身5遍完整状态继续到10遍已完成：knownDEV位移／应力25.60%／52.14%，热点41.93%，较5遍改善6.01／3.95／3.15个百分点。追加5遍纯训练97.18秒、完整阶段4.24分钟，开发评估77.76秒，CPU简报7.73秒；图已验看。[5→10遍简报](results/geobc_physical10_v16/BRIEF.md)。这是增加曝光量的结果，不能冒充同训练量方法比较；应力仍高，不默认40遍或24000步。同物理损失、同100CAD／5遍全局Hash对照也已完成：knownDEV U/S31.37%／54.45%、热点40.85%，比FNO5遍联合误差低0.94个百分点、热点低4.22个百分点，但未达预设3个百分点推进门槛，停止追加Hash。恢复段3.92分钟／纯训练104.76秒，首0更新失败138.68秒及修复gap保留，原起点至训练child终态跨度10.53分钟；DEV73.21秒、CPU简报5.12秒，图已验看。[同5遍解码器对照](results/geobc_global_hash_physical_v18/BRIEF.md)。Transolver／++／3紧凑全局空间token表示已通过6次真实全节点GPU反传，整轮8.19秒，最大病例单次反传约0.08／0.13／0.15秒；这仅是资源验证，完整100CAD训练与开发误差尚未发生。[资源简报](results/geobc_coarse_transolver_resource_v19/BRIEF.md)。

## 数据与资源

前瞻协议固定 **100 TRAIN／20 known DEV／20 FINAL CAD**：TRAIN为旧40加新60，每CAD三载荷；scaler沿用仅旧TRAIN40拟合的版本。三个集合CAD无交集，FINAL也不在原Hash50训练CAD中；本次FINAL仅一次冻结后评分。来源是可用官方TRAIN池，非官方独立测试或无偏抽样，单seed结果。[划分](notes/geobc_prospective_cad_split_v1.json)。

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

[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；最新已验收提交 `c849467`。本地已完成12000步同曝光比较及FINAL测试；本批12000端点和独立测试代码／结果已推送并验收 `50d45b4`（68.66秒），v10资源验证与v11短筛／配对评估代码结果已推送验收 `93651e7`（70.77秒），v12物理损失结果和v13 native++物理5遍启动代码已推送验收 `aa40a0a`（63.19秒）；v13负结果、v15／v16完成结果与图已推送验收 `54f8616`（94.55秒）；v17的0更新故障、v18真实恢复／同5遍开发结果、v19三版本真实GPU资源验证已推送验收 `c849467`（67.95秒）；紧凑Transolver完整训练和开发精度尚未发生，远端状态以[最新验收记录](notes/github_publish_plan.json)为准。[发布计划](notes/github_publish_plan.json)。

更早逐阶段运行叙述存于[README历史快照](results/README_STATE_HISTORY_20261011.md)，状态以当前真实报告和运行收据为准。

冻结source pins记录的是原运行文件字节，Git可能规范化行尾。新克隆运行当前v7／v8／v9／v10／v11／v12／v13／v15／v16协议前，先执行 `python tools/restore_geobc_source_line_endings.py --check`；如提示仅行尾不同，执行 `--restore` 可恢复匹配的原LF／CRLF字节。仅校验当前127个冻结依赖与交接工具，未知内容拒绝写入，不改科学pins、目标或代码逻辑；旧历史版本的代码差异不由此恢复。在WSL从项目根目录执行研究工具时使用 `python -m tools.<模块名>`，以保证项目包可导入。

三版紧凑Transolver同100CAD／5遍短筛已实际启动，共用一次GPU数据缓存；训练与新精度尚未完成。固定计划与运行身份见 `notes/geobc_coarse_three100_physical5_v20_plan.json`、`notes/geobc_coarse_three_v20_actual_launch.json`。完成后统一比较20个knownDEV CAD×三载荷，保持FINAL封存于未来选型之外。
