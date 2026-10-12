# GeoBC：原版DeepJEB求解器筛选

从几何、材料、载荷与约束预测位移和原signed应力，目标是新CAD泛化与连续空间查询。研究继续进行。

## 当前结论

**三种子对比后，默认采用普通FNO；Hash保留作热点对照，最终求解器仍未确定。**

| 同20遍／三seed／20个开发CAD×三载荷 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO | 21.72% | 49.33% | 37.53% |
| FNO + 条件Hash | 21.97% | 48.71% | 33.24% |

Hash应力只好0.62个百分点，联合误差好0.19个百分点，未达到预声明3个百分点推进门；热点好4.30个百分点。参数1225万对453万，新增seed学习6.39对5.90分钟，60场纯推理0.82对0.96秒。当前最好的Transolver同20遍／单seed位移35.79%、应力59.38%，仍落后；这些结果不代表架构上限。[三seed结果与成本](results/geobc_solver_seed_readout_v83/BRIEF.md)。

## 最近推进

几何点云规范化短对照已完成（45.48秒）：应力41.69%→40.72%，位移22.69%→29.57%，固定配置关闭。原符号／幅值联合头短验证也未通过门，不继续小调。[几何对照](results/geobc_canonical_shape_encoding_v86/BRIEF.md)。

位移真值的物理闭合诊断已完成（7.70秒）：固定恢复公式的应力幅值误差仍10.78%，旧有限元位移为10.77%。这部分差异主要不是位移求解造成；不能把此proxy当原应力精确定义，也不能当49%神经模型误差的下限。原signed标签保持，不猜符号规则。[诊断](results/geobc_reference_vm_closure_v87/BRIEF.md)。

局部空间FNO对照已完成（47.10秒）：位移22.69%→21.13%，应力41.69%→42.00%，未通过门，固定分支关闭。[结果](results/geobc_local_spatial_operator_v88/BRIEF.md)。

官方U-NO短筛选已完成：有效训练／读出63.42秒，九TRAIN场位移31.40%、应力56.21%，均劣于原FNO22.69%／41.69%，固定配置关闭。两次训练前接口失败均0标签／0更新，含修复等待的原任务总918.45秒；原1200秒总截止未重置。[结果](results/geobc_official_UNO_valid_skip_recovery_v91/BRIEF.md)。

GPU物理张量树模型已完成（整阶段19.90秒／GPU学习9.52秒）：旧两例knownDEV的应力误差30.92%，物理特征MLP为31.76%，只改善0.84个百分点；热点53.80%→36.57%，仍未达到固定推进门，配置关闭。它只覆盖旧3TRAIN／2knownDEV／ver，不是主100／20三载荷协议的公平架构排名，也不是可微连续求解器；20秒不包括继承的物理输入冷启动成本。[结果](results/geobc_physical_tensor_GBDT_v92/BRIEF.md)。

两个此前未使用标签的预留CAD×三载荷检查已完成（45.82秒）：FNO位移18.65%／应力45.12%，FNO+Hash18.55%／44.77%，Transolver36.69%／61.24%。Hash整体优势仍小，热点33.52%→29.54%；默认FNO保持。全部18份预测先提交再共享读两CSV原标签，无新训练／scaler拟合。只有两CAD，不能确定总体赢家；它们现已揭盲，禁止据此调参。 [结果](results/geobc_reserve_holdout_v93/BRIEF.md)。

动态图几何编码筛选已完成（68.87秒／学习58.37秒）：九TRAIN场应力41.69%→40.34%，位移22.69%→21.80%，联合只改善1.12个百分点，未达到固定3pp门；结构关闭，不扩大100CAD或扫参数。算子／查询头初始字节与真实242936全节点梯度资格已通过。 [结果](results/geobc_edgeconv_geometry_v94/BRIEF.md)。

载荷幅值一致性检查已完成（51.25秒）：FNO／Hash／Transolver的平均位移齐次响应偏差为45.62%／38.17%／44.27%，三者未通过固定1%诊断门。完整81预测／54性质、真实力／材料／零SPC与同步grid缩放见证通过，GT／训练／scaler拟合0；这是相对各自幅值1预测的物理一致性偏差，不是GT误差，原source控制局限保留。 [结果](results/geobc_load_amplitude_homogeneity_v95/BRIEF.md)。

显式载荷线性位移接口已实际验证（66.48秒）：三个冻结骨干共108份全节点预测，幅值／叠加一致性偏差降至浮点舍入量级，旧三TRAIN CAD的规范U预测和原encodedS保持，真实坐标／秩3载荷系数梯度通过。GT／训练0；这不是新载荷准确率改善。 [接口结果](results/geobc_load_linear_adapter_v96/BRIEF.md)。

新载荷对照已完成：20个knownDEV CAD的条件混合位移误差，FNO 41.93%→22.07%，Hash 37.42%→21.25%，Transolver 58.48%→37.54%；原规范载荷U／encodedS逐案例变化0，原GT误差保持。完整540预测先提交／复核再共享GT60一次。原计算217.09秒、CPU恢复228.38秒；含路径校验失败、WSL时轴恢复及工程等待的原总UTC跨度1085.18秒，预算未重置。原常驻输入的最终digest检查未执行，该项证据未知；不重读输入／PT或重播推理。**混合U参考仍为条件线性派生值，尚非独立新FEM真值。** [结果](results/geobc_load_linear_identity_readout_v99/BRIEF.md)。

独立近似物理对照已完成：固定120_153、六个独立零初值解，全6真残差≤1e−8，原ver／hor／tor位移误差0.01623%／0.02620%／0.00511%，兼容门通过。对独立近似P2混合位移，FNO39.93%→14.66%，Hash36.62%→15.56%，Transolver58.46%→29.03%；只覆盖一个已迭代开发CAD，不确定总体赢家或原OptiStruct新载荷真值。GPU原阶段164.15秒、两并行组各约77秒，dtype读出适配恢复3.93秒，含修复等待原总UTC918.34秒；无重播K／PCG／输入，GT3共享一次。原S问题仍未解决。 [结果](results/geobc_load_linear_physics_readout_v101/BRIEF.md)。

固定32步粗空间预条件器六载荷验证已通过：完整新阶段58.30秒，对原两组153.73秒减少62.08%，全6真残差≤1e−8、同原节点位移相对差约1e−10。缓存／粗构造2.14秒、两组49.77秒、读出0.57秒；GT0。仍是一个已迭代CAD／共享K的多载荷成本，不是fresh冷请求或原OptiStruct精确重放，S未解决。 [结果](results/geobc_sixload_coarse_preconditioner_v102/BRIEF.md)。

两个额外开发CAD的配对物理验证已完成（432.91秒）：131_345／131_351原三载荷位移误差均≤0.02620%，全部18个独立零初值解真残差≤1e−8，12个数值同解对照差≤1.91e−9。普通三载荷132.39／109.19秒，粗构造加粗三载荷48.75／38.11秒，分别提速63.17%／65.10%。完整18字段先提交再GT6共享一次；无新神经训练／推理。来源仍为近似P2／四Q／RBE3，两个已迭代CAD不证明总体族隔离盲测，signedS未解决。 [结果](results/geobc_physics_crossCAD_paired_v103/BRIEF.md)。

连续物理位移接口已完成：三CAD完整2304内部查询／960节点位置／192梯度点，原节点值和坐标／秩3载荷梯度、固定步长中心差分均通过。GPU阶段18.82秒，CPU整数dtype读出恢复2.71秒，含修复等待原UTC292.77秒，预算未重置／无GPU或标签重播；每CAD768次缓存查询与对照0.206–0.243秒。明确需要有效TET10单元编号，尚无全CAD定位／C1／独立离节点GT精度，应力未解决。 [结果](results/geobc_physical_U_query_readout_v105/BRIEF.md)。

下一步固定三个CAD，建立完整网格GPU空间索引和物理坐标定位，去掉调用者提供单元编号的要求；2304预测及定位决定先提交后只读已保存数值参考，GT0、无物理求解重播。当前仅设计，SOURCE／数组／GPU0。 [设计](notes/geobc_physical_U_global_locator_v106_design.json)。

固定XGBoost3.0.5依赖已准备：首次下载95MB包后因351MB NCCL超时耗180秒；根据实际二进制依赖复用缓存离线安装2.57秒，无重复网络请求。当前已完成真实GPU全TRAIN首轮与400轮固定训练，首次失败与离线恢复成本仍保留。

## 数据与复现边界

原版DeepJEB v1.0，CSV按GRID坐标对齐H5 vertices，原signed U(mm)/S(MPa)、全部原节点，旧H5-only结果撤回。100 TRAIN/20 knownDEV/20 FINAL CAD各三载荷，CAD隔离，scaler仅旧TRAIN40。knownDEV已迭代且受来源可用性影响，三seed共享同20CAD，不是180独立病例；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

30项可恢复slides实验已完成。原单／双已知场约0.1%是拟合；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。原目录、部分25特征schema、网格原点和50样本集合缺失，不声称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

连续查询／梯度接口通过，不等于独立离节点FEM精度；原应力后处理及signed-VM契约仍缺失。[来源语义](notes/deepjeb_signed_scalar_source_semantics.md)。

## 运行与交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，显存30.5GB SI，公共GPU锁；计划与真实终态绑定，不因旧status重复启动。短阶段5分钟监视，预计超过1小时改30分钟。[依赖](requirements-lock-wsl.txt)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore`，未知内容拒绝覆盖。[私人GitHub](https://github.com/baiming-zhang/GeoBC-Neural-Operator)发布代码、配置、标量报告、图和检查点索引；原数据及项目训练检查点留本地。

私人仓库按阶段交付代码、配置、真实实验报告、图表和失败历史；各阶段的实施状态见上文与独立报告。每次发布均核实私人仓库身份、远端确切提交／树／变化文件；发布索引按下一次有意义实验快照更新。[发布状态](notes/github_publish_plan.json)。
