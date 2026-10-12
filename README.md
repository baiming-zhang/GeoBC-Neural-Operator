# GeoBC：原版DeepJEB求解器筛选

从几何、材料、载荷与约束预测位移和原signed应力，目标是新CAD泛化与连续空间查询。研究继续进行。

## 当前结论

**物理位移路线已通过固定20开发CAD×三载荷：原节点U误差平均0.01589%、最差0.02621%；原signed应力仍未解决，完整U/S求解器未选定。** 新增17CAD、旧3CAD基解复用，整阶段730.25秒；真实同20/60完整屏障与原GT60一次共享。来源为近似曲面P2／四Q／RBE3，开发集已迭代、未证明shape-family隔离；不是原OptiStruct配置精确复现或最终盲测。 [完整覆盖](results/geobc_physics_U_knownDEV_coverage_v107/BRIEF.md)。

神经联合场路线保留普通FNO作为默认基线，Hash保留作热点对照。以下是神经模型同20遍／三seed结果，不把完整FE输入与点云模型当等预算架构排名。

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

完整网格GPU定位已实际通过（11.75秒）：三个CAD全部117554／167316／150574单元进入Bezier保守索引，384内部点均找到原单元，2304位移查询与旧连续场完全一致，48包围盒外点全部拒绝。定位不接收参考单元编号，完整预测先提交再数值参考共享读出，GT／K或PCG重算0。查询无需调用者单元编号，但固定有界逆映射不是任意点全局可逆／C1／独立离节点FEM精度，应力仍open。 [结果](results/geobc_physical_U_global_locator_v106/BRIEF.md)。

固定20开发CAD的物理U完整覆盖已完成：60场平均误差0.01589%、最差0.02621%，全部真残差≤1e−8；新17CAD实际51个独立零初值解，旧3CAD九基解只复用，完整60字段先提交再GT60一次共享。外整阶段epoch730.25秒／mono686.09秒，子阶段epoch726.45／mono682.29秒，两时轴差异如实列出、不缩放；新17三载荷组共566.72秒，读出7.16秒。signedS、独立新混合／离节点GT与完整冷部署仍未验证。 [结果](results/geobc_physics_U_knownDEV_coverage_v107/BRIEF.md)。

六维参考点力／矩扩展也已实际通过（232.70秒）：三个已验收CAD新求Fy／Mx／My与三固定六维混合载荷，共18独立零初值解，真残差≤1e−8；六基解与独立混合解差≤1.09e−9，1152连续查询与192点六维载荷梯度通过。原RBE3实际支持REFC123456，不把旧三个载荷方向冒充六维；GT0、原K／粗空间不重建。这仍只覆盖固定材料／零支撑／同参考点的近似P2数值响应，无新OptiStruct载荷GT或S标签。 [六维响应](results/geobc_physics_U_fullrank6_v108/BRIEF.md)。

统一接口已实际通过（13.89秒）：`LocatedForce6UResponse.query(xyz_mm, force6)`不需要调用者单元编号，三CAD新72位移查询／24定位／48外点拒绝先完整提交，后置同场值差0，坐标／六维力矩梯度与直接物理API一致。另3个“8内部＋1外部”批次整批拒绝，无子集输出。查询／拒绝／分块每CAD0.370–0.605秒，属于已有基解缓存接口，不含新几何冷求解；GT／K／PCG重播0，无独立新载荷或离节点GT精度结论，S仍未提供。 [接口结果](results/geobc_located_force6_U_response_v109/BRIEF.md)。

冻结响应清单已实际加载验收（外UTC8.021秒／mono12.649秒，双时轴均保留）：3CAD各一次几何＋六基解严格hash/load，9份篡改清单均在body／constructor前拒绝；36位移查询／12定位／3外点拒绝完整提交后同场差0。单CAD加载0.806–1.101秒，查询／分块／拒绝0.315–0.522秒；只加载已有基解，非新几何全冷求解，GT／K／PCG重播0，S仍open。 [清单加载结果](results/geobc_physical_U_response_package_v110/BRIEF.md)。

两份新物理几何完整部署与原规范U准确率已验收：50_380／624_483每CAD一次源／新K／粗预条件，18独立零初值数值场和24接口查询／8定位／2拒绝通过；源到六基解77.940／63.438秒，整阶段外UTC315.596／mono297.088秒，含98.621秒根审等待及额外独立混合验证。 [部署结果](results/geobc_physical_U_new_geometry_deployment_v111/BRIEF.md)。另独立CPU读出7.858秒，六场原signed U均值0.015735%、最差0.026198%，原节点／GRID完整对齐；S值选择0，不重跑物理或调参，两个reserve已揭盲不是新盲测。 [原U精度](results/geobc_physical_U_reserve_accuracy_v112/BRIEF.md)。

参考点六维运动／能量已实际验证：12运动请求／72分量／12能量先屏障，后6独立mixed场对照，最大同方程运动相对差2.622e-10；原始互易性2.309e-10／8.720e-11，能量正定条件数13.204／19.275，4个六维力矩梯度通过，rawC未修正。恢复外UTC6.429秒／mono6.429秒，原失败9.737秒＋工程等待计入总UTC282.823秒，跨boot总mono未知null，原900秒截止未延长。源reference GRID没有H5输出行，derived IDs为空，预声明输出节点见证不可得，完整门仍FAIL／数值运动能量检查PASS，不生成替代节点或GT。 [实际报告](results/geobc_reference_motion_recovery_v114/BRIEF.md)。

六维弹簧边界已实际完成：两CAD／6修改方程独立零初值解，全6真残差≤9.979e-9，紧凑位移对独立修改K全原节点最大相对差1.263e-9、参考运动7.436e-10，能量方程差1.705e-14。原fineK／rawC／basis字节保持，弹簧反力纳入平衡；原固定32预条件器未改，GT0。外UTC74.496秒／mono74.495秒，两个独立组三路PCG32.672／27.933秒，紧凑边界更新＋全N输出0.018081／0.002057秒（已有基解缓存，非新几何全冷／NN等预算排名）。 [实际弹簧边界结果](results/geobc_reference_spring_BC_v115/BRIEF.md)。

精确参考点运动边界已实际完成：六个几何抬升／零空间零初值解，全6真投影残差≤9.995e-9、约束差≤6.060e-16；屏障之前物理基解读取0，之后12基解各一次用于紧凑响应，全N U最大相对差5.050e-7、反力5.577e-8、能量1.276e-10。抬升没有基解／GT warmstart，原K／rawC保持、反力功明确计入，非大弹簧近似。外UTC65.1808秒／mono65.1808秒，独立组三路PCG27.539／23.804秒，紧凑更新＋全N输出.005285／.001428秒（缓存scope，不NN公平排名）。 [实际精确运动边界结果](results/geobc_reference_prescribed_motion_v116/BRIEF.md)。

首个真实源RBE2支撑端口已实际完成：213002／188002各440成员，master H5输出行0／未造节点；两个CAD共12新独立零初值场，先完整屏障后12旧加载六基解读出，最大真自由残差9.994e-9、固定边界差0、九方向响应对独立mixed U相对差1.024e-8、支撑运动能量做功差3.921e-11。外UTC159.314秒／mono146.116秒，双时轴均保留；新支撑三基解34.765／34.901秒，独立混合41.803／34.672秒，缓存API＋全N .014623／.000986秒，未知工程coldnull／非NN公平排名。 [实际新支撑端口结果](results/geobc_support_translation_port_v117/BRIEF.md)。

四个真实源支撑的12平移方向已实际覆盖：v118仅求剩余端口1／2／3基解，旧port0不重跑；24新独立零初值场先完整屏障，再18旧force／port0字段各一次。全场混合最大相对差4.14e-08、真自由残差9.99e-09、固定边界最大差0mm，四端口同平移的刚体数值兼容门通过。实际外UTC391.924秒／mono359.587秒，两时轴不缩放。 [四端口实际结果](results/geobc_all_support_translation_v118/BRIEF.md)。

v119实际连续XYZ／force6／四支撑12平移接口已完成：48新U、16原单元owner、2盒外拒绝先屏障，再8直接同场对照；4实际耦合点坐标／力N／矩Nmm／支撑mm梯度非零有限且同直接接口差0，chunk和值差0、输入与基解字节保持。外UTC22.502秒／mono18.158秒，既有基解加载与新几何全冷成本分开；原signedS仍open。 [连续多边界实际结果](results/geobc_located_all_support_U_response_v119/BRIEF.md)。

下一宏观任务补齐四个真实RBE2支撑的转角：必须先从原source GRID取得真实reference XYZ，不能用成员中心或虚构输出节点；只新求旋转基解与独立混合，原力和支撑平移不重跑，声明小变形刚体旋转数值兼容／能量门。当前设计＋一个已审冻结SOURCE原型，import／constructor／科学数组／GPU0。 [真实支撑转角设计](notes/geobc_all_support_rigid_motion_v120_design.json)。

固定XGBoost3.0.5依赖已准备：首次下载95MB包后因351MB NCCL超时耗180秒；根据实际二进制依赖复用缓存离线安装2.57秒，无重复网络请求。当前已完成真实GPU全TRAIN首轮与400轮固定训练，首次失败与离线恢复成本仍保留。

## 数据与复现边界

原版DeepJEB v1.0，CSV按GRID坐标对齐H5 vertices，原signed U(mm)/S(MPa)、全部原节点，旧H5-only结果撤回。100 TRAIN/20 knownDEV/20 FINAL CAD各三载荷，CAD隔离，scaler仅旧TRAIN40。knownDEV已迭代且受来源可用性影响，三seed共享同20CAD，不是180独立病例；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

30项可恢复slides实验已完成。原单／双已知场约0.1%是拟合；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。原目录、部分25特征schema、网格原点和50样本集合缺失，不声称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

连续查询／梯度接口通过，不等于独立离节点FEM精度；原应力后处理及signed-VM契约仍缺失。[来源语义](notes/deepjeb_signed_scalar_source_semantics.md)。

## 运行与交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，显存30.5GB SI，公共GPU锁；计划与真实终态绑定，不因旧status重复启动。短阶段5分钟监视，预计超过1小时改30分钟。[依赖](requirements-lock-wsl.txt)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore`，未知内容拒绝覆盖。[私人GitHub](https://github.com/baiming-zhang/GeoBC-Neural-Operator)发布代码、配置、标量报告、图和检查点索引；原数据及项目训练检查点留本地。

私人仓库按阶段交付代码、配置、真实实验报告、图表和失败历史；各阶段的实施状态见上文与独立报告。每次发布均核实私人仓库身份、远端确切提交／树／变化文件；发布索引按下一次有意义实验快照更新。[发布状态](notes/github_publish_plan.json)。
