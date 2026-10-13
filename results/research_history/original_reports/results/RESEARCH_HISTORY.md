# GeoBC研究过程记录

以下为第102项盲验交付前的阶段记录，时间指向各原报告；最新判断见[一分钟结果](SOLVER_BRIEF.md)。

# GeoBC：原版 DeepJEB 复现与泛化探索

**当前阶段（2026-10-10）：先筛选求解器，尚未选定最终方案；Hash专项微调已停止。** 十类学习结构探索后，FNO在现有配方下较有依据，但未知CAD的应力误差仍约50%。近似物理求解器在另一组固定五CAD/ver上，位移误差约0.0162%，全部真收敛。固定粗空间两例64秒、新增三例82秒，五例全部保持同方程、同数值解，完整核算冷成本77–101秒→31–42秒，降低55%–64%，通过五例25%收益门。它仍未解决原signed scalar stress，不能视为完整GeoBC方案。学习与物理评估集合、接口信息不同，不能把数字直接拼成精度排名。

**当前实际运行：** 五CAD/ver的位移盲验于北京时间2026-10-10 09:34启动，比较完整FE物理U与冻结FNO端点。两臂十个独立进程请求，所有原节点与物理四积分点预测完成后才共享读取原CSV U；累计预算40分钟。此次只检验U及节点U派生的P2梯度代理，所有新signed应力标签保持封存。尚无本轮精度结论。

**物理信息明显有用，完整应力仍未解决。** 同MLP加入六个物理张量槽的46秒粗筛，DEV signed stress均值31.8%、热点53.8%，未通过预定门。固定幅值诊断2.24秒，DEV均值／最差10.10%／10.56%，热点17.00%，严格10%门inconclusive。独立的物理幅值乘小模型因子候选4.79秒完成固定60次更新：DEV均值／最差30.41%／30.83%、热点29.72%，仍未达25%门；关闭该固定表示，不追加训练或微调Hash。一次[源输出接口核实](notes/geobc_signed_output_interface.md)已结束：原H3D请求可确认，CSV精确selector／符号／节点恢复仍未知，不枚举公式挑DEV最佳者。同CAD／同载荷的已有服务证据与完整请求计时已核对，分别判断位移与完整U/S任务，保留输入接口和成本差异。FNO初值配方已结束；新test5的signed应力标签继续封存，位移标签按预测屏障规则读取。先读[一分钟判断](results/SOLVER_BRIEF.md)，细节见[物理位移](results/elastic_crosscad/BRIEF.md)、[应力读出](results/elastic_stress_readout/BRIEF.md)、[因子候选](results/elastic_factored_stress/BRIEF.md)、[五例粗空间成本](results/elastic_coarse_extension/BRIEF.md)。

**最新两个选型实验已结束。** [同DEV2服务计时](results/solver_service/BRIEF.md)四次独立完整请求共10.72秒，FNO为2.62–3.68秒、MLP约2.04–2.05秒；起点为已给xyz／25条件，原CAD到条件生成的上游成本未包含，精度只引用已有端点。原node-only接口下速度可接受，signed应力精度仍不足。[直接粗空间候选](results/elastic_low_fidelity/BRIEF.md)实际51.07秒、含继承setup共67.91秒：两例true coarse1e-8，但对原fine数值位移差5.95%／7.76%，accounted cold32.97／35.03秒，精度和50%提速门均失败。关闭固定候选，不追加救参；它的投影平衡不等于完整fine平衡，也没有GT评价。保留粗空间作为完整物理求解预条件。

**最新应力对照已结束：** 父阶段按固定1800秒预算终止，两臂未完成公平600／600秒对照。独立冻结端点读出6.31秒：局部／FNO的DEV平均signed应力55.84%／48.74%，最差热点66.52%／46.70%，都未达25%门；检查点累计计费600.05／120.48秒，不能宣称FNO胜出。失败耗时与额外缓存读取均记录，原U字节保持。见[短报告](results/elastic_spatial_readout/BRIEF.md)。**连续场检查已完成：** 五CAD全部684,470单元／2,737,880四积分点，新增实际14.25秒；同P2 GT节点U参考下，物理U误差约0.01622%，梯度0.02394%–0.04172%，对称应变张量代理0.11782%–0.19889%。全部完整覆盖、C0／有界逆查询检查通过；非源求解器应变标签，也未验证快速全CAD定位器。见[连续场短报告](results/elastic_continuous_capability/BRIEF.md)。[节点张量对照](results/elastic_nodal_tensor_pair/BRIEF.md)新增10.95秒完成：两臂相同fresh7头各120完整CAD更新，完整预测后评价；节点张量没有改善，极端外推误差主导L2，两个门均失败，关闭固定候选。新test5的signed应力标签保持封存；FNO学习参考与物理U在全新CAD上的盲验接口已冻结，尚未选择最终solver。

**现有证据支持Hash的小样本拟合能力，尚不支持将它选定为GeoBC求解器。** 原实验主要测训练拟合，此前探索集中于Hash；现在已补上共同CAD划分下的六家族筛选、成本确认与最终检验。关闭Hash最后4层曾把旧开发验证位移均值22.23%降至19.97%，但应变代理仍差；它与本轮signed四场预测的配方不同，不能直接拼成跨架构排名。[选型计划](notes/geobc_solver_selection_plan.json)记录初筛范围。

首轮各480次完整案例更新，FNO验证位移／应力47.5%／66.7%；MLP为71.6%／97.6%，作为低成本对照。Hash为92.5%／92.8%，但最差位移191.4%且后半程位移反升，因此暂缓；Transolver、Transolver++、Transolver3仍接近零场基线。六家族的共同checkpoint与曲线见[初筛简表](results/solver_selection/BRIEF.md)。四次训练遍历不足以证明收敛；各家原生容量差异及FNO point-to-grid插值限制均明示。

已完成[固定成本确认](notes/geobc_solver_confirmation_plan.json)：FNO、MLP和一条低优先级慢结构路线Transolver++，各用2026／2027两个种子、每项累计600训练秒，串行队列共62.3分钟。2026从screen最后一步继承模型、Adam与全部RNG，父阶段时间计入600秒；2027重新初始化，复用相同数据、几何云与仅TRAIN拟合的scaler。完整120-case pass末停止并记录超预算时间，固定成本端点为主要比较，共同VAL-best另报；未改损失或家族参数。见[确认报告](results/solver_confirmation/BRIEF.md)。

两个seed的固定预算端点U／stress分别为FNO **36.28%／56.45%、36.22%／56.07%**，MLP **51.69%／75.62%、43.01%／73.24%**，++ **90.32%／98.45%、103.55%／98.57%**。FNO在这套配方与预算下最有依据，MLP保留为点式低成本对照；两seed不是统计显著性证明。FNO共同VAL-best第2640步分别为25.07%／55.44%、26.28%／55.05%；后段位移回退，因此端点与best分别报告。

最终评估的隔离入口已完成[源码检查](notes/solver_final_eval_source_check.json)，[规则](notes/geobc_solver_final_eval_plan.json)已在读取最终数据前冻结：FNO／MLP、两个seed、固定预算端点为主及既有VAL-best为次。输入准备6.1秒，160次全节点case推理及评估51.1秒，零训练、零scaler拟合。[final5结果](results/solver_final_eval/BRIEF.md)按known3与dia分栏：已知载荷FNO U33.09–36.17%／S49.23–50.39%，MLP U26.53–49.40%／S64.20–73.49%。MLP一个seed位移更低，FNO应力及跨seed表现较稳定，不能宣称所有指标都赢。

组合载荷dia的FNO U46.43–76.51%／S68.78–96.31%，MLP U57.71–131.70%／S84.19–98.90%，仍明显不足。最终结果不用于选seed、checkpoint或重训；后续适配研究使用另外冻结的test5，并在新路线、预算与checkpoint规则锁定前保持字段封存。

固定TRAIN案例的179,316个节点完成网格边界角色回查（1.7秒）：支持／载荷角色正节点重建均值为0.390／0.379，局部标记受到明显稀释。角色是描述符，非精确节点Dirichlet；这项输入诊断不证明预测误差的原因。

[连续能力诊断](results/solver_capability/BRIEF.md)已完成（3.718秒）：固定TRAIN101_479/ver的32 cells、8 faces、504查询，四个已锁FNO/MLP端点的几何与数值guards均通过。FNO节点U/S较准，但共同十节点P2代理误差507–610%，原生网格梯度535–640%；MLP共同P2为413–421%，原生offnode因原25输入延拓未定义而未实现。GT-P2 RMS约8.58e−4，FNO2026误差RMS约5.23e−3；梯度差距不能只看节点误差。P1/P2是这一局部的GT节点派生proxy，非solver应变GT；结构性C0通过不等于学习成功，不能外推全CAD。

[GINO候选](notes/solver_gino_candidate_design.md)的单TRAIN输入图可行性已完成（2.245秒）：R64、固定radius0.033，共6,547,680 pairs，全部179,316点有邻居，CSR缓存216,204,464 B，实测构图peak约589 MB。这仅测邻域枚举／排序／CSR，未读取BC或标签、未构造模型，不能当作GINO前反向资源或连续场精度。独立test5见[封存计划](notes/geobc_next_cycle_test_plan.json)；15个文件已下载完成，未解析任何字段，未替换CAD或改变旧数据；旧失败记录保留在独立attempt目录。

[GINO模型资源检查](results/gino_resource/BRIEF.md)另已完成4.985秒，8/8 checks PASS：全部179,316点输入encode 2.003秒、512点query 0.0437秒、一次无目标backward 0.3004秒，peak allocated/reserved为1.829/3.813 GB；另含4次小真实CSR后端autograd核对。零labels、零optimizer更新，没有训练或精度结果，此历史低频资源检查未验全N输出；随后匹配频带的480步正式训练已通过完整节点前反向。实际参数1,433,277（含PointNet，complex tensor按numel计），与旧FNO约14.18M的容量及频率实现不匹配，后续差异不能只归因于GNO。

新增拓扑邻居消息的四几何拟合对照用时56.6秒：位移10.59%→12.87%，P1约219%→219%，没有通过联合门限。两组初始化、采样、实际参数与首反向核对通过；这一固定预算结果不代表所有拓扑方法都无效。

连续查询几何接口已在一个TRAIN案例通过有界验收（45.1秒）。后续冻结场全CAD诊断（3.1秒）发现，同一案例的四角P1／完整十节点位移派生应变代理误差为22.4%／187.3%。这提示单看节点位移或P1会遗漏梯度问题；它不是solver应变GT，也没有证明所有Hash都存在同样缺陷。该发现保留为候选求解器后续一致检验的依据，当前不继续Hash专属局部优化。

![完整12层与前8层的验证结果](results/geobc_exploration/hash_bandwidth_comparison.png)

单CAD位移／P1代理可拟合到0.76%／17.9%；可靠跨几何预测仍未解决。先读 [一分钟结果](results/geobc_exploration/BRIEF.md)；详细证据见 [探索报告](results/geobc_exploration/REPORT.md)，执行计划见 [notes/geobc_exploration_plan.json](notes/geobc_exploration_plan.json)。

GeoBC Neural Operator 研究根据几何、材料、载荷和约束预测空间坐标处的位移与应力。当前已实现整体点云、局部边界几何及位移载荷响应基；未见 CAD 预测误差仍高，也未验证重网格泛化。研究主线是以几何和边界接口条件化连续场，同时保留应力集中处的局部细节；具体新方法及其有效性需要后续验证。

![固定验证案例的位移预测对照](results/geobc_exploration/load_transfer.png)

上图固定使用第一个代表案例102_249/dia，显示同一批7,822个原节点，前三面板共用位移模长mm色标。模长仅用于可视化，训练及指标使用原始signed uXYZ；此单案例不代表五CAD汇总。

使用 **原版 DeepJEB v1.0**，复现 [参考 slides](https://docs.google.com/presentation/d/1g5S0QCZSDR_igYFWFe20SQ3nw2yvgYbXk_t4QSmKpBU/edit) 的可恢复量化实验，优先单样本和五样本。30/30 个计划实验已完成，正式队列用时约3小时26分钟。完整本地结果见 [ALL_METHODS_RESULTS.md](ALL_METHODS_RESULTS.md)，可移植结果包见 [results/REPORT.md](results/REPORT.md)，全部运行命令见 [notes/all_methods_plan.json](notes/all_methods_plan.json)。每项完成状态和实际物理指标保存于对应 `summary.json`。

主要观测：坐标 MLP 单案例应力 L2 为39.52%；官方 FullyFused hash 单案例为0.414%，五案例平均/最差为1.450%/2.112%；192³ FNO 回查原始节点为14.34%。替代50案例共享场平均 L2 从24.43%降到16.17%（pooled13.94%，最差28.97%，最佳 epoch810）。TS3 长预算 pooled应力L2为74.40%，未达到原稿数值；其历史架构/优化配置缺失，不能把这次结果视为原 run 的精确重放。

原实验目录已不可访问。已恢复的案例、模型尺寸、输入/输出及更新预算按 slides 执行；缺失的学习率、缩放常数、部分模型配置和条件特征定义均在每次运行的配置中明示。不能把替代配置的结果称为原代码的精确数值复现。

> **数据对齐更正：旧 H5-only 结果已撤回物理场结论。** H5 中的节点场数组与 `vertices` 的排列不同，旧流程未核对逐节点对应。新数据按官方 Field CSV 的 GRID 坐标对齐到 H5 节点，训练入口拒绝没有对齐标记的数据。旧文件保留在 `data/prepared`、旧 `runs` 和 `plots/final`，仅供历史记录；[REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) 已标注撤回。

