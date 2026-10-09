# 纯FNO／纯物理／冻结初值＋物理校正（proposal_only）

2026-10-10。仅源码和小JSON metadata阅读；没有读数组、权重、GT、旧final或封存测试，没有模型/GPU执行。当前五CAD物理阶段仍由root唯一执行；本文不启动任务、不授权标签读取。若该阶段模型不兼容、未收敛或物理误差明显不符，**不要自动进入hybrid**，先由root判断模型可信度。

固定原顺序、原ver：TRAIN `101_428,101_56,101_78`；DEV `102_249,102_38`。失败不换CAD，无新训练、seed选择、Hash修改或预条件器扫描。

| 路线 | 输出与数值责任 | 比较角色 |
|---|---|---|
| 纯FNO | 冻结预测的原节点physical u；不保证1e−8残差 | 低成本近似值，误差、约束与真实残差另报 |
| 纯物理PCG | 零初值、既有近似K/真实约束/同block-Jacobi | 基准已收敛解及完整冷启动成本 |
| 冻结FNO＋PCG | 可行自由DOF初值，随后同K/同预条件/同停止规则 | 检验是否明显降低获得同数值精度的总成本 |

**冻结身份已确认。** `runs/geobc_exploration/solver_confirm_fno_seed2026_600s/last.pt`，不是best.pt或2027。完成summary为FNO/2026/endpoint2880；训练实际608.67788秒（完整pass overshoot），权重stat为170,464,827B，未读取或重新hash权重。实际首次加载核`step=2880`、canonical JSON配置/source_config与strict state_dict；训练成本独列，不能称免费初值。

API为 [SelectionModel](../geobc/solver_selection_models.py#L59)：`SelectionModel('fno',device,2026)`→`encode_geometry(cloud,mask)`→`forward_grid(grid29,ctx32)`，再 [query_grid](../geobc/solver_selection_data.py#L97) 查全部原节点。五case输入在 `data/geobc_solver_selection/manifest.json.records[case]`：`xyz/features/cloud/mask/grid/query_indices/query_weights`，1024点/seed2026、R64、原25。其scaler已确认与训练config相同；按 [decode](../geobc/benchmark.py#L50) 用FP64 `u=encoded[:,:3]*std[:3]+mean[:3]`，仅解码位移，不对u套asinh或评估stress。保持原FP32/TF32、no-autocast。

**初值接口。** 验证缓存xyz/节点顺序与物理canonical/Grid IDs身份一致，不用ID−1/nearest/role flags。取 `z0=u_FNO_mm[free_node_ids]`（FP64，M×3）；真实SPC/RBE2推导固定平移零，孤立loaded RBE3 reference仅Hbar恢复输出，不进A。初始化`r0=b−Az0,p0=M⁻¹r0`，不用旧零初值p/r；同 [MatrixFreeElastic](../geobc/elastic_pilot.py#L178)、[_restore_full](../geobc/elastic_pilot.py#L410)、K/geometry/Cholesky与cache身份。raw FNO与约束可行化初值分开报告，不能把后者当原FNO精度。

**不能直接调用的入口。** ① [selection_data.load_case](../geobc/solver_selection_data.py#L105) 会立即打开target/physical，因此需要新窄input-only loader，显式禁止这两个成员；不能因模型不消费标签便称零GT。② [elastic_crosscad.solve_case](../geobc/elastic_crosscad.py#L356) 及contract冻结了zero-start；应新隔离hybrid入口，不修改已完成源/config/checkpoint或让旧zero结果冒充可变初值。③已完成物理operator/8guard可严格复用而不重建，但成本不能因此消失；新缓存生成/host-to-device/权重加载的actual和继承成本必须分列。

**共同终点与总成本。** 两条PCG同真实 `||b−Az||/||b||≤1e−8`、10000步/solve300秒及验证间隔；纯FNO不保证此门，不能称三者同精度。冷账本包含神经几何/grid准备、checkpoint/scaler加载、推理/查询/解码、已实测FEM input/operator/8guards成本、warm solve/真实残差/endpoint验证与写盘；主门用单请求完整冷账本，五case模型加载均摊仅secondary。复用共同setup时列出新实测incremental和继承cold成本，称**accounted cold-start cost**，cache-hit延迟另列。历史zero PCG solve绝不计入warm预算；不重复守卫，也不把跳过setup称0cold。旧NN cache没有逐CAD冷frontend时价，未来需计时target-independent同schema准备或明确该项缺失、暂不裁定冷成本收益门。

**收益门与GT0。** 五CAD都取得共同真收敛终点；至少4/5 hybrid完整成本降低≥20%，建议任一CAD总成本比≤1.10为严重退化上限（待root预观察冻结）。列K调用、真残差/平衡、峰值，不能靠少几步或1–2%差就延续研究。小试GT读0：warm终点仅与冻结零初值`physical_u`作全节点mm差异/相对差；这是数值参考一致性，绝非GT误差。旧physical指标只属于旧端点，不转给FNO/warm；FNO若无当前case既有端点误差则null，仅共同DEV旧评估作上下文。模型不兼容/精度差、FNO非finite、关键case不收敛或成本不过门则结束，不救seed/宽度/损失、不按GT选初值。

这检验的是已有学习预测能否作为传统数值求解的有用初值，不是新神经泛化或本构学习结论；也不提供signed scalar stress、OptiStruct精确重放或工程合格证据。当前physics结果完成后才决定是否正式冻结。
