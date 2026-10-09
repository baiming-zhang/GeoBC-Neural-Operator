# 原版 DeepJEB signed stress：源输出接口边界

2026-10-10；一次有限 primary 查询，非实验授权。**作者意图已知，实际 H3D→CSV 选择与节点恢复仍未闭合；目前物理路线是高精度 U 参考，不能称完整源 signed-S 求解器。**

已有证据复用 `geobc_stress_semantics.json`、`deepjeb_original_sources.md`、`elastic_deck_controls_primary.md`。论文 Table 1/§3.3 把字段称为 signed von Mises，用符号区分拉压；未给导出算法。官方数据页/原版镜像列出 CSV/H5/FEM，所查公开入口未提供原导出器或 H3D 结果文件的对应契约；这不是对未公开工具的穷尽性断言。[作者论文](https://arxiv.org/html/2406.09047v2)、[官方数据页](https://dataset.narnia.ai/deepjeb.html)、[官方 v1.0 镜像](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/blob/160424ca2a5095315ff9ead8a5fc36d2667fd3c6/README.md)。

现 `101_428/input/manifest.json` 已记录 H3D/GPSTRESS/STRESS，但忽略输出卡的解析记录丢失括号参数；仅补读固定 TRAIN `data/raw/SolverDeck/101_428.fem` **前25行控制文本**，确认 Inspire 22.2 Build13568、MPA(mm t N s)，第12行 `FORMAT H3D`、16行 `GPSTRESS(H3D, ALL)=ALL`、18行 `STRAIN(H3D, ALL)=ALL`、19行 `STRESS=ALL`。该段没有显式 CORNER/GAUSS 选择；不能由此推断软件未计算其他节点结果。

| 新核实的官方接口事实 | 对 DeepJEB 能证明到哪一层 |
|---|---|
| OptiStruct `.res` solid 的 signed VM 明确为 `sign(P1+P2+P3) × VM`。[.res 规范](https://help.altair.com/hwsolvers/os/topics/solvers/os/stress_results_written_in_hm_bulk_res_r.htm) | 这是该输出格式定义；实际 deck 请求 H3D，不能自动转移为 CSV 的符号定义。 |
| FATPARM 的 signed VM 符号取 absolute-max-principal 的符号。[官方疲劳规范](https://help.altair.com/2023.1/hwsolvers/os/topics/solvers/os/fatparm_bulk_r.htm) | 属于另一接口语境；不代表该静态 CSV 使用它。二者不能按 DEV 表现择优。 |
| H3D solid ALL/TENSOR 的派生量由 HyperView 计算；DIRECT 可含 signed VM。[2022 H3D 规范](https://2022.help.altair.com/2022/hwsolvers/os/topics/solvers/os/stress_results_written_in_hv_h3d_r.htm) | 通用 ALL 请求不能确定最后选了 tensor 派生、DIRECT scalar，还是 GPSTRESS。 |
| GPSTRESS 支持 grid-point 输出，默认 BYPROP 同时输出全局及按 property 的平均；ALL 包含所有 stress 结果。[2022 GPSTRESS](https://2022.help.altair.com/2022/hwsolvers/os/topics/solvers/os/gpstress_gstress_io_r.htm) | 请求存在不等于 CSV 采用某一个 GPSTRESS 分支；该页没有给出足以复刻当前节点恢复的完整公式。 |
| HyperView 节点平均使用相邻单元 corner 结果，无 corner 时使用 centroid；Simple/Advanced 等选择不同。[2022.2 节点平均](https://2022.help.altair.com/2022.2/hwdesktop/hv/topics/panels/nodal_averaging_elemental_results_hv_r.htm)；新版分别称 compute-then-average / average-then-compute。[术语及顺序](https://2025.help.altair.com/2025/hwdesktop/hwx/topics/panels/nodal_averaging_elemental_results_hv_r.htm) | 标量先算再平均与张量先平均再算不等价；原会话/脚本选择尚未知。 |
| 未提供中边结果时，启用 Use Corner Data / Show midside 后，OptiStruct/Nastran 的中边 tensor由角点tensor平均，之后计算 invariant。[2022.2 中边恢复](https://2022.help.altair.com/2022.2/hwdesktop/hv/topics/panels/contour_panel_mid-side_node_elem_results_r.htm) | 这是条件性后处理规则；不能宣称 DeepJEB 已采用，也不能把当前四Q→体积加权节点 proxy 当作同一恢复。 |

真正缺失的是：原 H3D result datatype/component selector、GPSTRESS vs elemental STRESS、GLOBAL/BYPROP、corner/centroid/midside开关、平均顺序/跨组件规则、结果坐标系、signed 标量函数及零/相等主应力处理，以及导出器/软件版本。CSV `{load}_stress(MPa)` 本身不包含这些信息。Inspire 2022 文档确认拉压显示来自 signed VM，但没有补齐其数值定义。[Inspire 结果说明](https://2022.help.altair.com/2022/inspire/en_us/topics/inspire/structure/analysis_results_view_t.htm)。

**最小可解释基准有条件可行，当前尚不具备完整源契约。** 只固定 TRAIN `101_428/ver`：先取得该案例原 H3D＋原导出脚本/会话，或作者确认的准确 selector/recovery 与可复现同版本官方结果。先冻结唯一导出路线、单位、GRID对应和容差（仅覆盖已声明 CSV/FP32 导出舍入），随后一次比对其导出 scalar 与原始、未改动 signed CSV；按 corner/midside 和全N报告 signed L2、MPa RMS/max、符号不一致及热点。只有此接口对应通过，才将近似物理 tensor 按**该已确定路线**恢复并比较，区分导出匹配与近似模型误差；不训练、不校准、不枚举符号/平均方案。

若原导出器不可得，root仍可另行预声明一次 `101_428/ver` **官方接口兼容诊断**：每条路线注明软件/格式/工况、tensor来源及完整恢复前提，只执行条件成立的接口，不把 `.res` 或 FATPARM 规则冒充当前 H3D 静态路径。原signed标签只作诊断、不修改；不触DEV或新test，不按结果择优认定源公式。低误差只说明该proxy与数据兼容，不能确认原导出器；高误差也混合了近似解算、tensor求值、恢复和标量化差异，不能唯一归因于源operator。该诊断与上段“原H3D/原exporter复现CSV”的源接口确认是两个不同证据等级，当前均未授权执行。

源确认停止条件：缺原结果/可信导出说明，或唯一源路线不能在声明的导出舍入误差内复现 CSV，就保留 unknown。重新运行原 deck 不能补足未知原导出选择。约10%的 q 幅值一致性、9.77%的表示范围下界及模型 sign-error 都不定义源 S。`DEBUG LGSHPSNS`、积分/预处理未精确重放的限制仍在；物理求解器仍定位为U参考。

本次读取：已有小JSON/notes、单一TRAIN控制文本25行及官方网页；科学数组/GT/模型/权重/DEV/final/sealed/GPU访问均0，未修改任何科学源码或计划。
