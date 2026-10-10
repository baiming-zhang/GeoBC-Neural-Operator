# DeepJEB v1.0 signed scalar：可证实的语义与导出缺口

2026-10-10，独立有限源研究。**作者将字段称为 signed von Mises stress；当前取得的官方证据仍不足以无歧义实现原 `{load}_stress(MPa)` 的完整节点导出。不能把标准VM、最大主应力或自行选定的符号函数替换为源标签。** 这不是“作者从未说明”的穷尽性断言，也不是新的场值实验。

本次先检索项目 `notes/references/third_party/tools` 中的源码及小元数据，复用此前有位置与链接的 primary 核实记录 [`geobc_stress_semantics.json`](geobc_stress_semantics.json)、[`geobc_signed_output_interface.md`](geobc_signed_output_interface.md)。本地找到的是本项目的下载、读取和转存代码，未找到发布者的 Field 生成/H3D→CSV 导出脚本；`tools/prepare_deepjeb.py` 只读取源列并按坐标对齐，没有构造其符号或应力张量。官方镜像固定 revision `160424ca2a5095315ff9ead8a5fc36d2667fd3c6` 的已保存 tree 元数据列出组件归档、README和许可，不能由此断言归档内部或未公开仓库不存在导出器。

| 问题 | 已证实到哪一层 | 尚不能确定的原导出选择 |
|---|---|---|
| 标量类型、单位 | 论文 Table 1/§3.3 使用 signed von Mises；已有核实短句为 “We also provide signed von Mises stress data”，并将符号关联拉压。源接口为每工况每节点单个 MPa 标量。[作者论文](https://arxiv.org/html/2406.09047v2) | 这不是完整数学函数；不能由名称证明任一恢复张量的VM幅值恰等于该CSV幅值。 |
| 求值位置 | 原版发布包含节点场与二阶四面体网格；已有控制卡记录有 `STRESS`、`GPSTRESS(H3D, ALL)`。[论文](https://arxiv.org/html/2406.09047v2)、[2022 GPSTRESS](https://2022.help.altair.com/2022/hwsolvers/os/topics/solvers/os/gpstress_gstress_io_r.htm) | CSV取GPSTRESS还是elemental STRESS；积分点、centroid、corner及midside结果如何成为该节点值。输出请求存在不等于选中了某一分支。 |
| 节点聚合 | HyperView官方节点平均说明存在不同处理顺序；corner不可用时可涉及centroid结果。[2022.2节点平均](https://2022.help.altair.com/2022.2/hwdesktop/hv/topics/panels/nodal_averaging_elemental_results_hv_r.htm)、[顺序术语](https://2025.help.altair.com/2025/hwdesktop/hwx/topics/panels/nodal_averaging_elemental_results_hv_r.htm) | 先算标量再平均还是先平均张量再算；GLOBAL/BYPROP、跨组件规则、权重及跳变处理。当前体积加权节点proxy不能冒充原软件规则。 |
| 中边节点 | 官方描述一项条件性选项：未提供中边结果时，使用corner data可由角点tensor恢复中边，再算派生量。[2022.2中边结果](https://2022.help.altair.com/2022.2/hwdesktop/hv/topics/panels/contour_panel_mid-side_node_elem_results_r.htm) | 原导出是否启用它，还是取另一原生中边/节点结果。不能直接认定与全10节点P2导数相同。 |
| 符号 | `.res` solid规范给出 `sign(P1+P2+P3) × VM`；FATPARM语境采用绝对值最大主应力的符号。[.res规范](https://help.altair.com/hwsolvers/os/topics/solvers/os/stress_results_written_in_hm_bulk_res_r.htm)、[FATPARM](https://help.altair.com/2023.1/hwsolvers/os/topics/solvers/os/fatparm_bulk_r.htm) | 它们属于不同输出/分析接口。原deck请求H3D，不能移植其中一条为DeepJEB定义，也不能按DEV误差选择符号规则。零值/相等主应力的分支未知。 |
| 坐标系与datatype | H3D官方说明区分tensor、派生和DIRECT结果；HyperView可由tensor计算派生量。[2022 H3D规范](https://2022.help.altair.com/2022/hwsolvers/os/topics/solvers/os/stress_results_written_in_hv_h3d_r.htm) | 原datatype/component、张量结果坐标系和导出会话设置未绑定。坐标是全局GRID并不证明应力结果采用同一坐标系。 |

**已证实**：作者意图为signed-VM节点标量；原字段及signed标签应原样保留，官方原版出处明确。[官方数据页](https://dataset.narnia.ai/deepjeb.html)、[固定版本README](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/blob/160424ca2a5095315ff9ead8a5fc36d2667fd3c6/README.md)。**推断**：H3D/GPSTRESS/后处理恢复很可能参与生成，但实际分支未证实。**缺失**：原结果datatype/component、位置、平均顺序/权重、corner/midside规则、坐标系、符号与零值规则，以及能把这些选择绑定到原CSV的导出脚本/会话或作者确认。仅凭通用Nastran/OptiStruct后处理源码也不能补出这项选择。

最小有用的新信息是原H3D及对应导出脚本/会话，或作者提供唯一完整导出契约；这不是继续枚举符号heuristic、从signed scalar反推strain，或用当前幅值/sign-error诊断确认源公式。本note不授权执行源复现或打开S。物理FE的高精度U和native-P2能力可以独立成立，尚不能据此声明已复现原signed-S输出。

访问限制：本次官方web工具查询/打开连接失败；一次只读作者HTML的回退亦未取得页面，因此上表沿用已有primary核实记录，并明确不是本轮新取得原始导出代码。当前五CAD原stress值、科学CSV/H5字段、NPZ、GT、权重、训练与GPU访问均为0；未修改现有20pins、实验或U-only账本。
