# GeoBC：固定 TRAIN 连续场与运动学能力诊断

仅检验已锁 FNO/MLP 两 seed 的四个 `endpoint_primary`；路径、step、stat 继承 `notes/geobc_solver_final_eval_plan.json`。不再选 checkpoint、不使用 final 调参、不训练。目标是区分节点拟合、共同重建与 FNO 网格原生查询的能力；本页为下一次独立工具运行的设计。

固定 `101_479/ver`，直接复用 `notes/geobc_continuous_geometry_result.json` 的 32 原始 cell rows / 8 shared faces。320 个十节点局部出现、128 个既有四点正 Gauss、56 个面迹点，共 **504 logical queries**（物理点可去重）；不重跑全 CAD face/bounds/locator、域外点或 synthetic fixtures。物理映射仍是全十节点 `map_points`，梯度用 `shape_gradient` / `jacobian`；原 H5 仅 `vertices`，`data/geobc_gradient_supervision/101_479.npz` 仅 `cells10`。源身份继承已通过 receipt。

| 比较栏 | 固定输入、调用与解释 |
|---|---|
| 共同 FNO / MLP nodal full10 | 从 `data/geobc_solver_selection/geometry/101_479/{xyz,cloud,mask}.npy` 和 `cases/101_479_ver/features.npy` 读取原节点输入；每 checkpoint 一次 `encode_geometry`。MLP 对选中原节点 `forward_points`；FNO 一次 `forward_grid(grid[1,29,64,64,64], context)`，以缓存 `query_indices/query_weights` 和 `selection_data.query_grid` 回查同节点。用 manifest 的 `FieldScaler.decode` 后，`interpolate_nodal` 做同样的物理单位全十节点重建。 |
| FNO native grid query | 使用同一预测 grid，在上述 504 点以八角 stencil 查询；单列值、Gauss 应变 proxy 和成本，与共同 nodal full10 比较。网格三线性值在域内 C0；voxel face 上导数可跳，不等于值不连续。 |
| MLP native offnode | **notimplemented**：`forward_points` 有连续 xyz 参数，但原 25 features 的可信域外节点延拓尚未定义。不得把全十节点插值的角色权重称为物理 BC 或自动填补此栏。 |

FNO 原生查询唯一采用准备时已冻结的物理常数：`qcan=(qphys−bbox_min_mm)/L`，`L=184.05270385742188 mm`；常数来自既有几何 plan，不新增 bbox 拟合。新 stencil 显式要求坐标在 `[0,1]^3`，不以 clip/nearest 救失效。原生位移物理梯度为 `diag(std_u)·(∂encoded_u/∂qcan)/L`；包含 mm 尺度与 decode std。八角权重解析求导，无额外差分查询；网格面按既有 floor 一侧导数并报告次数。直接物理归一化与缓存节点 FP32 坐标/stencil 的重放差异照报，不修正；共同节点重建仍严格使用原缓存节点查询。

GT 仅未来正式运行读取 `cases/101_479_ver/physical.npy` 的选中节点（u XYZ/mm、signed stress/MPa）。值参考为 GT 节点 full10；P1/P2 为 **GT 节点派生运动学 proxy**，不是 offnode/strain GT。复用 `diagnose_quadratic_kinematics.kinematics/metrics` 的物理 affine P1、全十节点 P2、全 3×3 Frobenius 与 `E2=E1+Emid+signed_cross`，保留 cross 符号。P1/P2 各自 GT 能量分母明确列出；原生 FNO 梯度用 GT-P2 分母；真零分母返回 null。128 Gauss 任一 Jacobian 失效不得跳点后报部分结果。

最少输出：每 seed 三栏的物理值 RMS/relative error、P1/P2 strain proxy error 与能量分解、最差 cell、节点 replay/面迹 C0 的接口量、native 可用性/网格面次数、推理秒数/peak memory。重建的结构性 C0 不作学习成功证据。原 0:6 是 support-reference DOF、6 是 load membership，不自动表示节点 Dirichlet；已完成角色 splat 诊断只提供背景，不重编码、修复 BC 或推断误差因果。下一步仅一个 `tools/diagnose_solver_continuous_capability.py`，根审定后唯一正式 GPU 运行；不扩展到全 CAD、final 或新 solver。
