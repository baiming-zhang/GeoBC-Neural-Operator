# DeepJEB 原版来源与实际文件核对

核实日期：2026-10-08（Asia/Shanghai）。本工作只使用 **DeepJEB original v1.0**，不是 DeepJEB++ / DeepJEB-PP。

当前全方法数据：官方 Drive 中已找回原稿两个 CAD 的 H5/CSV/FEM，文件清单和 SHA256 见 `references/deepjeb_exact_files.json`；五样本和50样本的实际选择见各 prepared manifest 与 `notes/all_methods_plan.json`。训练标签必须用 CSV GRID 坐标对齐到 H5 vertices，不能直接采用 H5 nodal_variables 的数组顺序。下面首五个案例的取得过程是初始下载记录。

- [作者论文（arXiv v2）](https://arxiv.org/html/2406.09047v2)：2,138 个设计；二阶四面体网格；几何、4 个静态工况的节点场及模态场。
- [KAIST 官方原版镜像](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB)：固定 revision `160424ca2a5095315ff9ead8a5fc36d2667fd3c6`。
- [官方镜像 README](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/blob/main/README.md)：镜像 original v1.0，按组件 `.tar.gz` 分发。
- [实际许可文本](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/blob/main/LICENSE.txt)：ODC-By v1.0；本地副本 `data/raw/LICENSE.txt`。
- 逐案例文件已通过 [官方 DeepJEB 数据页](https://dataset.narnia.ai/deepjeb.html) 指向的原版 Drive 根目录取得。`tools/fetch_exact_deepjeb.py` 按观察到的官方文件 ID 和 SHA256 验证/下载原稿两个案例；镜像压缩包用于其余案例。

官方组件：`BRep` STEP；`SurfaceMesh` STL；`VolumeMesh` VTK；`FieldMesh` H5；`Field` CSV；`Scalar` CSV；`SolverDeck` FEM；`Metadata` JSON；`Image` PNG。完整 `FieldMesh.tar.gz` 为 53,817,481,481 bytes，`Field.tar.gz` 为 59,811,001,592 bytes。

## 初始最小批量下载记录

以 urllib HTTP 流 + gzip tar 流抽取最前 5 个完整 H5：`10_30`, `10_421`, `10_460`, `10_492`, `10_58`。H5 包含全部 4 个静态工况。相同 5 个原始 FEM deck、Scalar、Metadata 与许可文本也已取得。

H5 用时 47.7 s（含小组件），实际压缩前缀 121,896,960 bytes；FEM 用时 15.5 s，压缩前缀 38,236,160 bytes。未下载完整大型 archive。具体输出 SHA256、commit 与可恢复状态保存在 `data/raw/deepjeb_provenance.json`。5 个案例与幻灯片示例案例不同。

`tools/fetch_deepjeb.py` 支持 `--count 5` 或 `--cases ...`，缓存压缩前缀；中断后从缓存解压并通过 HTTP Range 追加不足的压缩数据。gzip 的后部任意案例必须扫过此前数据，所以首 5 个案例更适合目前 overfit 优先任务。

## 标签与语义

官方 Field CSV 首文件 `Field/10_30.csv` 的真实字段：

`nodeID`, `coord_x(mm)`, `coord_y(mm)`, `coord_z(mm)`；每个 `ver/hor/dia/tor` 工况有 `{load}_x_disp(mm)`, `{load}_y_disp(mm)`, `{load}_z_disp(mm)`, `{load}_resultant_disp(mm)`, `{load}_stress(MPa)`；另有两个模态形状。

应力目标保持原始 **source signed stress**，单位 MPa，允许负值，不取绝对值。Scalar 的 `10_30/ver` 最大值 1566.630981 MPa、最小值 -883.2739258 MPa。论文 Table 1 及正文使用术语 signed von Mises stress，原文短句："We also provide signed von Mises stress data"。源文件名称仅为 `stress(MPa)`，不含应力张量；训练与图表应称 source signed stress，不能由此重建另一个张量目标。

## 必须以 FEM deck 为准的载荷与 BC

论文文段将 horizontal / torsion 方向描述为 +x / -z；**实际原始 FEM deck 是 -x / +z**，因此训练使用 deck 的真实值，而不是论文的四舍五入值或其他数据版本的说明。

实际 `10_30.fem` 的四个工况（全局坐标）：

| Load | Force (N) | Moment (N mm) |
| --- | --- | --- |
| ver | (0, 0, 35585.77) | (0, 0, 0) |
| hor | (-37809.9, 0, 0) | (0, 0, 0) |
| dia | (-28276.21809372, 0, 31403.902135776) | (0, 0, 0) |
| tor | (0, 0, 0) | (0, 0, 564924.2) |

FEM header 一致单位为 `MPA (mm t N s)`；MAT1 为 E=113800 MPa、nu=0.342、rho=4.47e-9 t/mm³（4.47e-6 kg/mm³）。每个案例由 4 个 RBE2 支撑接口 + 1 个 RBE3 加载接口定义。SPC 对 4 个 RBE2 参考节点施加 DOF `123456`=0，加载作用于 RBE3 参考节点；成员节点 ID 由卡片及续行精确取得。

`tools/parse_deepjeb_deck.py` 返回全 GRID 的 nodeID → xyz 映射、4 个支撑组、加载组、实际 force/moment 与材料；不要假设 H5 顶点索引 = nodeID - 1。H5 顶点需要按坐标映射到原始 GRID。5 个 FEM 案例支撑成员数均 1760；加载成员数依次 811、814、814、802、799。JSON 摘要在 `data/raw/BoundaryConditions/`。
