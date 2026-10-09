# 下一独立 solver 周期可行性

`metadata_only / proposal_only`，2026-10-10。当前 final 只作本轮报告；此页不授权立即实验。仅查源码、清单/stat、installed dist-info；未读标签数组/权重、下载、安装或运行 GPU，未改正式计划。

**新 5 CAD：可获取，尚未形成独立盲测集。** [split](</C:/Users/ASUS/Desktop/GeoBC Neural Operator/notes/geobc_split.json>) 与 `data/raw/{FieldMesh,Field,SolverDeck}` 各 50 CAD 完全重合。`data/raw_exact` 另有 `506_25/506_533`，三组件合计 155,694,435 bytes，但 [旧 prepared](</C:/Users/ASUS/Desktop/GeoBC Neural Operator/data/prepared_exact_multifield/manifest.json>) 已使用，不能充当新盲测。[官方本地索引](</C:/Users/ASUS/Desktop/GeoBC Neural Operator/notes/original_deepjeb_drive_index.json>) 三组件共同 2,138 CAD，排除已见 52 ID 后余 **2,086**。示例候选 **120_122/120_131/120_142/120_153/120_284** 均未在本地 raw 出现；仅确认 ID 独立，几何近重复、代表性及 tar 顺序未验收，同前缀不等于统计独立。

精确来源：官方 revision `160424ca2a5095315ff9ead8a5fc36d2667fd3c6` 的 [FieldMesh.tar.gz](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/resolve/160424ca2a5095315ff9ead8a5fc36d2667fd3c6/FieldMesh.tar.gz)、[Field.tar.gz](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/resolve/160424ca2a5095315ff9ead8a5fc36d2667fd3c6/Field.tar.gz)、[SolverDeck.tar.gz](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB/resolve/160424ca2a5095315ff9ead8a5fc36d2667fd3c6/SolverDeck.tar.gz)，成员分别为 `FieldMesh/<ID>.h5`、`Field/<ID>.csv`、`SolverDeck/<ID>.fem`；逐文件 Drive URL 见索引 `component.entries[name]`。远端当前可访问性未验证。

| 可复用本地 gzip 前缀（完整路径） | stat bytes |
|---|---:|
| `data/raw/.download_cache/160424ca2a50_FieldMesh.tar.gz.part` | 1,277,224,960 |
| `data/raw/.download_cache/160424ca2a50_Field.tar.gz.part` | 1,420,544,000 |
| `data/raw/.download_cache/160424ca2a50_SolverDeck.tar.gz.part` | 402,841,600 |

这是 **3.10 GB 前缀，非完整包**，未证实多出完整成员。[下载器](</C:/Users/ASUS/Desktop/GeoBC Neural Operator/tools/fetch_deepjeb.py>) 支持 `--cases`/HTTP Range，但 gzip 要回扫，且会更新原 provenance。建议另行授权后先预登记新 5 ID，在 `data/raw_next_cycle` 独立记录，复制缓存或小改为只读旧前缀/独立追加；保留原数据，标签封存，未来沿原 CSV GRID→H5 vertices 对齐。按历史 50 CAD 均值，5例粗算 **0.31 GB 新压缩尾部+0.68 GB 解包**；复制前缀方案额外磁盘约 **4.09 GB**。历史50例三组件约15分钟，均摊5例约90秒，仅算术参考；成员位置/大小、回扫及网络未测，不能当 ETA。Drive 逐文件路线免回扫，但索引无 size/SHA，[exact 脚本](</C:/Users/ASUS/Desktop/GeoBC Neural Operator/tools/fetch_exact_deepjeb.py>) 的清单生成固定旧六文件，需新的获取 receipt。

**GINO：规则-grid CSR 是最小候选，后端尚未验收。** [现有设计](</C:/Users/ASUS/Desktop/GeoBC Neural Operator/notes/solver_gino_candidate_design.md>) 的 R64/radius0.033/官方 pin 尚未取得运行证据。WSL `/home/research/.venvs/geobc` dist-info 确认 Torch **2.14.1**、Triton **3.8.0**、NumPy **2.5.3**、SciPy **1.18.1**；未见 neuraloperator/Open3D/torch-cluster/torch-scatter/CuPy metadata。Torch 的 CUDA13.0.3 依赖声明不能证明运行兼容或性能。

建议 Torch GPU 整数分箱/排序/scan，Triton 可做分块 count/fill：按 `h=1/63` 枚举点的局部格点盒、保守外扩一格，用统一精度 **d²≤r²** 精筛，两遍 count→prefix-sum→fill。一次 edge 集合建 grid 行 CSR，再转置为 node 行 CSR，保留原 ID/稳定排序/dist²；新 query 局部枚举固定 grid。禁止完整 `torch.cdist(N,262144)` 或邻居截断；官方积分核/segmented reduction/weights 适配仍待验证，不静默 fallback。接口为 `encode_case(inputs)->state`、`query(state,xyz)->signed4`，保留整 case 状态及显式 mollifier 距离权重。

存储为 **O(N+R64+E)**，edge 核激活/反传仍可能主导显存；原25角色也仍可能混合。覆盖、空邻域、截止连续性与实际耗时/peak均未知。下一独立周期若获授权，先只做既定 TRAIN 单 CAD 全 N 输入型可行性验收，再决定架构实验；新5CAD始终不参与开发选择。
