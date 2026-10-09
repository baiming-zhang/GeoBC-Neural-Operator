# GeoBC solver selection: source-only adapter proposal

2026-10-09。状态：proposal only；本次只读源码、slides 文本、配置 JSON 和官方 primary sources，未读科学数组、实例化模型、初始化 CUDA 或运行训练。先比较适合几何/BC 条件场的 solver；不继续 Hash 单点优化。旧冻结模块保持不变。

建议先做 **MLP、Transolver、Transolver++、Transolver-3、FNO** 五类非 Hash solver 的共同接口。Hash 只保留已有参考结果，不作为此次调参中心。历史 slides 使用不同数据、预算与输出处理，不能据旧误差排序当前 CSV GRID→H5 vertices 对齐数据上的 solver。

## 已有实现与原配置

| Solver | 可复用源/工厂 | 现有 preset；规模依据 | 全 CAD 与主要成本 |
|---|---|---|---|
| MLP | `geobc.models.build_model(encoder="mlp", backend="torch")`；`geobc.benchmark.model_factory` 支持多输出末层 | slides：4 hidden×256/ReLU、xyz-only、uXYZ+signed stress，199,428 params；若共同输入28、输出3，按层公式为205,571 params | 点独立，分块前向/累积全节点损失不会改变模型；对几何的全局关系表达较弱，是低成本基线 |
| Transolver | `geobc.transformers.build_transformer("transolver", input_dim, output_dim)` | pinned constructor defaults：5×256、8 heads、32 slices、ratio1；现28→4 metadata为1,818,572 params。slides未给原始模型配置 | 全节点共同参与每层物理状态；whole-case forward，checkpoint减少训练activation，不能独立小块推理 |
| Transolver++ | 同工厂 `"transolver_pp"` | slides明确16×256；其余8 heads/32 slices/ratio1为官方defaults，非完整slides恢复；28→4为4,459,668 params | whole-case；现adapter world_size=1 group保留官方all_reduce。官方Gumbel在eval仍采样，必须单独固定/记录evaluation RNG策略 |
| Transolver-3 | 同工厂 `"ts3"` | 官方volume preset16×256、8 heads、64 slices、ratio2；28→4为7,612,036 params。slides只给28→4、训练预算，深度等不可恢复 | 现adapter用官方每层两pass `forward_chunks` 汇总全部原节点状态，再分块deslice/MLP；默认chunk32768保持全CAD上下文 |
| FNO3d | `geobc.fno.FNO3d(in_channels, out_channels, width, modes, layers)` + `query_grid` | slides64³：width16/modes12/4blocks；现18→4为14,159,860 params。192³/modes16/stress-only约33.56M。head128/padding0属当前明确替代，原值缺失 | 整cube FFT，输出后统一查询全部原节点；64³起步比192³低27倍体素数，不能靠点query chunk降低cube前向成本 |

上述param数来自现有config JSON，仅说明已有配置规模；新input/output配置须在首次正式实例化时记录实际参数数，不宣称跨架构初参数相等。

`third_party/transformer_sources.json` 的全部已列源SHA本次stdlib核对PASS：原Transolver commit `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`，++ `d5a23bc734a0ebac56384cf72049a26af9673452`，TS3 `ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37`。这些是本地固定实现，不自动追最新仓库。[官方Transolver](https://github.com/thuml/Transolver)、[官方++固定源码](https://github.com/thuml/Transolver_plus/blob/d5a23bc734a0ebac56384cf72049a26af9673452/models/Transolver_plus.py)、[官方TS3固定源码](https://github.com/thuml/Transolver-3/blob/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37/models/Transolver_chunk_opt_matrix_mul.py)。

## 最小共同输入与 API

首轮建议单一 `features25` track：`canonical_xyz[N,3] + original BC/material[N,25]`，显式保留force/moment输入，预测直接signed `uXYZ`；不用九basis、Hash、case ID、GT输入、teacher/residual或学得的历史context。完整mesh坐标本身就是Transolver-family几何输入。可将现有target-free local13/Euclidean5作为后续**所有solver同用**的另一feature track，不能与首轮模型结果混成架构排名。

```text
SolverSpec(method, preset, input_schema, output_channels, precision, source_pins)
CaseInputs(canonical_xyz, physical_xyz_mm, node_features,
           original_node_ids, cells10, case_key_for_provenance_only,
           coordinate_normalization, optional_geometry_grid)
build_solver(spec) -> adapter
adapter.forward_case(inputs) -> Prediction[N,C]  # common training/evaluation path
adapter.predict_nodes(inputs, rows=None) -> Prediction
adapter.query(context, physical_xyz, declared_query_features) -> QueryResult  # optional
adapter.capabilities -> {full_case_context, independent_node_chunks,
                        cached_state_query, nodal_extension, eval_stochastic}
```

`rows`只能决定输出gather/损失节点：对Transolver/++/TS3不能先删掉未查询节点再算context。Pointwise MLP允许直接分块；TS3允许官方global chunk path；original/++必须whole case。不同节点数CAD首轮逐case处理，避免用未mask的padding节点改变官方聚合。训练时任何learned-state/prediction cache随每次参数更新失效。

`forward_case`返回统一channel order、dtype与coordinate/source identity；target scaling和loss由公共runner处理，不在adapter内各自fit。首轮u-only可沿用现有冻结common-zero-center scaler/case weights，或root另冻统一train-only recipe；signed stress若纳入则所有模型同一单独transform，不能模型各自选linear/asinh。原benchmark runner的scaler、best选择与旧默认数据路径不直接继承。

FNO需同一输入信息的grid适配：建议occupancy + signed quadratic interpolation of `[canonical_xyz3, features25]`，即29 channels，outside input为0。其FNO cube坐标与原canonical normalization分开保存；query mapping从physical original vertices建立，输出严格恢复原node顺序。现 `prepare_grids.py` 的18-channel替代schema只保留部分features且clip前7标志；不宜直接当共同输入，需新隔离input-only adapter，full10 numeric interpolation不clip、不称物理binary BC。

现 `query_grid` 对无occupied corner的原节点返回0。公共adapter必须暴露supported counts/unsupported IDs/status，全部原节点仍纳入结果；不能静默把0当合法场、删点或nearest-copy。旧grid Newton helper不能升级宣称全CAD inverse/Jacobian/ambiguity已证明；Bezier/shape等正确原语可复用，geometry状态沿已冻结明确协议处理。

## 连续接口的准确声明

- MLP：native pointwise query，但需要统一声明offnode BC/geometry descriptor extension；不能直接赋予任意物理点nodal flags。
- Transolver/++：现工厂为discrete whole-mesh operator。可对固定全节点prediction做full10 TETRA10 signed interpolation，声明为nodal field extension；加入query节点重新聚合会改变context，不能称相同固定continuous field。
- TS3：已有 `third_party/Transolver-3/models/Transolver_chunk_opt_matrix_mul_amortize.py`，提供官方 `PhysicalStateCachingModel` + `FullMeshDecodingModel` 的fixed-state query路径。现manifest没有pin此文件，后续可独立新source manifest/pin，再验证原node replay和query-batch不改变state；当前不将其标为已实现/验证能力。[官方amortize源码](https://github.com/thuml/Transolver-3/blob/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37/models/Transolver_chunk_opt_matrix_mul_amortize.py)。
- FNO：grid output + declared trilinear query；occupied-support状态和grid表示误差必须单列，不能称exact curved FEM field。[FNO官方固定实现](https://github.com/neuraloperator/physics_informed/blob/3b6bc307c63c64057d0496753bf1af5f44ac5108/models/fourier3d.py)。

连续能力作为单独capability与后续验证维度，不阻止首轮先比较原nodes上的geometry/BC-conditioned solver。既有连续locator只验证selected32，并无全CAD arbitrary-point inverse证明；本次不扩大该科学结论。

## 广筛与成本建议

最小共同protocol是每个update看到完整同一CAD，并在全部原nodes计算相同nodal loss；MLP可按全node分块累积一次optimizer update，FNO整cube算一次再query全部nodes，Transolver-family保持全CAD context。固定短budget、同seed/source/split/loss/scaler、固定endpoint，不做architecture-specific sweep；报告等update、实际fullnode exposure与GPU seconds，不能称等计算量或匹配初参数。后续shared train/validation才回答geometry泛化；单TRAIN拟合不能直接选出泛化winner。

对当前N=179316，float32一份N×256 activation约175MiB；original/++一份N×8×32 slice weights也约175MiB。TS3 chunk32768×8×64约64MiB/临时slice tensor，但16层训练checkpoint仍有额外activation/重算成本，非固定64MiB总显存。FNO width16单份64³ activation为16MiB，192³为432MiB；FFT/workspace/gradient/optimizer另计。上述仅shape算术，未作当前GPU profile或ETA承诺。

建议先保留每家现有可解释preset作五类广筛，不以同width代表同capacity：MLP4×256，original5×256，++16×256，TS3volume16×256，FNO64³/width16/modes12/4blocks。192³与其他改配等待首轮source/schema/数值/成本结果再决定。唯一公共接口经root审定后，才实现新的isolated adapters/runner；本note不授权任何数组读取、profile或训练。

来源定位：`geobc/models.py`、`geobc/benchmark.py`、`geobc/transformers.py`、`geobc/fno.py`、`tools/prepare_grids.py`、`tools/run_slides_suite.py`、`references/slides_text.json`、`third_party/transformer_sources.json`、现有slides runs的config metadata。无scientific binary、GT、weights或model实例访问。
