# GeoBC：原版 DeepJEB 复现与求解器筛选

项目研究如何根据几何、材料、载荷与约束预测原节点及连续空间的位移、signed应力。**最终solver尚未选定；Hash专项微调已停止。**

最新统一表示／容量对照已完成：同R64输入、native25查询头、约4.53M参数、同TRAIN120和4800更新，FNO的新60例U／S为27.9%／56.1%，Transolver++为41.4%／86.4%。FNO领先的证据不再只来自旧比较中容量与表示的差异；但这是当前固定配方的开发比较，不代表架构上限或最终测试。训练UTC耗时分别26.97／144.36分钟，mono轴分别24.86／132.36分钟；两轴如实分别记录。读出接口首次出现空TRAIN键的技术错误，预测／GT均为0，修正调用端后沿用原UTC截止，保留第一次失败及跨WSL启动的真实时间分段。见[同配置简报](results/geobc_common_grid_comparison/BRIEF.md)。

随后固定单病例诊断已完成：FNO9在既有TRAIN病例10_58/tor上，共享训练该例U／S为41.5%／50.9%，fresh独立40更新为96.2%／99.7%，独立4800更新降为3.9%／11.5%，训练UTC2.57分钟／mono2.39分钟。这证明同一模型在该例上可以拟合得明显更好，大误差包含共享训练拟合不足；曝光与优化历史不同，不能把差异全归因于一个因素，也未达到Hash另一个单例的亚1%精度。后续先处理拟合与训练目标，再扩大数据；见[三行诊断](results/geobc_single_case_fit/BRIEF.md)。

前瞻100／20／20划分中，新增TRAIN60×三载荷的180例目标已完成CPU准备及退出验收，UTC84.70秒／mono78.79秒；原CSV GRID对齐H5 vertices，继承旧TRAIN40-only scaler，DEV／FINAL标签读取均为0。新80个CAD的几何／25特征／R64输入尚未准备，新100CAD训练尚未启动；这是target-only阶段。见[数据准备简报](results/geobc_prospective_train_preparation/BRIEF.md)。当前状态以下列真实报告及运行收据为准。

下一阶段已在北京时间05:36:22启动：从两臂各自完整4800检查点保留Adam／RNG／病例顺序，继续到12000／24000更新，即每例100／200次曝光；两个求解器都比较，不按FNO预选赢家。全部TRAIN输入与encoded目标按文件路径去重常驻GPU，实际560个文件／6.83GiB，载入22.66秒；首病例张量与原加载器逐项完全一致，FNO真实全节点梯度与显存门通过。++的实际资源门将在其开始时检查。保留原损失、输入、模型与scaler；开发60读出在训练全部结束后独立执行。见[曝光计划](notes/geobc_exposure_extension_plan.json)。

原版DeepJEB的30项可恢复slides实验已完成，随后探索十类学习结构及物理求解路线。全部阶段证据见[一分钟结果](results/SOLVER_BRIEF.md)，详细记录见[探索报告](results/geobc_exploration/REPORT.md)。

## 最新20个新CAD、三载荷结果（2026-10-11）

**现阶段优先FNO；Transolver系列保留++作为对照，统一表示／容量对照已完成，最终选型仍待训练拟合及前瞻测试。** 九候选固定540项提交后共享读20个原CSV；480份有效预测，缺失++24000保留60项失败。包含启动修复共17.91分钟UTC，实际运行的monotonic轴另记9.38分钟。

| 方法 | 位移误差 | 应力误差 | 训练分钟 | 60例模型推理秒 |
|---|---:|---:|---:|---:|
| FNO grid25查询头 | 27.1% | 53.4% | 25.20 | 2.047 |
| 最佳Transolver++ | 39.0% | 79.4% | 68.55 | 10.750 |
| 旧FNO | 26.8% | 55.0% | 20.11 | 2.232 |

FNO查询头相对++降低位移11.9、应力26.1个百分点；每个CAD三载荷平均的位移／应力／热点均20／20例更好，三种载荷的均值也都领先。相对旧FNO仅降低应力1.6个百分点、位移增加0.3个百分点，暂不为这个小差距继续调参。模型推理计时不包含初始化、输入读取和结果保存；完整阶段成本另列。应力误差仍超过50%，这些可用官方TRAIN池新CAD是开发比较，不能宣布最终测试或架构上限。见[精简结果](results/geobc_available_cad_pilot/BRIEF.md)与[选择结论](notes/geobc_available_pilot_actual_result_decision.json)。

实际连续查询检查已完成：3个CAD×三载荷，原三模型诊断25.73秒，独立FP32查询头对照9.60秒。旧FNO查询分块不影响结果，物理梯度AD／有限差分最大相差0.98%；新查询头TF32下相差471%，只改FP32头后降到0.66%，分块差也从6.56e-4降到4.77e-7。++换推理种子后编码场相差6.8%–9.0%，直接改变节点集合也会改变全局上下文。这验证函数数值一致性，不是独立离节点FEM精度。见[查询简报](results/geobc_field_queries/BRIEF.md)。

共同R64／容量对照及固定单例诊断已完成，结果见本文开头。资源探测中的完整TRAIN病例显存峰值为FNO0.80、++9.32GiB；正式4800对照并未用旧20例进行小幅超参搜索。

共同训练的全TRAIN120端点：FNO位移／应力19.5%／39.0%，++40.6%／85.9%。训练拟合本身仍有明显瓶颈；完整两臂与同60例开发比较见[同配置简报](results/geobc_common_grid_comparison/BRIEF.md)。

固定单病例诊断已真实运行并完成，未使用开发集训练。结果见[单病例简报](results/geobc_single_case_fit/BRIEF.md)，不能把曝光与优化历史不同的结果作为单因素因果结论。

当前查询插值的合成CPU检查已完成，用时0.833秒：实际调用链使用八角点普通三线性权重，跨六个网格面、三个距离的18项值连续性检查通过，没有发现按占据掩码把场突然归零的机制。但左右导数可以不同，因而“连续场”不等于“全局连续导数”；先前单元内AD／有限差分一致也不能证明后者。未读取真实数据、模型或标签，未初始化CUDA，不据此解释实测误差或宣称离节点精度。见[合成检查](notes/geobc_synthetic_query_continuity.json)。

物理损失的固定单例对照也已完成：同FNO9、10_58/tor、fresh初始化、Adam及4800次更新，仅改变为`0.5 × (U相对L2² + signed S相对L2²)`，训练UTC2.81分钟。位移从3.90%降为2.22%，应力从11.51%升为11.97%，热点也变差。不能据此解释应力瓶颈或替换共享训练损失；保留原encoded4 MSE进入曝光实验。实际初始物理标量完全一致，全节点梯度及显存门通过，原signed标签与scaler未改。见[损失对照](results/geobc_physical_loss_pair/BRIEF.md)。

前瞻协议为100 TRAIN／20 known DEV／20 reserved FINAL CAD：TRAIN=旧40+新增60，DEV沿用当前20，FINAL按固定ID哈希规则另留20，不替代失败ID。新增TRAIN60的三载荷目标已准备，新80个CAD的几何输入尚未准备。FINAL数值标签仍未读、不得用于训练／scaler拟合／调参。该协议来自可用官方TRAIN池，不是无偏官方TEST。见[固定划分](notes/geobc_prospective_cad_split_v1.json)。

新划分与现有输入准备数学的元数据绑定已完成，用时0.063秒，检查240份源文件stat并保留接收SHA。新增TRAIN60与FINAL20合计80个CAD／240病例都进入原准备函数的“只读输入”分支，逐CAD保留真实科学角色，关闭会解析标签的`train_new160`分支；技术上的validation别名不能当作开发集划分。继承现有TRAIN40-only scaler及R64／25特征配方，未重新拟合scaler。新数据数组、数值标签、训练与GPU执行均为零；实际准备守卫和预算尚待单独确定，不能直接启动原40／200流程。见[输入绑定](notes/geobc_prospective_target_free_input_contract.json)。

## 已有独立五CAD结果（2026-10-10）

五种方法都在相同40个CAD、120个载荷病例上完成4800次全案例更新。随后固定五个未参与本轮训练的CAD/ver，比较全部1,080,435个原节点；先提交25份预测，再共享读取五份原signed应力，历史位移缓存另行评估：

| 方法 | 五CAD位移误差 | 五CAD应力误差 | 原训练分钟 |
|---|---:|---:|---:|
| **FNO** | **20.6%** | **55.1%** | **20.11** |
| Transolver++ | 25.9% | 74.5% | 68.55 |
| 原版Transolver | 28.0% | 81.1% | 23.34 |
| Transolver-3 | 29.2% | 83.0% | 73.37 |
| MLP | 30.9% | 73.6% | 9.84 |

**当前配方首选FNO，Transolver系列首选++。** 表中为五例相对L2均值；FNO的位移最差27.9%、应力最差64.1%，++分别39.7%、77.8%。应力热点误差FNO47.4%、++80.5%，FNO应力在五例上全部领先，绝对应力精度仍不足。应力阶段实际19.95秒，已知位移读出2.07秒，不是服务请求速度基准。

五例包含两例官方TRAIN与三例官方TEST，但均与本轮TRAIN40、新增TRAIN160、DEV20互斥。历史U已知、CSV字节/坐标曾读，只称U-known/S-field-blind；不是全局未观察文件或最终测试。单seed、ver单载荷、容量及输入表达不同，不能断言家族上限。Hash单／五样本应力拟合约0.4%／1.5%，不作为跨CAD赢家证据。完整FE物理路线此前U为0.01623%，需要更多完整网格/材料/真实BC输入，原CSV signed S的导出语义仍未绑定，不能据位移准确宣称完整U/S已解决。见[五方法简报](results/geobc_fixed4800_pilot/BRIEF.md)、[Transolver对比](results/TRANSOLVER_COMPARISON.md)与[物理U历史报告](results/blind_solver_service/BRIEF.md)。

## 当前推进（2026-10-11）

FNO 局部25条件配对已完成，实际 UTC 耗时58.56分钟。两臂均完成4800次更新及全TRAIN120读出。旧FNO的位移／应力误差为14.0%／34.9%；grid25新查询头为18.1%／32.6%，native25为18.3%／33.0%。应力略有改善，位移明显变差，native25配对门槛未通过，暂不替换旧FNO。见[配对简报](results/geobc_fno_local_pair/BRIEF.md)。

Transolver++新训练制度在中断恢复后触发原24GiB reserved上限，实际停在23166／24000；最终24000物理指标缺失。新4800端点的TRAIN120位移／应力44.2%／80.1%，旧端点37.6%／77.6%。资源失败不能解释为整个Transolver家族的效果上限。见[制度确认结果](results/geobc_pp_regime_confirmation/BRIEF.md)。

固定20个新CAD×三载荷的九候选比较已经真实完成，全部540份预测或端点失败提交后才共享读取20份原CSV标签。初次启动的锁路径与CUDA设备索引错误发生在数据读取之前，记录已归档；显式技术恢复保留北京时间01:55:37准备截止、02:55:37整体截止，不重置预算。原20DEV仍封存；这20例来自可用的官方TRAIN池，是开发比较，不能宣称无偏最终测试。见[当前计划](notes/geobc_available_cad_pilot_plan.json)。

下一步根据新几何、三载荷的误差、热点和成本决定：若领先清晰，验证领先者与最佳Transolver的连续查询能力；若结果接近，设计统一网格输入、条件和查询头的容量配对，分清算子与表示的贡献。停止Hash小幅微调。见[后续选择分支](notes/geobc_solver_choice_branches.json)。

连续查询适配器已通过合成CPU数值检查，支持TET10位置、局部条件及物理坐标梯度；尚未验证训练模型的任意位置准确性，单元内节点插值也不等同于独立的离节点FEM真值。

原40／200CAD十臂协议因输入下载317／540、缺223，在原6小时预算内终止；固定ID和封存DEV未替换。见[输入失败简报](results/geobc_coverage_input_negative/BRIEF.md)。

私人GitHub最新真实验收提交为`84f4a48`，包含新20CAD完整结果、原节点云图、真实查询检查、FP32精度对照，以及正在运行的统一配置训练代码与启动证据；尚无两臂4800终态结果。仓库：[GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)。

## 复现与数据边界

使用原版DeepJEB，所有有效指标来自官方Field CSV按GRID坐标对齐H5 vertices。旧H5-only结果已撤回。保留原signed uXYZ与stress，不伪造符号。原实验目录、部分配置、slides的25特征schema、网格原点及50样本名单缺失，当前复现不能称为原run的精确数值重放。

[Slides复现总表](ALL_METHODS_RESULTS.md) · [正式结果包](results/REPORT.md) · [Transolver对比](results/TRANSOLVER_COMPARISON.md) · [研究过程记录](results/RESEARCH_HISTORY.md) · [执行计划](notes/geobc_exploration_plan.json)

## 实验范围

30 个实验包含 MLP、4-level NGP 的应力/位移拟合、12-level hash grid、官方 tiny-cuda-nn FullyFusedMLP、Transolver、Transolver++、Transolver-3，以及冻结 NGP 加 TS3/++ 输出残差、FNO 64³ 联合/应力输出、FNO 192³ 应力输出、FNO→hash 蒸馏和共享 50 样本 hash grid 两个训练阶段。单样本、双样本、五样本各自列出，避免混淆预算和案例。

Slides 第 2 页提到的 Geometry+BC 分支、mesh-flow prior、geometry-latent TS3、BenchCAD 和 FNO 分支仅有历史概述，没有可运行配置或量化源记录，报告列为不可恢复项。

| 数据目录 | 用途 | 原始 FEM 节点数 |
| --- | --- | ---: |
| `data/prepared_exact_multifield` | 原稿 `506_533/ver`、`506_25/tor` | 243,564 |
| `data/prepared_benchmark5` | 上述两个加 `10_30/ver`、`10_421/hor`、`10_460/dia` | 791,820 |
| `data/prepared50_multifield` | 官方压缩包前 50 个 CAD，循环分配四种载荷 | 10,684,820 |

50 样本原先的 270-CAD 训练池及抽样列表缺失，当前集合明确作为替代集合。全局 batch 51,200 对应每 epoch 209 次更新；1000 个真实 epoch 共 209,000 次更新。随机采样阶段结束后只继承最佳权重，重新初始化 Adam。

数据来源是 [官方 DeepJEB 站点](https://dataset.narnia.ai/deepjeb.html) 的原版 Drive 文件和 [KAIST 官方镜像](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB)，镜像 revision 固定为 `160424ca2a5095315ff9ead8a5fc36d2667fd3c6`。原稿两个案例的六个源文件及 SHA256 在 `references/deepjeb_exact_files.json`；全部准备参数在各数据目录的 `manifest.json`。

uXYZ 和有符号应力直接来自 CSV，不改变源符号。25 个条件特征由 FEM 几何、载荷、材料、约束接口计算，不包含标签或样本 ID；这是记录完整的替代 schema。水平载荷和扭矩采用实际 FEM 中的 −x 和 +z。二次 TETRA10 原始节点保留，绘图从体单元提取二次边界面。

## 配置与评估

Transformer 使用固定提交的官方实现，见 `third_party/transformer_sources.json`。Transolver++ 采用 slides 的 16×256；TS3 未公开尺寸，采用官方 volume preset。完整案例共享全局 context，TS3 的分块保留全局两遍计算。冻结 NGP 仅加在残差模型输出，不作为输入。

主要 hash 配置为 12 层、每层 2 特征、log2 表上限 19、base 8、growth 1.6，64 宽、2 隐藏层 ReLU。官方 FullyFusedMLP 与 Triton 编码加 PyTorch MLP 分开列出；前者原生 FP16 网络，后者 FP32。单样本只输入 xyz，多案例增加 25 个条件特征。每更新均匀抽取每案例 16,384 节点，1000 次随机更新不称为 1000 个 epoch。

Stress asinh 使用记录在配置中的 pooled P90 尺度、中心和标准差，严格逆变换后计算物理单位指标。所有节点模型评估全部原始 FEM 节点，报告 physical relative L2、案例均值/最差值、最高绝对应力的 1% 热点及位移向量误差。检查点选择标准在各配置中记录，不延长原稿预算来替换主要比较。

FNO 使用真实 TETRA10 二次插值构建网格；64³ 和 192³ 使用同一物理域，192³ 间距为 0.954956 mm。原网格原点缺失，当前采用居中补齐的 182.396596 mm 立方域，作为显式假设。occupied-grid、回查 FEM、真值网格往返及蒸馏 teacher/FEM 指标分开报告。蒸馏只以 teacher 标签训练和选择检查点。

上述30项 slides 实验评估训练案例的拟合能力。独立的 GeoBC 探索阶段已开展迁移诊断与按CAD隔离的验证，结果见探索报告；新模型的最终测试字段未参与训练或选模；原50案例历史实验使用过部分同CAD，不能称为全局从未观察的盲测。

## 运行

本机环境为 WSL `cold-surge-ubuntu-24.04`、Python `/home/research/.venvs/geobc`、RTX 5090 32 GB。锁定依赖见 `requirements-lock-wsl.txt`，tiny-cuda-nn 编译说明见 [notes/install_tcnn.md](notes/install_tcnn.md)。以下命令在 WSL 项目目录执行。

```bash
# 查看全部配置；已完成实验会复用
python tools/run_slides_suite.py --dry-run
# 顺序运行 GPU 阶段，复用已完成实验
python tools/run_slides_suite.py
# 只执行指定阶段
python tools/run_slides_suite.py --only tcnn_hash_single_exact
```

数据准备可独立重放；下载器复用已验证文件，原始节点顺序按 CSV 坐标匹配。

```bash
python tools/fetch_exact_deepjeb.py
python tools/prepare_multifield.py --build-inputs --raw data/raw_exact --output data/prepared_exact_multifield --cases 506_533/ver 506_25/tor --workers 2
python tools/fetch_deepjeb.py --count 50 --components FieldMesh Field SolverDeck
python tools/prepare_multifield.py --build-inputs --raw data/raw --output data/prepared50_multifield --archive-count 50 --workers 3
python tools/assemble_benchmark5.py
python tools/prepare_grids.py --prepared data/prepared_exact_multifield --raw data/raw_exact --cases 506_533/ver --resolution 64 --domain-length-mm 182.396596 --fields joint
python tools/prepare_grids.py --prepared data/prepared_exact_multifield --raw data/raw_exact --cases 506_533/ver --resolution 192 --spacing-mm 0.954956 --fields joint
```

单独的 FullyFusedMLP 单案例实验：

```bash
python -m geobc.benchmark --method hashgrid --backend tcnn-fused --data data/prepared_exact_multifield --case-keys 506_533/ver --outputs stress --input-mode xyz --protocol sampled --stress-transform asinh --selection worst-stress --steps 1000 --learning-rate 0.003 --output runs/my_hash_single
```

每次实验保存配置、best/last 检查点、全节点预测、最佳/最终物理指标、训练曲线及带 ETA 的 `status.json`。CPU 汇总与少量代表性云图：

```bash
python tools/summarize_suite.py
python tools/plot_suite_fields.py
```

30项 slides 实验及其 GitHub 交付已经完成，原监视已删除。用户进一步授权的 GeoBC 自主研究现由新的30分钟监视持续推进；阶段结束后根据证据安排下一阶段。代码、配置和可移植结果持续更新至私人 GitHub 仓库 `baiming-zhang/GeoBC-Neural-Operator`。导出/发布参数在 `notes/github_publish_plan.json`，发布快照保存在 `publication/GeoBC-Neural-Operator`。旧 `tools/replay_runs.py` 只描述已撤回的历史实验，不作为本次全方法入口。

GINO训练前已按[频带口径](notes/gino_vs_fno_implementation_scope.md)统一为官方(24,24,22)，对应旧FNO的x/y总24、z12；此前资源检查采用较小的默认频带，完整训练配置的首个325387节点case已通过，480更新结果已完成：GINO验证U54.42%/S72.46%，可比训练740.1秒；旧FNO480为47.51%/66.71%、99.8秒。GINO均值、最差与成本本轮均没有优先证据。新训练不会重复低频候选或据旧final调参。


