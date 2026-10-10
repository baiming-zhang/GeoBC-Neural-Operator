# GeoBC：原版 DeepJEB 复现与求解器筛选

项目研究如何根据几何、材料、载荷与约束预测原节点及连续空间的位移、signed应力。**最终solver尚未选定；Hash专项微调已停止。**

原版DeepJEB的30项可恢复slides实验已完成，随后探索十类学习结构及物理求解路线。全部阶段证据见[一分钟结果](results/SOLVER_BRIEF.md)，详细记录见[探索报告](results/geobc_exploration/REPORT.md)。

## 最新结果（2026-10-10）

全新五CAD/ver盲验实际用时3分13秒，十个独立进程完整预测后才共享读取原CSV位移标签：

| 路线 | 五CAD节点位移平均／最差相对L2 | 独立完整请求耗时 |
|---|---:|---:|
| 完整FE物理U | 0.01623%／0.01623% | 20.2–33.5秒 |
| 冻结FNO参考 | 23.51%／30.38% | 8.7–10.5秒 |

五例物理U门均通过，原signed应力标签仍封存。物理native P2梯度误差0.022%–0.040%，参考为原节点U的同P2派生代理，**非源应变GT**。物理路线需要完整TET10网格、材料、真实约束和载荷，FNO使用xyz／当前25条件schema／点云；输入信息不同，不能宣称架构因果排名。完整请求计入本地raw→输入→场与物理查询，数据获取和上游网格生成另计；FNO离线训练608.7秒另报。见[盲验短报告](results/blind_solver_service/BRIEF.md)。

Hash单／五样本应力拟合约0.4%／1.5%，这不是跨CAD泛化证据。FNO当前学习配方较有依据，但旧final已知载荷signed应力仍约49–50%，组合载荷更差；物理信息应力读出候选也未达固定门，完整U/S任务仍未解决。

## 下一阶段

当前学习训练集只有40CAD×三载荷。下一步做FNO／MLP的40对200CAD覆盖对照，保留各结构、输入、原signed目标与原TRAIN40-only scaler；新增160TRAIN和20DEV按CAD隔离，固定端点完整预测后再共享读取DEV标签。新增540原始文件尚未获取，下载时间粗估2.5小时，新增存储约73GiB。见[数据获取可行性](notes/geobc_coverage_data_delivery_feasibility.json)。没有设计族谱，只声明CAD ID互斥。

## 复现与数据边界

使用原版DeepJEB，所有有效指标来自官方Field CSV按GRID坐标对齐H5 vertices。旧H5-only结果已撤回。保留原signed uXYZ与stress，不伪造符号。原实验目录、部分配置、slides的25特征schema、网格原点及50样本名单缺失，当前复现不能称为原run的精确数值重放。

[Slides复现总表](ALL_METHODS_RESULTS.md) · [正式结果包](results/REPORT.md) · [研究过程记录](results/RESEARCH_HISTORY.md) · [执行计划](notes/geobc_exploration_plan.json)

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
