# DeepJEB 全方法复现进度与结果

快照时间：2026-10-08T14:15:11.854380+00:00。已完成 30/30 个计划实验。

本表使用 original DeepJEB v1.0；应力保留源数据符号。只汇总当前计划明确列出的实验，旧 H5 字段与坐标错位的结果已撤回，不纳入本表。所有误差均为解码后的物理相对 L2，表内单位为 %，没有截断。— 表示该实验不输出/不涉及该指标或案例；pending 仅表示尚无保存的测量。原稿未公开的指标写“未提供”。进行中实验的最佳值只是当前已观测最佳。

主对照是 506_533/ver 与 506_25/tor，共 243564 个原始节点。五样本补充对照再加入 10_30/ver、10_421/hor、10_460/dia。50 CAD 使用官方前 50 个 CAD 的明确替代集合；原 270 训练池和历史随机 50 CAD 无法获取，因此不能把其数值与历史结果视为同一集合的重复实验。

这些实验检验训练样本拟合；没有证明 heldout、重网格或跨边界条件泛化。不同容量、预算和 checkpoint 目标不能作为等容量模型排名。

## 复现范围

| 原稿中的方法/实验 | 本次状态 |
|---|---|
| 坐标 MLP、小 NGP、12-level neural field、Transolver++、TS3、输出端 residual、64³/192³ FNO、FNO teacher distillation、50 CAD shared field | 有量化实验与可恢复的部分配置，按计划实际运行；已完成/待运行见下表。原始 Transolver 为用户新增对照。 |
| Geometry+BC separate branches、mesh-flow priors、geometry latent TS3、BenchCAD tension/multi-BC、FNO branches | 仅历史概述；本 slides 未提供量化结果/完整配置，当前无法精确复现。未声称这些历史架构已训练完成。 |

## FEM 节点结果

| 实验 | 状态 | 最佳步 / 已跑 / 计划 | 应力 pooled L2 % | 应力 case均值 / 最差 % | 位移 pooled L2 % | 已耗时 |
|---|---|---:|---:|---:|---:|---:|
| mlp_single_exact | 已完成 | 500 / 500 / 500 | 39.51514 | 39.51514 / 39.51514 | 3.63705 | 4.23 s |
| ngp4_506_533_ver_stress | 已完成 | 1000 / 1000 / 1000 | 11.45004 | 11.45004 / 11.45004 | — | 3.65 s |
| ngp4_506_533_ver_u | 已完成 | 1000 / 1000 / 1000 | — | — | 0.90981 | 3.38 s |
| ngp4_506_25_tor_stress | 已完成 | 1000 / 1000 / 1000 | 14.33958 | 14.33958 / 14.33958 | — | 3.40 s |
| ngp4_506_25_tor_u | 已完成 | 1000 / 1000 / 1000 | — | — | 1.32269 | 3.54 s |
| hash_single_exact | 已完成 | 400 / 1000 / 1000 | 0.28911 | 0.28911 / 0.28911 | — | 6.28 s |
| hash_two_exact | 已完成 | 1000 / 1000 / 1000 | 1.18041 | 1.32604 / 1.56358 | — | 8.40 s |
| hash_five_aligned | 已完成 | 800 / 1000 / 1000 | 2.06856 | 2.53442 / 3.40315 | — | 11.30 s |
| tcnn_hash_single_exact | 已完成 | 500 / 1000 / 1000 | 0.41394 | 0.41394 / 0.41394 | — | 18.37 s |
| tcnn_hash_two_exact | 已完成 | 900 / 1000 / 1000 | 0.75745 | 0.76813 / 0.78424 | — | 26.13 s |
| tcnn_hash_five_aligned | 已完成 | 900 / 1000 / 1000 | 1.32410 | 1.44950 / 2.11226 | — | 52.08 s |
| transolver_single | 已完成 | 400 / 500 / 500 | 86.52684 | 86.52684 / 86.52684 | 32.67956 | 46.27 s |
| transolverpp_single_asinh | 已完成 | 500 / 500 / 500 | 91.61171 | 91.61171 / 91.61171 | 75.39369 | 3.36 min |
| ts3_single | 已完成 | 300 / 500 / 500 | 85.61803 | 85.61803 / 85.61803 | 33.70822 | 3.50 min |
| transolver_five | 已完成 | 2500 / 2500 / 2500 | 90.82016 | 91.76998 / 102.38315 | 55.56531 | 4.96 min |
| transolverpp_five | 已完成 | 2500 / 2500 / 2500 | 88.59534 | 90.56315 / 102.37348 | 43.92361 | 21.96 min |
| ts3_five | 已完成 | 2100 / 2500 / 2500 | 91.52027 | 92.13894 / 100.90270 | 73.65086 | 24.13 min |
| ts3_two_short | 已完成 | 500 / 800 / 800 | 93.12498 | 96.15642 / 100.78256 | 74.83699 | 5.92 min |
| transolver_two | 已完成 | 1000 / 1000 / 1000 | 87.50472 | 92.43156 / 100.07161 | 35.92429 | 1.59 min |
| transolverpp_two_linear | 已完成 | 900 / 1000 / 1000 | 88.38303 | 93.00824 / 100.16094 | 36.66647 | 6.82 min |
| transolverpp_two_asinh | 已完成 | 1000 / 1000 / 1000 | 98.56451 | 99.20608 / 100.16895 | 72.05267 | 6.82 min |
| ts3_residual_two | 已完成 | 200 / 1000 / 1000 | 11.97323 | 12.89790 / 14.35323 | 0.92491 | 7.46 min |
| transolverpp_residual_two | 已完成 | 200 / 1000 / 1000 | 11.96878 | 12.89117 / 14.34272 | 0.92564 | 6.79 min |
| fno64_joint | 已完成 | 500 / 500 / 500 | 30.37069 | 30.37069 / 30.37069 | 2.25498 | 12.40 s |
| fno64_stress | 已完成 | 500 / 500 / 500 | 30.36147 | 30.36147 / 30.36147 | — | 12.17 s |
| fno192_stress | 已完成 | 9700 / 9905 / 9905 | 14.34100 | 14.34100 / 14.34100 | — | 26.98 min |
| fno192_hash_distill | 已完成 | 500 / 1000 / 1000 | 21.00189 | 21.00189 / 21.00189 | — | 6.12 s |
| ts3_two_long | 已完成 | 8900 / 10000 / 10000 | 74.40051 | 83.32091 / 97.82881 | 15.40255 | 1.22 h |
| hash50_sampled | 已完成 | 10000 / 10000 / 10000 | 22.07414 | 24.43082 / 39.02995 | — | 46.55 s |
| hash50_full1000ep | 已完成 | 169290 / 209000 / 209000 | 13.94474 | 16.16664 / 28.96709 | — | 7.59 min |

最佳 checkpoint 的选择标准见下表；显示 FEM 误差并不表示使用 FEM 误差选择网格模型或蒸馏模型。

## 网格与 teacher 的独立误差

| 实验 | occupied-grid stress L2 % | teacher点 stress L2 % | FEM-node stress L2 % | 真场 roundtrip stress L2 % |
|---|---:|---:|---:|---:|
| fno64_joint | 2.27438 | — | 30.37069 | 30.36503 |
| fno64_stress | 1.05819 | — | 30.36147 | 30.36503 |
| fno192_stress | 1.09912 | — | 14.34100 | 14.39559 |
| fno192_hash_distill | — | 0.30218 | 21.00189 | — |

真场 roundtrip 是实际 FEM→网格→节点的表示/插值损失，不是模型误差的数学下界。蒸馏仅使用 FNO occupied-grid 预测监督；FEM 标签不参与神经场 scaler、loss 或 checkpoint 选择。

## 与原稿数字逐案例对照

原稿列来自实际图中文字或 speaker notes，不是已恢复的原 run/summary.json。图中一位小数保持一位；备注给出更精确数字时明确标为“备注”，fraction 转换的显示精度也已记录在 JSON。原稿短 TS3 是 update 800 的评估，本次列为所运行预算中的最佳 checkpoint，二者选择时点不同。同一 run 的各案例/指标都来自同一个选定 checkpoint，并非逐案例另挑最低值。当前进行中的值仍是暂定最佳。

| 实验 / 指标 | 506_533/ver 原稿 % | 本次 best % | 506_25/tor 原稿 % | 本次 best % | 原稿预算 → 本次预算 | 主要差异 / 缺口 |
|---|---:|---:|---:|---:|---|---|
| mlp_single_exact / stress FEM nodes | 40.7356（备注） | 39.51514 | — | — | epochs=500 → 500 epochs / 500 updates | MLP 历史学习率未知；本次 0.003。 |
| mlp_single_exact / u vector FEM nodes | 未提供 | 3.63705 | — | — | epochs=500 → 500 epochs / 500 updates | 原稿未给该指标；未补造数字。 MLP 历史学习率未知；本次 0.003。 |
| ngp4_506_533_ver_stress / stress FEM nodes | 8.4780（备注） | 11.45004 | — | — | updates=1000 → 1000 updates | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 原稿 stress-only 76928 parameters；本次 76129 parameters。完整设置不可恢复，不能声称等容量。 |
| ngp4_506_533_ver_u / u vector FEM nodes | 0.390（备注） | 0.90981 | — | — | unknown → 1000 updates | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 原稿 u-only 参数总数未知；本次 76195 parameters。完整设置不可恢复，不能声称等容量。 |
| ngp4_506_25_tor_stress / stress FEM nodes | — | — | 12.88（备注） | 14.33958 | updates=1000 → 1000 updates | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 原稿 stress-only 76928 parameters；本次 76129 parameters。完整设置不可恢复，不能声称等容量。 |
| ngp4_506_25_tor_u / u vector FEM nodes | — | — | 1.037（备注） | 1.32269 | unknown → 1000 updates | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 原稿 u-only 参数总数未知；本次 76195 parameters。完整设置不可恢复，不能声称等容量。 |
| hash_single_exact / stress FEM nodes | 0.0773（备注） | 0.28911 | — | — | updates=1000, best update=600 → 1000 updates | 历史 asinh 常数/学习率未知；本次 pooled \|stress\| P90、中心化标准化、LR 0.003。 MLP 使用 torch 实现；历史 fused MLP 后端未复刻。 |
| hash_two_exact / stress FEM nodes | 0.088（图） | 1.08851 | 0.1077（备注） | 1.56358 | updates=1000 → 1000 updates | 历史 asinh 常数/学习率未知；本次 pooled \|stress\| P90、中心化标准化、LR 0.003。 MLP 使用 torch 实现；历史 fused MLP 后端未复刻。 |
| hash_five_aligned / stress FEM nodes | — | 1.30279 | — | 2.89763 | 本 slides 未给对应配置/数值 → 1000 updates | 新增/补充对照，无同预算原稿数字。 历史 asinh 常数/学习率未知；本次 pooled \|stress\| P90、中心化标准化、LR 0.003。 MLP 使用 torch 实现；历史 fused MLP 后端未复刻。 |
| tcnn_hash_single_exact / stress FEM nodes | 0.0773（备注） | 0.41394 | — | — | updates=1000, best update=600 → 1000 updates | 官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。 历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。 历史 LR/asinh 常数未给；本次采用已保存的 P90/中心化标准化与 LR0.003。 官方 FullyFused native FP16 / FP32 master；与早期 Triton encoding + torch MLP 分列。 |
| tcnn_hash_two_exact / stress FEM nodes | 0.088（图） | 0.75202 | 0.1077（备注） | 0.78424 | updates=1000 → 1000 updates | 官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。 历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 官方 FullyFused native FP16 / FP32 master；与早期 Triton encoding + torch MLP 分列。 25 通道为真实 FEM deck 替代 schema，历史定义缺失。 |
| tcnn_hash_five_aligned / stress FEM nodes | 0.088（图） | 0.94379 | 0.1077（备注） | 0.88145 | updates=1000 → 1000 updates | 官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。 历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 官方 FullyFused native FP16 / FP32 master；与早期 Triton encoding + torch MLP 分列。 25 通道为真实 FEM deck 替代 schema，历史定义缺失。 原稿共享2案例、本次共享5案例；同名两case仅并列，训练集合及每update总采样量不同。 |
| transolver_single / stress FEM nodes | — | 86.52684 | — | — | 本 slides 未给对应配置/数值 → 500 epochs / 500 updates | 新增/补充对照，无同预算原稿数字。 原始 Transolver 为新增对照；采用官方 constructor defaults。 |
| transolver_single / u vector FEM nodes | — | 32.67956 | — | — | 本 slides 未给对应配置/数值 → 500 epochs / 500 updates | 新增/补充对照，无同预算原稿数字。 原始 Transolver 为新增对照；采用官方 constructor defaults。 |
| transolverpp_single_asinh / stress FEM nodes | — | 91.61171 | — | — | 本 slides 未给对应配置/数值 → 500 epochs / 500 updates | 新增/补充对照，无同预算原稿数字。 单样本为两样本实验的补充。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| transolverpp_single_asinh / u vector FEM nodes | — | 75.39369 | — | — | 本 slides 未给对应配置/数值 → 500 epochs / 500 updates | 新增/补充对照，无同预算原稿数字。 单样本为两样本实验的补充。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| ts3_single / stress FEM nodes | — | 85.61803 | — | — | 本 slides 未给对应配置/数值 → 500 epochs / 500 updates | 新增/补充对照，无同预算原稿数字。 TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| ts3_single / u vector FEM nodes | — | 33.70822 | — | — | 本 slides 未给对应配置/数值 → 500 epochs / 500 updates | 新增/补充对照，无同预算原稿数字。 TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| transolver_five / stress FEM nodes | — | 90.98056 | — | 102.38315 | 本 slides 未给对应配置/数值 → 500 epochs / 2500 updates | 新增/补充对照，无同预算原稿数字。 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| transolver_five / u vector FEM nodes | — | 49.69664 | — | 116.14049 | 本 slides 未给对应配置/数值 → 500 epochs / 2500 updates | 新增/补充对照，无同预算原稿数字。 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| transolverpp_five / stress FEM nodes | — | 88.19046 | — | 102.37348 | 本 slides 未给对应配置/数值 → 500 epochs / 2500 updates | 新增/补充对照，无同预算原稿数字。 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| transolverpp_five / u vector FEM nodes | — | 36.18252 | — | 106.48475 | 本 slides 未给对应配置/数值 → 500 epochs / 2500 updates | 新增/补充对照，无同预算原稿数字。 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| ts3_five / stress FEM nodes | — | 91.85206 | — | 100.90270 | 本 slides 未给对应配置/数值 → 500 epochs / 2500 updates | 新增/补充对照，无同预算原稿数字。 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| ts3_five / u vector FEM nodes | — | 72.47335 | — | 103.90083 | 本 slides 未给对应配置/数值 → 500 epochs / 2500 updates | 新增/补充对照，无同预算原稿数字。 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| ts3_two_short / stress FEM nodes | 45.0（图） | 91.53028 | 78.4（图） | 100.78256 | eval update=800, stop update=897 → 800 updates | TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| ts3_two_short / u vector FEM nodes | 未提供 | 72.84598 | 未提供 | 116.77584 | eval update=800, stop update=897 → 800 updates | 原稿未给该指标；未补造数字。 TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| transolver_two / stress FEM nodes | — | 84.79150 | — | 100.07161 | 本 slides 未给对应配置/数值 → 500 epochs / 1000 updates | 新增/补充对照，无同预算原稿数字。 原始 Transolver 为新增对照；采用官方 constructor defaults。 |
| transolver_two / u vector FEM nodes | — | 32.27392 | — | 89.97331 | 本 slides 未给对应配置/数值 → 500 epochs / 1000 updates | 新增/补充对照，无同预算原稿数字。 原始 Transolver 为新增对照；采用官方 constructor defaults。 |
| transolverpp_two_linear / stress FEM nodes | 90.3（图） | 85.85553 | 99.0（图） | 100.16094 | epochs=500, updates=1000, LR=0.003 → 500 epochs / 1000 updates | slides 仅提 25 维，未定义通道；本次从真实 FEM deck 构造明确的替代 BC/material schema。 |
| transolverpp_two_linear / u vector FEM nodes | 未提供 | 33.03353 | 未提供 | 90.91588 | epochs=500, updates=1000, LR=0.003 → 500 epochs / 1000 updates | 原稿未给该指标；未补造数字。 slides 仅提 25 维，未定义通道；本次从真实 FEM deck 构造明确的替代 BC/material schema。 |
| transolverpp_two_asinh / stress FEM nodes | 70.3（图） | 98.24321 | 100.0（图） | 100.16895 | epochs=500, updates=1000, LR=0.003 → 500 epochs / 1000 updates | 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| transolverpp_two_asinh / u vector FEM nodes | 未提供 | 68.22111 | 未提供 | 141.00278 | epochs=500, updates=1000, LR=0.003 → 500 epochs / 1000 updates | 原稿未给该指标；未补造数字。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| ts3_residual_two / stress FEM nodes | 8.5（图） | 11.44258 | 12.9（图） | 14.35323 | epochs=500, LR=0.003 → 500 epochs / 1000 updates | 冻结 NGP 只在输出相加，未作为 correction 输入。 小 NGP 完整配置缺失，采用已记录的替代值。 |
| ts3_residual_two / u vector FEM nodes | 未提供 | 0.90551 | 未提供 | 1.35126 | epochs=500, LR=0.003 → 500 epochs / 1000 updates | 原稿未给该指标；未补造数字。 冻结 NGP 只在输出相加，未作为 correction 输入。 小 NGP 完整配置缺失，采用已记录的替代值。 |
| transolverpp_residual_two / stress FEM nodes | 8.5（图） | 11.43961 | 12.9（图） | 14.34272 | epochs=500, LR=0.003 → 500 epochs / 1000 updates | 冻结 NGP 只在输出相加，未作为 correction 输入。 小 NGP 完整配置缺失，采用已记录的替代值。 |
| transolverpp_residual_two / u vector FEM nodes | 未提供 | 0.90620 | 未提供 | 1.35275 | epochs=500, LR=0.003 → 500 epochs / 1000 updates | 原稿未给该指标；未补造数字。 冻结 NGP 只在输出相加，未作为 correction 输入。 小 NGP 完整配置缺失，采用已记录的替代值。 |
| fno64_joint / stress FEM nodes | 31.2571（备注） | 30.37069 | — | — | epochs=500 → 500 updates | 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno64_joint / u vector FEM nodes | 2.862（备注） | 2.25498 | — | — | epochs=500 → 500 updates | 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno64_joint / stress grid points | 未提供 | 2.27438 | — | — | epochs=500 → 500 updates | 原稿未给该指标；未补造数字。 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno64_joint / u vector grid points | 未提供 | 0.71786 | — | — | epochs=500 → 500 updates | 原稿未给该指标；未补造数字。 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno64_stress / stress FEM nodes | 30.6208（备注） | 30.36147 | — | — | epochs=500, best epoch=300 → 500 updates | 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno64_stress / stress grid points | 未提供 | 1.05819 | — | — | epochs=500, best epoch=300 → 500 updates | 原稿未给该指标；未补造数字。 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno192_stress / stress FEM nodes | 14.3782（备注） | 14.34100 | — | — | stop epoch=9905, best epoch=8200 → 9905 updates | 复刻 9905 次停止预算，不称自动收敛。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno192_stress / stress grid points | 2.4553（备注） | 1.09912 | — | — | stop epoch=9905, best epoch=8200 → 9905 updates | 复刻 9905 次停止预算，不称自动收敛。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno192_hash_distill / stress FEM nodes | 17.8743（备注） | 21.00189 | — | — | updates=1000 → 1000 updates | teacher 选本次最佳网格 checkpoint，不强制历史 epoch 8200。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 官方 FullyFused native FP16 / FP32 master；与早期 Triton encoding + torch MLP 分列。 |
| fno192_hash_distill / stress teacher points | 0.070（图） | 0.30218 | — | — | updates=1000 → 1000 updates | teacher 选本次最佳网格 checkpoint，不强制历史 epoch 8200。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 官方 FullyFused native FP16 / FP32 master；与早期 Triton encoding + torch MLP 分列。 |
| ts3_two_long / stress FEM nodes | 5.5437（备注） | 68.81301 | 11.5111（备注） | 97.82881 | epochs=5000, updates=10000, best update=9900 → 5000 epochs / 10000 updates | TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| ts3_two_long / u vector FEM nodes | 0.5768（备注） | 13.30807 | 2.2882（备注） | 43.36969 | epochs=5000, updates=10000, best update=9900 → 5000 epochs / 10000 updates | TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| 冻结 NGP reference（slide 6） / stress | 8.5（图） | 11.45004 | 12.9（图） | 14.33958 | 1000 NGP updates → 本次每案例 1000 updates | 原图 8.5/12.9 仅一位小数；不代表与 residual 的底层误差完全相同。NGP 完整配置/容量缺失。 |

本次 25 通道并非演示稿定义：slides 只注明 25 维，本次从真实 FEM deck 构造了明确替代 schema。小 NGP 完整容量未知；大 hash 的历史 LR/asinh 常数及 fused-MLP 后端细节也未知。因此保留差异，不为对齐数字调整配置或指标。

| 50 CAD 统计量 | 原稿 % | 本次 best % | 原稿预算 → 本次预算 | 集合差异 |
|---|---:|---:|---|---|
| hash50_sampled / stress_case_mean | 23.70（图） | 24.43082 | sampled updates=10000 → 10000 updates | 原270池随机50 vs 官方前50；case/node数不同，不能视为同集合复现。 |
| hash50_full1000ep / stress_case_mean | 15.1108（备注） | 16.16664 | epochs=1000, best epoch=990, best update=215820, updates=218000 → 1000 epochs / 209000 updates | 原270池随机50 vs 官方前50；case/node数不同，不能视为同集合复现。 |
| hash50_full1000ep / stress_pooled | 12.8665（备注） | 13.94474 | epochs=1000, best epoch=990, best update=215820, updates=218000 → 1000 epochs / 209000 updates | 原270池随机50 vs 官方前50；case/node数不同，不能视为同集合复现。 |
| hash50_full1000ep / stress_best | 8.7621（备注） | 8.50951 | epochs=1000, best epoch=990, best update=215820, updates=218000 → 1000 epochs / 209000 updates | 原270池随机50 vs 官方前50；case/node数不同，不能视为同集合复现。 |
| hash50_full1000ep / stress_worst | 28.4432（备注） | 28.96709 | epochs=1000, best epoch=990, best update=215820, updates=218000 → 1000 epochs / 209000 updates | 原270池随机50 vs 官方前50；case/node数不同，不能视为同集合复现。 |

## 配置与缺口

原实验服务器目录不可访问；演示稿明确给出的配置予以保留，缺失部分使用明确记录的官方 defaults/preset 或替代值，不能称逐项恢复了历史配置。Transformer 保持整张 FEM 网格上下文；TS3 分块是官方全局统计/回写算法，未把独立分块误当全网格评估。Transolver++ 保留官方 eval 时的随机 Gumbel 行为。

| 实验 | 演示稿可确认项 | 本次配置 / 来源 | checkpoint 选择 | 配置缺口与补充 |
|---|---|---|---|---|
| mlp_single_exact | slide 8: 4×256 ReLU, xyz, joint output, 500 epochs | xyz→4×256 ReLU→uXYZ/stress | physical-mse | MLP 历史学习率未知；本次 0.003。 |
| ngp4_506_533_ver_stress | slide 9: 4 levels, 76928 parameters, 1000 updates | 4 levels; cap log2=14, base=8, growth=2, MLP 2×32 (recorded substitute) | physical-mse | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 |
| ngp4_506_533_ver_u | slide 9: 4 levels, 76928 parameters, 1000 updates | 4 levels; cap log2=14, base=8, growth=2, MLP 2×32 (recorded substitute) | physical-mse | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 |
| ngp4_506_25_tor_stress | slide 9: 4 levels, 76928 parameters, 1000 updates | 4 levels; cap log2=14, base=8, growth=2, MLP 2×32 (recorded substitute) | physical-mse | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 |
| ngp4_506_25_tor_u | slide 9: 4 levels, 76928 parameters, 1000 updates | 4 levels; cap log2=14, base=8, growth=2, MLP 2×32 (recorded substitute) | physical-mse | 小 NGP 仅 4 levels 可确认；hash cap/base/growth/MLP width 为记录的替代值。 历史采样量/学习率/缩放细节缺失；使用并保存本次明确设置。 |
| hash_single_exact | slides 14–17: hash dimensions; original fused MLP | 12×2 hash, cap log2=19, base8, growth1.6, MLP 2×64; torch MLP | worst-stress | 历史 asinh 常数/学习率未知；本次 pooled \|stress\| P90、中心化标准化、LR 0.003。 MLP 使用 torch 实现；历史 fused MLP 后端未复刻。 |
| hash_two_exact | slides 14–17: hash dimensions; original fused MLP | 12×2 hash, cap log2=19, base8, growth1.6, MLP 2×64; torch MLP | worst-stress | 历史 asinh 常数/学习率未知；本次 pooled \|stress\| P90、中心化标准化、LR 0.003。 MLP 使用 torch 实现；历史 fused MLP 后端未复刻。 |
| hash_five_aligned | slides 14–17: hash dimensions; original fused MLP | 12×2 hash, cap log2=19, base8, growth1.6, MLP 2×64; torch MLP | worst-stress | 历史 asinh 常数/学习率未知；本次 pooled \|stress\| P90、中心化标准化、LR 0.003。 MLP 使用 torch 实现；历史 fused MLP 后端未复刻。 |
| tcnn_hash_single_exact | slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID | 12×2 hash, cap log2=19, base8, growth1.6; official FullyFusedMLP 2×64 ReLU; input=xyz; native FP16 / FP32 master；[3daa6e5b8b8e](https://github.com/NVlabs/tiny-cuda-nn/tree/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38) | worst-stress | 官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。 历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。 历史 LR/asinh 常数未给；本次采用已保存的 P90/中心化标准化与 LR0.003。 |
| tcnn_hash_two_exact | slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID | 12×2 hash, cap log2=19, base8, growth1.6; official FullyFusedMLP 2×64 ReLU; input=both; native FP16 / FP32 master；[3daa6e5b8b8e](https://github.com/NVlabs/tiny-cuda-nn/tree/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38) | worst-stress | 官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。 历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 slides 仅提25维；本次从真实 FEM deck 构造明确替代 schema，历史定义缺失。 |
| tcnn_hash_five_aligned | slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID | 12×2 hash, cap log2=19, base8, growth1.6; official FullyFusedMLP 2×64 ReLU; input=both; native FP16 / FP32 master；[3daa6e5b8b8e](https://github.com/NVlabs/tiny-cuda-nn/tree/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38) | worst-stress | 官方 TCNN FullyFusedMLP 实现原稿明确的 hash/MLP 架构；native FP16 computation / FP32 master。 历史 LR/asinh 常数/初始化 seed 未给；采用明确保存的本次值。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 slides 仅提25维；本次从真实 FEM deck 构造明确替代 schema，历史定义缺失。 |
| transolver_single | {} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 5, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[75e0f6764380](https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7) | physical-mse | 原始 Transolver 为新增对照；采用官方 constructor defaults。 |
| transolverpp_single_asinh | {"output_scaling": ["linear", "invertible asinh stress"], "updates": 1000, "n_layers": 16, "epochs": 500, "n_hidden": 256, "selection": "decoded physical normalized MSE", "learning_rate": 0.003} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[d5a23bc734a0](https://github.com/thuml/Transolver_plus/tree/d5a23bc734a0ebac56384cf72049a26af9673452) | physical-mse | 单样本为两样本实验的补充。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| ts3_single | {"long_epochs": 5000, "short_stopped_update": 897, "short_reported_update": 800, "input_channels": 28, "long_updates": 10000, "output_channels": 4} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 2, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 64, "space_dim": 3, "unified_pos": false}；[ef4fee9fa08d](https://github.com/thuml/Transolver-3/tree/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37) | physical-mse | TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| transolver_five | {} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 5, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[75e0f6764380](https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7) | physical-mse | 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| transolverpp_five | {"output_scaling": ["linear", "invertible asinh stress"], "updates": 1000, "n_layers": 16, "epochs": 500, "n_hidden": 256, "selection": "decoded physical normalized MSE", "learning_rate": 0.003} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[d5a23bc734a0](https://github.com/thuml/Transolver_plus/tree/d5a23bc734a0ebac56384cf72049a26af9673452) | physical-mse | 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| ts3_five | {"long_epochs": 5000, "short_stopped_update": 897, "short_reported_update": 800, "input_channels": 28, "long_updates": 10000, "output_channels": 4} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 2, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 64, "space_dim": 3, "unified_pos": false}；[ef4fee9fa08d](https://github.com/thuml/Transolver-3/tree/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37) | physical-mse | 五样本为补充实验，每个案例完整遍历 500 次。 未公布的配置使用官方 preset/default，不能称历史配置完全恢复。 |
| ts3_two_short | {"long_epochs": 5000, "short_stopped_update": 897, "short_reported_update": 800, "input_channels": 28, "long_updates": 10000, "output_channels": 4} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 2, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 64, "space_dim": 3, "unified_pos": false}；[ef4fee9fa08d](https://github.com/thuml/Transolver-3/tree/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37) | physical-mse | TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| transolver_two | {} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 5, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[75e0f6764380](https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7) | physical-mse | 原始 Transolver 为新增对照；采用官方 constructor defaults。 |
| transolverpp_two_linear | {"output_scaling": ["linear", "invertible asinh stress"], "updates": 1000, "n_layers": 16, "epochs": 500, "n_hidden": 256, "selection": "decoded physical normalized MSE", "learning_rate": 0.003} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[d5a23bc734a0](https://github.com/thuml/Transolver_plus/tree/d5a23bc734a0ebac56384cf72049a26af9673452) | physical-mse | slides 仅提 25 维，未定义通道；本次从真实 FEM deck 构造明确的替代 BC/material schema。 |
| transolverpp_two_asinh | {"output_scaling": ["linear", "invertible asinh stress"], "updates": 1000, "n_layers": 16, "epochs": 500, "n_hidden": 256, "selection": "decoded physical normalized MSE", "learning_rate": 0.003} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[d5a23bc734a0](https://github.com/thuml/Transolver_plus/tree/d5a23bc734a0ebac56384cf72049a26af9673452) | physical-mse | 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| ts3_residual_two | {"long_epochs": 5000, "short_stopped_update": 897, "short_reported_update": 800, "input_channels": 28, "long_updates": 10000, "output_channels": 4} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 2, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 64, "space_dim": 3, "unified_pos": false}；[ef4fee9fa08d](https://github.com/thuml/Transolver-3/tree/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37) | physical-mse | 冻结 NGP 只在输出相加，未作为 correction 输入。 小 NGP 完整配置缺失，采用已记录的替代值。 |
| transolverpp_residual_two | {"output_scaling": ["linear", "invertible asinh stress"], "updates": 1000, "n_layers": 16, "epochs": 500, "n_hidden": 256, "selection": "decoded physical normalized MSE", "learning_rate": 0.003} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 1, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 32, "space_dim": 3, "unified_pos": false}；[d5a23bc734a0](https://github.com/thuml/Transolver_plus/tree/d5a23bc734a0ebac56384cf72049a26af9673452) | physical-mse | 冻结 NGP 只在输出相加，未作为 correction 输入。 小 NGP 完整配置缺失，采用已记录的替代值。 |
| fno64_joint | slides 10–13: grid size / width / modes / blocks / 18 inputs | 64³, 4 blocks, width16, modes12, 18 channels; projection128, padding0 | physical normalized MSE on occupied grid, never original nodes | 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno64_stress | slides 10–13: grid size / width / modes / blocks / 18 inputs | 64³, 4 blocks, width16, modes12, 18 channels; projection128, padding0 | physical normalized MSE on occupied grid, never original nodes | 原 18 网格输入定义缺失；使用 occupancy+16 基本 BC/material+尺度。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno192_stress | slides 10–13: grid size / width / modes / blocks / 18 inputs | 192³, 4 blocks, width16, modes16, 18 channels; projection128, padding0 | physical normalized MSE on occupied grid, never original nodes | 复刻 9905 次停止预算，不称自动收敛。 FNO projection/padding/学习率的历史值缺失；本次 projection128/padding0/LR0.003。 |
| fno192_hash_distill | slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID | 12×2 hash, cap log2=19, base8, growth1.6; official FullyFusedMLP 2×64 ReLU; input=xyz; native FP16 / FP32 master；[3daa6e5b8b8e](https://github.com/NVlabs/tiny-cuda-nn/tree/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38) | minimum decoded teacher-point relative L2 | teacher 选本次最佳网格 checkpoint，不强制历史 epoch 8200。 历史 asinh 常数缺失；使用本次记录的 pooled P90/均值/标准差。 |
| ts3_two_long | {"long_epochs": 5000, "short_stopped_update": 897, "short_reported_update": 800, "input_channels": 28, "long_updates": 10000, "output_channels": 4} | {"act": "gelu", "dropout": 0.0, "fun_dim": 25, "mlp_ratio": 2, "n_head": 8, "n_hidden": 256, "n_layers": 16, "out_dim": 4, "ref": 8, "slice_num": 64, "space_dim": 3, "unified_pos": false}；[ef4fee9fa08d](https://github.com/thuml/Transolver-3/tree/ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37) | physical-mse | TS3 历史骨干/优化配置缺失；使用官方 volume preset 16×256、64 slices，已记录替代。 |
| hash50_sampled | slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID | 12×2 hash, cap log2=19, base8, growth1.6; official FullyFusedMLP 2×64 ReLU; input=both; native FP16 / FP32 master；[3daa6e5b8b8e](https://github.com/NVlabs/tiny-cuda-nn/tree/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38) | minimum worst-case all-original-node decoded stress relative L2 | 原 270 训练池缺失，改用官方前 50 CAD；13 ver/13 hor/12 dia/12 tor。 slides 仅提25维；本次从真实 FEM deck 构造明确替代 schema，历史定义缺失。 |
| hash50_full1000ep | slides 14–17: stated hash dimensions and 2×64 fused MLP; no sample ID | 12×2 hash, cap log2=19, base8, growth1.6; official FullyFusedMLP 2×64 ReLU; input=both; native FP16 / FP32 master；[3daa6e5b8b8e](https://github.com/NVlabs/tiny-cuda-nn/tree/3daa6e5b8b8e0fa17971a1a20acfbb3754494b38) | minimum worst-case all-original-node decoded stress relative L2 | 50 CAD 为明确替代集合，与历史随机 50 CAD 不同。 真 epoch 更新数按实际节点总数重算。 slides 仅提25维；本次从真实 FEM deck 构造明确替代 schema，历史定义缺失。 |

已运行实验的精确参数、scaler 数值、精度设置、数据路径和原始配置缺口保存于 notes/all_methods_results.json，以及各 run/config.json。Torch/Transformer 使用 float32 张量并明确允许 TF32；官方 TCNN FullyFused 使用 native FP16 computation、FP32 master parameters、返回 float32 输出。历史精度模式未知。

50 样本随机阶段使用全池均匀、有放回节点采样；后续每个 true epoch 将实际入选节点打乱后逐一遍历，最后不足 batch 的部分也计入。batch 总量为 51200，实际 updates/epoch 由节点数重新计算；continuation 从本次最佳权重开始，重新初始化 Adam。

hash50_sampled 当前最佳：0/50 个拟合案例应力 L2 <1%；最优 10_80/hor（16.77577%），最差 101_479/tor（39.02995%），case均值 hotspot L2=17.63454%。

按 load 的逐案例平均应力 L2：dia: 27.70239% (12 cases), hor: 20.91740% (13 cases), tor: 29.88464% (12 cases), ver: 19.89003% (13 cases)。

hash50_full1000ep 当前最佳：0/50 个拟合案例应力 L2 <1%；最优 10_80/hor（8.50951%），最差 101_479/tor（28.96709%），case均值 hotspot L2=11.57047%。

按 load 的逐案例平均应力 L2：dia: 17.99758% (12 cases), hor: 11.49243% (13 cases), tor: 23.76915% (12 cases), ver: 12.13304% (13 cases)。

## 静态科学图

- [stress_cases_01_00.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/stress_cases_01_00.png>)
- [stress_cases_01_01.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/stress_cases_01_01.png>)
- [stress_cases_02_02.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/stress_cases_02_02.png>)
- [stress_cases_05_03.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/stress_cases_05_03.png>)
- [stress_cases_50_04.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/stress_cases_50_04.png>)
- [learning_curves_01.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/learning_curves_01.png>)
- [learning_curves_02.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/learning_curves_02.png>)
- [learning_curves_03.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/learning_curves_03.png>)
- [learning_curves_04.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/learning_curves_04.png>)
- [learning_curves_05.png](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/learning_curves_05.png>)

## 代表性场图

同一案例的 GT / prediction 使用共同 signed color scale；误差图为 absolute residual / true RMS，颜色与指标均不截断。仅绘制已完成且与正确 prepared 节点顺序、标签吻合的实验。

- [Coordinate MLP (4 x 256 ReLU) — 506_533/ver，stress L2=39.51514%](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/fields/mlp_single_exact__506_533_ver.png>)
- [Official TCNN FullyFused (12-level hash) — 506_533/ver，stress L2=0.41394%](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/fields/tcnn_hash_single_exact__506_533_ver.png>)
- [FNO 192^3 (stress only) — 506_533/ver，stress L2=14.34100%](<C:/Users/ASUS/Desktop/GeoBC Neural Operator/plots/all_methods/fields/fno192_stress__506_533_ver.png>)
