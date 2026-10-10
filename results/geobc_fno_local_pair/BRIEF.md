# FNO局部条件表示配对：保存标量导出

两臂完成固定4800更新及全部TRAIN120读出。 native25与grid25的唯一局部因子是原节点features25或同一输入网格的roundtrip features25；新decoder其余结构、初始张量、曝光规则和scaler相同。

TRAIN表示推进门未通过，该相对结果不自动选择新decoder。应同时对照旧FNO4800的绝对U、signed S、top1%abs原S hotspot误差、最坏病例、成本与参数数。TRAIN120不是held20结果，也不提供总体泛化、家族上限或工程资格。

| 固定4800参考 | U均值 / 最坏 | signed S均值 / 最坏 | hotspot均值 / 最坏 | 参数 | monotonic训练秒 / arm秒 |
|---|---:|---:|---:|---:|---:|
| 新grid25 | 0.1813 / 0.3666 | 0.3263 / 0.418 | 0.3705 / 0.5273 | 14205716 | 1511.92 / 1726.51 |
| 新native25 | 0.1834 / 0.3603 | 0.3301 / 0.4191 | 0.3753 / 0.5718 | 14205716 | 1293.87 / 1490.86 |
| 旧FNO4800 | 0.1398 / 0.3328 | 0.3486 / 0.4387 | 0.3832 / 0.5596 | 14177284 | 1206.37 / 1343.57 |

新native25仍高于旧FNO的完整120指标：U mean、U worst、hotspot worst；即使配对门通过，也须保留旧FNO作为绝对参考，不能据此自动替换decoder。

新pair保存计时：native闭合monotonic累计 3229.401 秒；外guard launch至child退出monotonic 3229.887 秒，真实epoch/UTC 3513.355 秒，两轴差(epoch−monotonic) 283.4678 秒。预算要求两条外轴各≤9000秒、native累计<9000秒，各臂原monotonic训练≤3600秒；外timeout=False，exit=0，cost accounting exact=True。未知区间保留为nonexact成本。native各phase/arm/训练保留原monotonic值；phase/arm/训练的epoch/UTC耗时未记录，均为null，不按总轴差缩放。旧FNO表中wall是其保存的原训练attempt，读出成本单列于metrics，二者不冒充相同总计算预算。

同名初始张量完全一致=True；真实首个完整TRAIN反传通过（grid/native）=True/True。具体梯度、参数组、曝光、停止原因、预算和缺失原因见[metrics.json](metrics.json)。

图仅使用保存的loss及fixed0/4800 full120聚合；不重跑推理。loss图状态=saved，聚合图状态=saved。

导出产物状态=completed；protocol qualification=True；TRAIN scientific qualification=False。产物完整性不代表实验PASS。

[scalar CSV](scalars.csv) · [loss_comparison.png](loss_comparison.png) · [loss_comparison.svg](loss_comparison.svg) · [aggregate_comparison.png](aggregate_comparison.png) · [aggregate_comparison.svg](aggregate_comparison.svg)
