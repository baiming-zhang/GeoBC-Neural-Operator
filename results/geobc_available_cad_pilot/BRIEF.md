最优 eligible：FNO grid25。最佳 TS：Old PP。
最佳相对最佳 TS：均U差 -11.93、均S差 -26.07 个百分点（负值表示误差更小）。
推理成本差 -8.703 秒，继承训练成本差 -43.35 分钟。

| 方法 | 均U | 均S | 训练分钟 | 推理秒 |
|---|---:|---:|---:|---:|
| 最佳 eligible：FNO grid25 | 27.1% | 53.4% | 25.2 | 2.047 |
| 最佳 TS：Old PP | 39% | 79.4% | 68.55 | 10.75 |
| 旧 FNO：Old FNO | 26.8% | 55% | 20.11 | 2.232 |

真实准备/readout/campaign外跨度（epoch / monotonic / 两轴差，秒）：准备 595.2 / 121.3 / 473.8；readout 479.1 / 441.6 / 37.54；campaign 1074 / 563 / 511.4。成本exact=True，协议scientific qualification=True。
UTC全阶段区间：准备 2026-10-10T16:55:37.955027+00:00 → 2026-10-10T17:05:33.127208+00:00；readout 2026-10-10T17:05:33.142245+00:00 → 2026-10-10T17:13:32.276692+00:00；campaign 2026-10-10T16:55:37.955027+00:00 → 2026-10-10T17:13:32.340823+00:00。
固定准备/readout各3600秒、campaign7200秒，epoch与monotonic均未超限才有预算资格。native各phase仅保留原monotonic耗时；未记录phase UTC成本，不按总跨度缩放或分摊。
输入negative=False；真实终态清单540/540；共享GT原CSV解析20/20次，U/S病例读取60/60（期望各60）；oldDEV/final/originalnewDEV20/dia/strain/H5 nodal读取均0。
终态原因：固定流程终态；详见真实失败与外成本记录。
完整九候选、hot/worst、失败、参数、曝光和每载荷20/20CAD配对差：[九行 CSV](metrics.csv) · [详细 JSON](metrics.json) · [标量比较图](comparison.png)。图状态：completed。
这些20 CAD按缓存可用性与字符串排序选自官方TRAIN池，仅对old40训练候选提供独立候选pilot；不是原封存20DEV或无偏最终test。以后进入200TRAIN即失去held资格。TRAIN曝光、容量、优化制度及成本不同，不能据此宣称架构上限。
原source540/input660/prediction2400、20源码pins及317/540 acquisition input-negative保持原样。
