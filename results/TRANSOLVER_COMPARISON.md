# FNO与Transolver：目前如何选择

**当前证据支持FNO优先，但最终求解器未选定。** 学长没有指定Transolver版本；当前TS系列以++为主，下一轮检验原生节点上下文，保留这条路线。

同R64输入、native25查询头、约4.53M参数、TRAIN40 CAD×三载荷，分别延长各自完整4800检查点至12000更新；在同开发20 CAD×三载荷、全原节点上比较：

| 方法 | 位移相对L2 | signed应力相对L2 |
|---|---:|---:|
| FNO | 26.50% | 55.55% |
| Transolver++ | 52.35% | 83.99% |

FNO联合误差在20/20 CAD更低，均值领先27.14个百分点；60例推理约2.16秒对16.07秒。读出完整120份预测先提交再共享已知GT60一次，实际97.15秒。[完整同曝光对照](geobc_short_screen/BRIEF.md)。

**继续延长当前++的收益不足。** 同100遍TRAIN120的FNO U/S17.39%/30.67%，++52.66%/81.89%；++位移比自身40遍更差，低／中位移病例恶化较多，大位移病例改善。联合目标的权重原因尚未做因果验证，不能只归因于架构或换指标宣布改善。原双24000计划已主动终止，全部日志和检查点保留，不称完成。[TRAIN中间证据](geobc_exposure_comparison/12000_INTERIM.md) · [误差分布](geobc_exposure_comparison/MIDPOINT_DIAGNOSTIC.md)。

**下一步改为低成本表示筛选。** 三方法FNO、原生节点Transolver++、整体几何条件化Hash从fresh同seed开始，每方法10遍／1200更新、训练上限30分钟、30.5GB显存。三者初筛及完整开发60例已完成，详见末尾最新40遍确认；初筛原生++拟合差且慢，暂降低投入优先级。Hash容量／TCNN精度不同，如实报告。

较早九候选比较的最佳FNO U/S27.1%/53.4%、最佳++39.0%/79.4%，不同表示／容量；此次延长训练未取代那个略低的联合误差记录，不围绕小差距微调。[历史九候选结果](geobc_available_cad_pilot/BRIEF.md)。Hash少样本stress-fit低于1%不能和新CAD联合U/S误差相比。

开发20 CAD来自可用官方TRAIN池，属于已知开发集、单seed，非最终盲测；reserved FINAL数值标签未读。原版CSV按GRID对齐H5 vertices，signed U(mm)/S(MPa)保持；撤回H5-only不引用。TET10／AD-FD连续查询一致性不等于独立离节点FEM准确性，原stress符号及恢复公式未知。当前结果不能宣布Transolver架构上限。

[更早比较及失败的完整历史](TRANSOLVER_COMPARISON_HISTORY_20261011.md) · [当前状态](../README.md) · [私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)（最新已验收7d8d556；本次自动交接及简报整理待下一实质结果一起发布）。

最新40遍低成本确认：FNO U/S **27.93%/56.06%**，几何条件Hash **49.19%/64.96%**，总阶段4.93/3.43分钟。两者初期高误差明显改善，FNO在20/20 CAD的U/S更低；原生++10遍较差且慢，暂降低预算优先级。下一步检验FNO的100CAD覆盖，不死磕小参数。[最新两行报告](geobc_two_warm_screens_v3/BRIEF.md)。原三方法短筛与双warm40均已完成，新100训练尚未启动。
