# Transolver与FNO：当前可比结论

**当前配方首选FNO；继续Transolver时首选++。** 这来自相同4800次训练更新后的新CAD评估，最终solver仍待更广验证。Hash只在单／少样本拟合领先，不作为本次泛化选择依据。

五种方法均在同40CAD×3载荷=120病例训练，seed2026，原TRAIN-only scaler、全节点signed U(mm)/S(MPa)。独立固定五CAD/ver全部与本轮TRAIN40、新增160、DEV20互斥；每法预测全1,080,435个原节点，25份预测提交完后才共享读五份应力。历史位移已知，另用缓存独立读出，不重跑模型。均值为五个案例相对L2的等权平均：

| 方法 | 五CAD位移误差 | 五CAD应力误差 | 训练分钟 |
|---|---:|---:|---:|
| **FNO** | **20.6%** | **55.1%** | **20.11** |
| Transolver++ | 25.9% | 74.5% | 68.55 |
| 原版Transolver | 28.0% | 81.1% | 23.34 |
| Transolver-3 | 29.2% | 83.0% | 73.37 |
| MLP | 30.9% | 73.6% | 9.84 |

FNO的U／S最差为27.9%／64.1%，++为39.7%／77.8%；热点S为47.4%和80.5%。FNO在五个S案例上全部领先。MLP应力均值略好于++、位移较差，可作低成本参照。应力预测与读出总19.95秒，独立已知U读出2.07秒；模型已加载和输入已缓存，这些耗时不是完整服务基准。

需要分开两个瓶颈：同ver载荷的TRAIN40中FNO U／S为7.9%／35.1%，五CAD20.6%／55.1%，训练拟合残差与CAD外差距并存。++的TRAIN40/ver应力74.6%、五CAD74.5%均偏高，末段loss仍下降11.6%，更支持拟合／训练制度瓶颈；不能把未拟合好解释为已过拟合。两组CAD难度未控制，不能直接因果归因泛化。完整TRAIN120诊断FNO U／S为14.0%／34.9%，++37.6%／77.6%，600/600次预测实际4分43秒；FNO对全部40CAD和120载荷的U／S／热点均优于其余方法，非少数案例驱动。见[TRAIN紧凑报告](geobc_train_fit/BRIEF.md)与[CAD配对解读](../notes/geobc_TRAIN_fit_interpretation.md)。

这是当前配方效果，而非架构家族上限：参数量FNO14.18M、原版1.85M、++4.49M、3 7.65M、MLP0.23M；FNO用节点splat的64³网格，其他方法直接处理点，全局几何context固定。统一Adam常数学习率并非三版作者各任务的完整训练制度；旧slides的25特征schema和部分原配置不可恢复。五例包含两官方TRAIN、三官方TEST，但全五不参与本轮训练。CSV字节／坐标／U曾读，只称**U-known/S-field-blind**，单seed、ver单载荷、不替代最终测试或冻结20DEV。

下一步只保留两条重点：Transolver++作者训练制度确认，以及FNO应力泛化的宏观改动。先冻结目的／预算／端点，不能按已打开五例做earlystop或超参挑选；不继续Hash微调。当前未释放新增训练。配方范围见[官方配置分析](../notes/transolver_recipe_scope_next.md)。

正式40／200CAD×五方法协议仍冻结，SCI `bcce7d422749ca26b72ecba57ddcbbb985dcd3643ac77cf87f5671310b1aa01b`和20源码pins保持。新增原始文件完整317/540、缺223，Drive/HF连通故障未恢复；下载完成时间未知。新20DEV仍封存，五例评估不能替代十臂/60病例比较，也不能把输入故障记为求解器失败。40／200规模同时改变覆盖与每例曝光40／8轮，未来不能单独归因数据量。见[冻结协议](../notes/geobc_coverage_comparison_plan.json)与[真实网络故障](../notes/geobc_drive_transport_incident.json)。

物理FE此前五CAD位移0.0162%，但依赖完整TET10网格／材料／真实BC，不是学习架构公平排名；原CSV signed scalar到物理张量的导出语义缺失，不能据U准确宣称完整U／S已解决。见[原应力源语义](../notes/deepjeb_signed_scalar_source_semantics.md)。

[末段训练曲线与同ver参照](../notes/geobc_fixed4800_training_regime_evidence.md)仅分析已保存标量；后段梯度与逐步LR未记录，不能声称任何方法已充分收敛。
