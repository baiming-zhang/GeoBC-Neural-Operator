# GINO：有界模型资源通过，尚无训练或精度结果

固定TRAIN101_479/ver实际任务 completed / feasible，4.985秒、8/8 checks PASS：全部179,316点encode为2.003秒，512原节点query为0.0437秒，一次无目标backward为0.3004秒；另有4次小真实CSR后端autograd核对。实测Torch peak allocated/reserved为1,829,223,424 / 3,812,622,336 B（1.829 / 3.813 GB），零labels、零已存权重、零optimizer更新。此资源范围是全输入/512输出，首例全N输出训练仍未验。实际参数1,433,277＝PointNet16,736＋official GINO1,416,541，complex tensor按numel计；与旧FNO约14.18M的容量及频率实现不同，不能归因单GNO组件。尚无训练、准确度、BC满足或连续场证据，新test字段仍封存。

来源：[实际小summary](../../runs/geobc_exploration/gino_native_resource_101_479/summary.json)。本报告仅读该JSON，未执行模型、数组、GPU或新检查。
