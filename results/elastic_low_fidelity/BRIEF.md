# Direct masked-Galerkin coarse：两例低保真筛选

固定门：FAIL；零GT，旧fine精度未转移。
| CAD/ver | coarse真实残差 | 相对fine数值U差 | RMS/max mm | accounted cold / 上限s |
|---|---:|---:|---:|---:|
| 101_428/ver | 9.95149e-09 | 0.0595411 | 0.00475273/0.0189225 | 32.9744/21.0539 |
| 102_249/ver | 9.86629e-09 | 0.0776472 | 0.0154452/0.0494459 | 35.0266/18.246 |

5%为对原fine近似模型的数值差距，非CSV GT误差。fine残差与PT投影残差分别在metrics.json；不要求fine1e-8。
计费cold组成账本包含继承setup与新IO/求解/恢复/验证/保存/数值参考，不是fresh联合cold；cached实际延迟另列。
完整TET10/约束物理接口不同于学习点云25接口；本候选没有source signed-S，不作完整solver排名。
缺失/预算/后端失败关闭固定候选，不换CAD、救容差或二次回放。
