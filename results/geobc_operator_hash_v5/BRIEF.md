# 全局几何／BC控制连续Hash：是否比现有方法更好

同100TRAIN CAD×三载荷、4800更新／16遍、seed2026、old40-only scaler；knownDEV20×三载荷全原节点，未读FINAL。

| 方法 | 位移误差 | 应力误差 | 60例推理秒 |
|---|---:|---:|---:|
| FNO | 28.59% | 56.49% | 1.51 |
| Geometry Hash | 41.27% | 61.61% | 0.91 |
| Whole-BC operator Hash | 29.86% | 54.01% | 1.00 |

同4800联合评分最低：Whole-BC operator Hash。当前更充分曝光的FNO100／12000更新23.17%／50.85%仍单独作主候选参考；不与4800冒称相同曝光。

全局Hash复用FNO完整几何／BC／材料空间latent，Hash配置沿用原参数。总容量和混合精度不同，不能称纯容量匹配消融；真实初始化／最大全节点梯度与资源均在训练预算内核验。

两个推理消融只将解码head的16维latent或32维几何descriptor置零，编码器／算子及训练权重保持；geometry仍可能通过其余路径存在。它们是分布外依赖诊断，未重训、不参与求解器选型。
operator_latent_zero：位移／应力误差变化+53.86／+45.99个百分点。
geometry_context_zero：位移／应力误差变化+5.57／+10.29个百分点。

成本JSON保留初始元数据故障、原失败126秒、恢复尝试和原campaign全部wall（含修复间隔），原绝对预算不重置。FNO4800为既有中间端点，隔离完整成本unknown=null，不冒用12000成本。

240份有限预测先保存／重验，再knownGT60共享一次；原CSV／FINAL／训练更新／新scalerfit均0。物理U(mm)和signed S(MPa)保留；原节点精度与连续查询接口一致性不代表独立离节点FEM准确性。
