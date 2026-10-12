# 实际RBE2支撑平移端口与加载参考点六维力矩响应

固定两CAD／12新场门：PASS；外部UTC 159.313916秒／mono 146.115542秒。

选择原源RBE2中reference GRID最小的支撑，三平移非零、转角0，其他三支撑仍0；加载RBE3六维力矩保持独立。端口来自真实源reference排序和region_support_0成员，不按文件名或误差选取，H5没有master行时不生成替代节点。

| CAD | 支撑reference／成员行 | 新三支撑基解秒 | 三独立混合解秒 | 九方向更新＋全N秒 | 混合U最大相对差 |
|---|---:|---:|---:|---:|---:|
| 50_380 | 213002／440 | 34.765 | 41.803 | 0.01462 | 1.02e-08 |
| 624_483 | 188002／440 | 34.901 | 34.672 | 0.00099 | 4.08e-09 |

每CAD三新支撑单位基解(.01mm)的外力为0，再三个source force6＋支撑运动的独立混合解。四组三RHS各独立零初值自由未知量、FP64／每路alpha beta残差停止，真实RHS=b_free−K·lift_free；纯支撑运动用非零−K·lift归一化，不用零外力范数。每25步原子保存z/r/p/lift/rhs/目标与成本。
实际完整物理场保留固定节点的非零平移，再填自由位移，不用旧零边界restore丢掉lift。真实region0所有成员属于原fixed、与其他regions无交叠；master仅已有H5行才加入。实体几何、材料、free/fixed分区、fineK与原32步预条件器保持；不重建K／diag／粗矩阵或重播旧8guards。
完整12新字段／真平衡／支撑反力先finite、hash、body、Cartesian屏障，之后12旧六力矩基解各一次；新支撑基解没有用于独立混合解warmstart或RHS。真实九方向API=sum(forceB⁻¹f·旧6基解)+sum(支撑平移/.01·新3基解)，固定六混合场全原节点对照。
bodyenergy=0.5外部nodalforce·u+0.5所选supporttranslation·portreaction，其余不动支撑做功0；平衡含所有固定支撑反力。局部支撑master力矩没有实际参考XYZ时明确null，整体力矩仍关于已有加载reference XYZ；不假称新支撑六旋转方向已支持。
**这是声明的新非零支撑方程及多端口数值响应，不是原OptiStruct新支撑／载荷GT。**近似curvedP2四Q／weightedLS RBE3及未知DEBUG／PSOLID效果保留，两样本已揭盲，不证明任意支撑选择或工程资格，signedS仍未提供。
新三支撑基解、独立混合验证、cache／API时间与继承源到旧六力矩基解77.94036／63.43762秒分列。所有import、缓存hash／IO、资格、PCG、保存、屏障、对照、失败退出计入双时轴；未知完整工程冷启动null，缓存接口不是NN同预算排名。
