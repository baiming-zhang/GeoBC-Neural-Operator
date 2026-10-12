# 四个实际源支撑平移端口与加载参考点六维力矩

固定两CAD／24新独立场：PASS。实际外UTC 391.923769秒／mono 359.586961秒。

| CAD | 真实四支撑reference GRID | 剩余九基解秒 | 三独立混合秒 | 全N缓存更新秒 | 混合U最大相对差 | 刚体平移相对差 |
|---|---|---:|---:|---:|---:|---:|
| 50_380 | [213002, 213003, 213004, 213005] | 146.132 | 49.199 | 0.01070 | 4.14e-08 | 2.05e-07 |
| 624_483 | [188002, 188003, 188004, 188005] | 131.895 | 47.369 | 0.00150 | 1.69e-08 | 1.73e-07 |

实际源RBE2按reference GRID排序对应region_support_0..3。每个端口真实fixed成员和已存在master输出行均核验；四端口互不交叠、并集恰等于原fixed分区，转角始终0，未生成缺失reference节点。
仅剩余端口1／2／3各三方向(.01mm)新求解，旧port0不重跑。每CAD九新支撑基解及三四端口＋force6混合解，共八组三RHS／24独立零初值FP64场；每路alpha、beta、真实残差停止独立，固定32步原预条件器及原fineK不改。
几何lift在真实各fixed端口赋对应运动、自由行0；RHS=b_free−(K·lift)_free，用此真实非零RHS归一化，包括纯支撑运动外力0的情况。完整物理u保留非零fixedlift；每25步原子保存z/r/p/lift/rhs/四端口目标与成本。
全部24场、真平衡和四端口反力完成finite/hash/body/Cartesian屏障后，12旧force6基解与6旧port0基解各一次共享读取。新18基解从屏障常驻复用，未用于独立混合warmstart或RHS。实际18方向API由force6和四支撑translation4×3求全N物理U。
能量包含0.5外部nodalforce·u＋0.5所有支撑motion·reaction，整体平衡纳入全部固定反力。局部master力矩没有真实参考XYZ时明确null。四支撑同.01mm x平移、无外力的刚体见证是数值兼容性，不是新FEM真值精度。
**两CAD已揭盲；这是近似curvedP2／四Q／weightedLS RBE3的数值响应，未知DEBUG／PSOLID仍保留。** 未测原OptiStruct新支撑或载荷GT，不证明任意支撑选择、支撑旋转或工程资格，signedS未解决。
剩余18基解、6独立混合、cache／API成本分别记录；继承旧force6源到基解77.94036／63.43762秒、旧port0基解34.76535／34.90065秒，均不重跑。全stage包括import/hash/IO/资格/PCG/checkpoint/屏障/对照/退出；双时轴不缩放、未知工程全cold为null，缓存接口不是NN等预算排序。
