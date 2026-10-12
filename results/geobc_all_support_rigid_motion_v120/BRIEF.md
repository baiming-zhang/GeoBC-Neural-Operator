# 四个真实RBE2支撑转角与30方向物理U响应

固定两CAD／30新独立场门：PASS。实际外UTC 506.394500秒／mono 462.097119秒。

| CAD | 真实参考坐标扫描秒 | 新十二旋转基解秒 | 三独立混合秒 | 全N缓存更新秒 | 混合U最大相对差 | 刚体小转角相对差 |
|---|---:|---:|---:|---:|---:|---:|
| 50_380 | 1.018 | 194.990 | 50.434 | 0.00634 | 3.81e-08 | 2.69e-07 |
| 624_483 | 1.129 | 179.295 | 46.432 | 0.00317 | 1.68e-08 | 1.63e-07 |

每CAD原deck严格hash一次、目标无关卡片流扫描一次。四支撑参考XYZ来自真实GRID／CP=CD=0及GRDSET展开，RBE2 CM123456和成员GRID与原缓存region_support_0..3精确一致；不使用成员中心／定位投影或虚构reference H5行。全卡片计数、原未知DEBUG效果、原支持范围与源身份保留。
每个真实端口新增xyz三方向 .0001rad旋转、外力0，两CAD共24新旋转基解；再六个declared force6＋四端口平移／转角混合解。十组三RHS均独立零自由初值FP64 PCG，每路alpha／beta／真实残差停止独立，原fineK与32步预条件器不改。
几何lift=t_mm＋theta_rad×(vertices_mm−真实reference XYZ_mm)，free0；RHS=b_free−(K·lift)_free，以这个真实非零RHS归一化。完整物理u保留非零固定lift，不以原基解warmstart或大罚刚度近似。每25步原子保存z/r/p/lift/rhs/4×6目标／真实XYZ／端口与成本。
完整30场／真平衡／四端口反力与力矩先finite/hash/body/Cartesian屏障，之后原12force＋24translation基解各一次共享读取；24新旋转基解从屏障常驻复用，未参与独立mixed warmstart或RHS。真实30方向API提供force6＋四端口motion6(mm/rad)到全原节点物理U(mm)。
端口F_N=sum固定反力，M_Nmm=sum((x−真实referenceXYZ)×反力)。bodyenergy=0.5外部nodalforce·u＋0.5Σ(t_mm·F_N＋theta_rad·M_Nmm)；整体力／矩平衡包括所有固定反力，不混单位，也不把未知支撑参考坐标或局部力矩填0。
无外力、四端口相同小转角，平移按theta×(referenceXYZ−加载参考原点)设置，实际缓存响应与全N theta×(vertices−原点)比较；这是线性化刚体转角数值兼容，不是有限旋转／新FEM GT或近零能量归一化。
**两CAD已揭盲，原OptiStruct新支撑／载荷／转角或独立offnodeGT未测。**近似curvedP2四Q／weightedLS RBE3与未知DEBUG／PSOLID限制保留，signedS仍未提供，未证明任意支撑选择或工程资格。
新24旋转基解、6独立混合、真实source XYZ扫描、cache/API/readout与继承旧force6／translation12成本分列。全部import/hash/IO/资格/PCG/checkpoint/屏障/对照/退出计入真实双时轴，不缩放差异，不将并行lane时间相加；既有K／基解缓存不是新几何工程全冷或NN等预算排名，未知完整engineeringcold为null。
