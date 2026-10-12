# 连续XYZ＋六维力矩＋四支撑平移的物理U接口

固定两CAD／48查询／16owners／2盒外拒绝／4实际耦合梯度点：PASS。外UTC 22.501809秒／mono 18.157835秒。

| CAD | 构造／读取秒 | 全CAD索引秒 | 查询＋拒绝＋chunk秒 | 最大同场相对差 | 坐标／力／矩／支撑梯度相对差 |
|---|---:|---:|---:|---:|---|
| 50_380 | 2.2803 | 0.1417 | 5.1327 | 0 | [0.0, 0.0, 0.0, 0.0] |
| 624_483 | 1.7979 | 0.0230 | 0.6008 | 0 | [0.0, 0.0, 0.0, 0.0] |

真实接口仅接受physical XYZ(mm)、force6(前三N／后三Nmm)及四支撑平移(mm)，不要求caller单元IDs。全CAD保守Bezier桶定位在no_grad离散分支执行，之后真实curvedP2 Newton逆映射、18物理基解插值和力／支撑系数保留autograd图。四个支撑转角仍为0。
每CAD8个全原单元linspace位置以新bary [.25,.30,.20,.25]生成geometry-only XYZ，fixtureIDs只生成坐标，不传facade／locator。三固定请求共48U、16owners及2真实全Bezier盒外点先finite/hash/body/Cartesian屏障，随后8次同物理API显式单元对照。额外两个8有效＋1无效混合批次整批返回None，未预测子集。
实际4点分别验证XYZ坐标、前三force N、后三moment Nmm、十二support mm梯度，全部非零有限，按各单位分别比较直接接口，不用mm／rad／N的混单位L2假精度。chunk／同场／owner与输入和API基解GPU字节保持核验。不是旧192点梯度重播。
两own NPZ仅读vertices／cells，36物理force＋support基解各严格singlehashload；新索引构造、API、查询、梯度、屏障保存和退出均计入实际双时轴，首次CUDA开销不归因几何。旧源到force6、port0与剩余9基解成本分别继承；加载既有基解不是新几何全冷启动，未知完整工程cold为null。
**同场值／梯度保持是接口数值证据，未测原OptiStruct新载荷、支撑、旋转或独立offnodeGT。** 近似curvedP2四Q／weightedLS RBE3与未知DEBUG／PSOLID限制保留，两CAD已揭盲，不证明C1、shape梯度、任意位置全局逆映射或工程资格；signedS仍未解决。
