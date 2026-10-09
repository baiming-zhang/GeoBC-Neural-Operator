# 真实力学算子：输入基础与缺口

2026-10-10，SOURCE／已有小元数据可行性记录。没有读取原deck、科学数组、Field、权重或任何封存数据，没有准备输入、构造模型或运行GPU。本记录不授权第二个并行求解器任务。

`tools/parse_deepjeb_deck.py` 已保留SPC节点／DOF／值及SPCADD展开，已有45CAD记录确认同一CAD各载荷共享四个参考节点的123456零约束。保留RBE2参考／成员连接与DOF；RBE3保留参考与成员DOF／权重，但当前只支持单组weight=1、memberDOF=123。FORCE／MOMENT参考节点、向量及LOAD展开、MAT1的E／ν和PSOLID材料一致性均有已有元数据。单位是mm／N／MPa；cells10及原顶点可用于曲线TET10映射。

**仍缺完整力学契约。** GRID CD坐标DOF、CTETRA→PSOLID全映射、积分选项、其他控制／约束形式尚未完成覆盖证明；真实RBE2／RBE3约束矩阵、载荷转移与消元没有实现。RBE3是参考运动的加权插值约束，不能把参考力／力矩直接均分给成员，见[OptiStruct官方定义](https://help.altair.com/hwsolvers/os/topics/solvers/os/rbe3_bulk_r.htm)。support-reference标签也不能当作成员节点Dirichlet。原有V/10质量代理和角点体积不能代替刚度积分。

若下一次黑箱结构仍无学习机会，最小后续问题是一个固定TRAIN101_479的输入-only有限可行性：卡片覆盖、坐标DOF和TET10顺序／积分，K对称性与刚体模态、约束秩、载荷转移后的合力与合矩守恒。需要先冻结具体规则才读取此输入；不读取Field、不用标签优化。通过后再考虑variational或约束消元求解器，当前不宣称已经复现完整线弹性系统。

原版DeepJEB的有符号标量stress出口公式仍不完整，见[语义记录](geobc_stress_semantics.json)。不由这个标量反推tensor strain或假造本构监督。
