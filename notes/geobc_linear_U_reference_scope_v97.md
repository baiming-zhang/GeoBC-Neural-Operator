# 新载荷位移参考的适用范围

v96只证明真实接口一致性和旧三TRAIN CAD的规范预测保持，未测新载荷GT误差。下一阶段固定20个已迭代knownDEV CAD，比较三个冻结20遍骨干；不增加训练、不挑载荷系数。

不能仅凭ANALYSIS=STATICS认定原问题线性：该选项也包含非线性准静态gap分析。[Altair ANALYSIS定义](https://2022.help.altair.com/2022/hwsolvers/os/topics/solvers/os/analysis_sub_r.htm)。MAT1用于线性各向同性材料的官方示例见[线性静力教程](https://2021.help.altair.com/2021/hwsolvers/os/topics/solvers/os/linear_static_analysis_plate_hole_r.htm)。材料非线性、接触、随动载荷和大变形可以破坏固定刚度关系，见[非线性准静态分析](https://2022.help.altair.com/2022/hwsolvers/os/topics/solvers/os/analysis_nonlinear_quasi_static_c.htm)。GAPPRM GAPCMPL=NO本身不证明无接触。

实际执行前逐CAD保存完整源卡清单，核实活动静力工况、显式MAT1、三载荷同几何材料及零约束、实际FORCE/MOMENT与缓存特征一致，拒绝当前解析器不支持的材料／接触／gap／随动／非线性卡，保留所有忽略的控制设置。来源门不通过不得换CAD、按子集排名或解锁GT。此处源码设计不冒充这些20个源文件已完成数值检查。

在固定K、零约束和线性小变形假设成立时，由原三个规范位移标签构成 U_ref(c)=sum_i c_i U_GT,i。它是条件线性静力派生参考，**不是独立重新求解的新载荷FEM真值**。原DEBUG LGSHPSNS与PSOLID数值设置仍未完整重建，即使来源门通过也保留该条件与数值保真限制。只在实际三列力／力矩的秩3子空间内使用；不是六独立载荷方向或离节点FEM精度证明。

所有540份完整原节点预测先不可变提交、hash/body/Cartesian屏障通过，才共享读取既有原始physical GT60一次。规范载荷的U/S精度用原GT评价；混合载荷仅评价上述派生U参考。原signedS使用原模型输出，不产生应力叠加、幅值或符号伪参考。canonical预测保真、条件mixed-U指标与物理一致性分别报告。
