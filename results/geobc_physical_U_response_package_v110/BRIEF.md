# 冻结物理位移响应包的单次加载

整阶段外UTC8.021秒／monotonic12.649秒，原时轴差异保留、不缩放，清单加载门：PASS。

`load_response(d, np, torch, descriptor, reviewed_digest, guard)`单次加载完整几何与六个收敛基解，返回可直接调用的`query(xyz_mm, force6)`对象。descriptor身份必须来自独立冻结的计划／根审，不依赖调用者清单的自称digest。

| CAD | 完整节点／单元 | 清单加载总秒 | API构造／索引构造秒 | 查询／拒绝／分块秒 | 同场最大相对差 |
|---|---:|---:|---:|---:|---:|
| 120_153 | 188196／117554 | 1.101 | 0.210／0.210 | 0.522 | 0 |
| 131_345 | 267563／167316 | 1.017 | 0.003／0.056 | 0.315 | 0 |
| 131_351 | 239956／150574 | 0.806 | 0.003／0.051 | 0.319 | 0 |

每CAD一次geometry NPZ（仅vertices／cells）与六份FP64物理位移字段各严格hash／load一次；三CAD总3＋18次科学body读取。九份仅元数据的unit／field identity／basis order篡改清单，即使重新生成自称digest，仍在科学body／构造前被原reviewed digest拒绝。原科学数组不修改、不复制到报告或私人仓库；packages目录只含带原本地路径的元数据。
固定新4几何点／CAD、三六维mix，共36位移值、12owner和3明确整批外点拒绝，先完成有限hash／body／Cartesian屏障，之后9次显式单元同物理API对照。完整同场值／分块／定位保持；只继承v109真实耦合坐标／力矩导数证据，本阶段不重复梯度测试，GT0。
API／index计时通过scoped profiler观察原构造函数并同步CUDA，不改数学实现。加载总时间还包含metadata校验、hash、IO和GPU staging；剩余项不是纯磁盘IO。所有import／构造／查询／保存／屏障／对照／退出计入全阶段，双时轴和显存实测见JSON。
**这是已有基解的缓存加载，不是新几何全冷物理解算，也没有新的OptiStruct离节点／新载荷GT精度证据。**继承源、K与六基解成本独立列出，未知完整冷部署成本保留null；原20CAD规范U精度只继承旧证据。
固定geometry／material／zeroSPC／RBE参考点六维力矩；近似curvedP2／四Q／weighted-LS RBE3与未知DEBUG／PSOLID局限保留。定位为有界单初值逆，不保证任意曲面全局唯一逆／C1／shape-gradient，文件名隔离未证明shape-family；signedS仍未提供，完整U／S求解器未选定。
