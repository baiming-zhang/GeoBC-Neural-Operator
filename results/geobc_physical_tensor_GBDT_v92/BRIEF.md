# 物理张量＋GPU树模型：路线对照

**旧三个TRAIN CAD／两个knownDEV CAD／ver，全部原节点；GPU树模型应力误差30.92%，旧物理＋MLP31.76%。热点改善明显，但预声明门仍FAIL，关闭固定配置。**

| 同旧两开发CAD | signed应力误差 | 热点误差 |
|---|---:|---:|
| 物理＋MLP（冻结标量） | 31.76% | 53.80% |
| 物理＋GPU树模型 | 30.92% | 36.57% |

应力-0.84、热点-17.23个百分点。两例应力分别34.51%／27.34%，每例须<25%、热点平均<25%的门：FAIL。

完整阶段19.90秒，GPU训练9.52秒，固定400轮；683842个TRAIN节点、五例1092529个节点。实际首轮配置CUDA0／GPUhist，编译信息未替代真实运行。保存模型复载后UBJ模型身份一致，再做全节点预测。

物理输入的历史GT-free求解／应力恢复仍需每CAD约77–101秒，不是毫秒求解器。本次只复用缓存，没有新FEM求解；缓存增量、模型加载、组装cold成本、共享初始化分别保存，未重新测量联合freshcold。GPU独占不保证，不能把墙钟差当孤立架构加速。

输入来自原GT-free有限元解的chosen sigma6恢复proxy，使用旧TRAIN3-only归一化＋validity；原signedS MPa监督，原公式／符号不猜测或改写。输入7维、rawMPa目标、权重／优化器与旧67维asinh MLP不同，不能归因仅树学习器。

直接回归原S、每病例S²能量倒数加权并均值归一为1；400轮／depth6／eta.05／bins256／seed2026固定，没有DEV早停、截断、q范围约束、伪tensorGT、材料拟合或符号规则救结果。分位cut仅TRAIN，DEV矩阵ref TRAIN。

两个DEV的全部有限不可变预测先保存、文件哈希与数组Cartesian重验，再各读取一次原缓存S；TRAIN_S3、DEV_S2、sigma/valid5各一次。原CSV/H5/deck/archives、物理U／GTU／PT、新20DEV／FINAL、scalerfit、新物理解算及旧基线重跑均0。

**这是历史3TRAIN／2knownDEV／ver路线比较，不是主100TRAIN／20knownDEV三载荷的公平FNO排名或最终盲测。** Tree输出分段常数不可微，不能当已实现连续hash算子的替代或独立离节点FEM精度。

依赖准备另计182.59秒：首次NCCL下载超时180.02秒，复用95MB缓存offline安装2.57秒，无再次网络下载。二进制／数据／模型主体留本地，只发布标量与源码。
