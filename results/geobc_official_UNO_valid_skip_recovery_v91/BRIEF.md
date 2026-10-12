# 官方U-NO：双尺度场算子短筛选

**同三CAD×三载荷／100遍／900次全原节点更新；旧普通FNO仅复用冻结标量。原PointNet／查询头初始bytes、输入、物理联合损失、scaler保持；场算子更换为官方U-NO。**

| 场算子 | 位移误差 | signed应力误差 | 热点误差 |
|---|---:|---:|---:|
| 原FNO | 22.69% | 41.69% | 40.06% |
| 官方U-NO／64与32双尺度 | 31.40% | 56.21% | 50.67% |

新算子减旧基线：位移+8.72、应力+14.52、热点+10.61个百分点，联合+11.62个百分点。预声明TRAIN门：FAIL。

有效恢复执行63.42秒／学习51.27秒。原任务从首次启动到结束918.45秒，包含两次零标签／零更新接口失败和修复等待，原1200秒绝对截止未重置。跨WSL boot原monotonic生命周期未知，原记录保留；本次执行epoch/mono分别记录，不零填或缩放。

官方代码commit b307030d9b7ee530823d6a05bddb8238ba22a743／MIT。61输入→16latent；hidden16、层宽16/32/32/32/16、5层modes12/12/12、64→64→32→32→32→64，水平skip4←0/3←1；已有xyz输入所以禁用默认positional embedding。所有固定参数与源身份保存。

参数storage数4,709,636、real自由度9,354,500，含4,644,864个complex64参数；原FNO4,526,858实数参数。容量、谱模式约定、FFT/MLP/双尺度/checkpoint策略均不同，不是单因素归因或等容量对照。复数参数使用Adam fused=False/foreach=False，lr/betas/eps/weight_decay与旧基线相同；不声称等计算成本。

初次私有namespace缺绝对neuralop.utils，第二次soft-gating不支持变化通道；两次均在读取标签／资格／优化前失败，完整保存，不能当方法科学负结果。修复loader只别名同一官方utils，不改官方数学；有效配置将channelMLP skip改为官方测试采用的linear，显式新SCI记录，不能称科学配方完全不变。未恢复不存在的旧Adam/RNG/PT；依指定seed重建fresh0。

真实242936节点全encoder/head/官方operator梯度（abs-square包含实部与虚部）及context/latent leaves、实际64query坐标链和分块检查通过；资格model／Adam／四RNG保持，原encoder/queryhead初值未更改。新operator初始化与旧FNO函数不同，不作零扩展parity宣称。

**仍是九TRAIN场拟合，不是新CAD泛化。** 原物理9标签本次各读一次（前两次0），三个端点常驻复用27；编码／CSV／DEV／FINAL／父PT／scaler拟合0。不是原作者DeepJEB精确配置复现或独立离节点FEM精度。

通过门仅支持另冻结短100TRAIN宏观比较；失败关闭固定有效U-NO配置，无width/modes/scale/depth/seed/loss或DEV救结果，不用训练集选最终solver。
