# 紧凑全局Transolver：资源与反传验证

**三个官方版本均通过两个TRAIN真实病例的全原节点反传；尚未训练，不能据此评价新几何精度。**

| 版本 | 参数量 | 最大325387节点单次反传wall |
|---|---:|---:|
| transolver | 1.88M | 0.076秒 |
| transolver_pp | 4.52M | 0.133秒 |
| ts3 | 7.68M | 0.150秒 |

完整阶段8.19秒，采样设备峰值2.97GB；输入两个TRAIN病例共享加载、原物理场各读一次，6次全节点forward/backward。无更新、无DEV／FINAL或新scaler。

全局attention一次处理全部4096个R16空间token，native25／geometry32／latent16解码器仍监督每一个原节点；没有分块attention或节点采样。初始化encoder／head对同seed原FNO构造器逐张量一致。

测试包含首调用开销，CUDA事件与epoch／monotonic原计时各自保留，未做换算或隐藏warmup。参数均float32、caller TF32关闭；未profile所有内部kernel。

新表示用原R64输入的occupied-voxel均值，可能损失薄壁和局部约束；插值使用真实block中心，边缘按最近中心延伸。它改变表示，不是与native或FNO R64的纯架构因果对照，也未测离节点FEM精度。

尚未加载完整300病例GPU训练缓存；正式训练入口必须在30.5GB下再次通过全缓存＋最大全节点梯度门。接下来原版／++／3同输入、同物理损失、同100CAD、同5遍短筛，当前实际训练0。
