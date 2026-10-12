# GeoBC：原版DeepJEB求解器筛选

从几何、材料、载荷与约束预测位移和原signed应力，目标是新CAD泛化与连续空间查询。研究继续进行。

## 当前结论

**三种子对比后，默认采用普通FNO；Hash保留作热点对照，最终求解器仍未确定。**

| 同20遍／三seed／20个开发CAD×三载荷 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO | 21.72% | 49.33% | 37.53% |
| FNO + 条件Hash | 21.97% | 48.71% | 33.24% |

Hash应力只好0.62个百分点，联合误差好0.19个百分点，未达到预声明3个百分点推进门；热点好4.30个百分点。参数1225万对453万，新增seed学习6.39对5.90分钟，60场纯推理0.82对0.96秒。当前最好的Transolver同20遍／单seed位移35.79%、应力59.38%，仍落后；这些结果不代表架构上限。[三seed结果与成本](results/geobc_solver_seed_readout_v83/BRIEF.md)。

## 最近推进

几何点云规范化短对照已完成（45.48秒）：应力41.69%→40.72%，位移22.69%→29.57%，固定配置关闭。原符号／幅值联合头短验证也未通过门，不继续小调。[几何对照](results/geobc_canonical_shape_encoding_v86/BRIEF.md)。

位移真值的物理闭合诊断已完成（7.70秒）：固定恢复公式的应力幅值误差仍10.78%，旧有限元位移为10.77%。这部分差异主要不是位移求解造成；不能把此proxy当原应力精确定义，也不能当49%神经模型误差的下限。原signed标签保持，不猜符号规则。[诊断](results/geobc_reference_vm_closure_v87/BRIEF.md)。

下一项**仅设计，尚未实现或运行**：在FNO频域块内加入局部3D空间传播，先做原九TRAIN场／900次更新短对照，检验场表示结构，而非Hash、损失或训练时长微调。[设计](notes/geobc_local_spatial_operator_v88_design.json)。

## 数据与复现边界

原版DeepJEB v1.0，CSV按GRID坐标对齐H5 vertices，原signed U(mm)/S(MPa)、全部原节点，旧H5-only结果撤回。100 TRAIN/20 knownDEV/20 FINAL CAD各三载荷，CAD隔离，scaler仅旧TRAIN40。knownDEV已迭代且受来源可用性影响，三seed共享同20CAD，不是180独立病例；旧FINAL已揭盲，不再调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

30项可恢复slides实验已完成。原单／双已知场约0.1%是拟合；共享50场平均15.11%，FNO网格2.46%而原节点14.38%。原目录、部分25特征schema、网格原点和50样本集合缺失，不声称精确数值复现。原200CAD输入失败及旧++中断成本保留。[Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

连续查询／梯度接口通过，不等于独立离节点FEM精度；原应力后处理及signed-VM契约仍缺失。[来源语义](notes/deepjeb_signed_scalar_source_semantics.md)。

## 运行与交付

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，显存30.5GB SI，公共GPU锁；计划与真实终态绑定，不因旧status重复启动。短阶段5分钟监视，预计超过1小时改30分钟。[依赖](requirements-lock-wsl.txt)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore`，未知内容拒绝覆盖。[私人GitHub](https://github.com/baiming-zhang/GeoBC-Neural-Operator)发布代码、配置、标量报告、图和检查点索引；原数据及项目训练检查点留本地。

最新已验证私人提交 `b2c7a0f` 包含符号／幅值短验证及几何设计。本轮几何实际结果与物理诊断当前已验收，发布准备中；下一局部空间算子仍仅设计。[发布状态](notes/github_publish_plan.json)。
