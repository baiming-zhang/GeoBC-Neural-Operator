# GeoBC：DeepJEB 场预测与求解器筛选

输入几何、材料、载荷和约束，预测原节点及连续空间的位移与signed应力。使用**原版DeepJEB**；目标是能迁移到新CAD／新载荷的共享求解器。

## 目前结论

**当前以FNO为主模型，全局Hash保留热点对照；绝对误差仍高，最终求解器尚未确定。Hash已知单样本<1%是拟合结果，不能当新几何误差。**

同100训练CAD×三载荷、物理损失、5遍／1500步，在knownDEV20个新CAD×三载荷、全部原节点上的结果：

| 方法 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO | 31.61% | 56.10% | 45.08% |
| 全局Hash | 31.37% | 54.45% | 40.85% |

Hash联合优势0.94个百分点，未达预设推进门槛，停止追加Hash微调。[同5遍报告](results/geobc_global_hash_physical_v18/BRIEF.md)。FNO从自身5遍继续到10遍，位移／应力降到25.60%／52.14%，追加纯训练97.18秒；增加曝光量的结果不当同训练量比较。[5→10遍报告](results/geobc_physical10_v16/BRIEF.md)。

## 正在推进

三版紧凑Transolver同100CAD／5遍训练及新几何评估已完成，共用缓存整轮16.08分钟，DEV166.98秒。同5遍FNO U/S31.61%／56.10%；原版TS52.06%／93.20%、++55.26%／94.23%、3 52.81%／93.43%。原版是本轮TS系列最好，但仍明显落后；TRAIN应力也约94%，当前问题是拟合不足，不能宣称架构上限。[四方法简报](results/geobc_coarse_three_physical5_v21/BRIEF.md)。

相同R16输入的FNO控制已完成：训练整轮4.14分钟、DEV1.16分钟，U/S30.78%／64.98%；输入压缩本身不足以解释TS约94%的应力误差，算子／容量／配方交互仍未分离。[R16控制](results/geobc_coarse_fno_control_v23/BRIEF.md)。

R64 FNO容量控制已完成同5遍训练和DEV：隐藏width9→16、参数4.527M→14.206M，其余网格／频带／头／物理损失不变；新几何U/S29.03%／54.45%，同5遍六候选中联合最好，但3.138倍容量仅改善联合2.11个百分点，未达3pp推进门槛，不追加长训。训练整轮3.98分钟、DEV1.28分钟。[容量控制](results/geobc_wide_fno64_v25/BRIEF.md)。

原版TS空token控制也已完成：TRAIN300平均仅10.67%的R16单元有节点；剔除空token并将其latent设零后，TRAIN应力93.62%（原版93.67%）、DEV93.15%（原版93.20%），基本无改善，停止该路线。整轮训练5.31分钟、DEV1.61分钟；初始完整权重SHA与原版一致。[空token对照](results/geobc_occupied_ts_v27/BRIEF.md)。

同R16双模型训练目标对照已完成：标准化MSE使TS新几何U/S变差至65.62%／94.60%，FNO16至48.71%／72.14%，停止此路线。训练整轮6.69分钟，DEV1.97分钟，保持原物理损失。[目标控制](results/geobc_encoded_pair_v29/BRIEF.md)。固定输入F/M/E/L幅值先验同5遍也已完成：TS新几何U/S58.05%／94.18%，FNO16 30.60%／68.92%，均未改善联合误差；训练7.45分钟、DEV2.00分钟，停止追加。[幅值先验](results/geobc_response_gain_v31/BRIEF.md)。TRAIN编码器／查询头迁移对照已完成：TS新几何U/S53.01%／94.37%，FNO重置29.61%／61.86%，FNO完整5→10遍27.65%／58.96%；新整轮训练9.03分钟、DEV2.80分钟，teacher旧4.19分钟另列。迁移未帮助TS；完整FNO续训改善自身5遍，但仍不能抹平与R64基线的表示及曝光差异。[迁移对照](results/geobc_geometry_head_v34/BRIEF.md)。固定teacher中间场诊断已完成：TRAIN节点latent误差TS98.62%、FNO7.56%；新CAD98.66%／8.73%，但teacher原FEM应力误差64.98%，FNO学生仍65.90%，没有解决绝对精度。新训练整轮7.49分钟、DEV3.39分钟，停止该latent路线的长训。[诊断简报](results/geobc_node_latent_teacher_v37/BRIEF.md)。官方structured3D TS同物理5遍已完成：新CAD U/S52.33%／93.36%，原TS52.06%／93.20%，无改善；整轮训练5.19分钟、DEV67.06秒，停止该候选长训。[结构网格对照](results/geobc_structured_TS_v40/BRIEF.md)。FNO R64→R128同参数／初值／病例顺序对照也已完成：新CAD U/S31.85%／56.45%，原R64 31.61%／56.10%；纯训练耗时1.90倍，停止扩大网格。训练整轮4.57分钟、DEV53.95秒。[分辨率对照](results/geobc_resolution_FNO_v43/BRIEF.md)。下一检查固定已训练的全局FNO场，添加初值严格为零的MLP／连续Hash残差，仍由原物理标签监督；只完成合成CPU接口，真实训练尚未实现／启动。当前负结果不证明TS架构上限。
原全节点++物理5遍训练耗时31.81分钟，TRAIN应力误差93.81%，没有通过训练门槛，因此该配方新DEV未运行。[负结果](results/geobc_native_physical_TRAIN_v13/BRIEF.md)。紧凑表示已通过真实GPU全节点反传验证；R16池化与容量差异明确记录，不能视为纯架构消融。[资源验证](results/geobc_coarse_transolver_resource_v19/BRIEF.md) · [当前计划](notes/geobc_coarse_three100_physical5_v20_plan.json) · [真实启动身份](notes/geobc_coarse_three_v20_actual_launch.json)。

## 数据与证据

固定100 TRAIN／20 knownDEV／20 FINAL CAD，三个集合按CAD隔离，每CAD三载荷；scaler仅旧TRAIN40拟合。来源是可用官方TRAIN池、单seed，不能称官方无偏测试。[划分](notes/geobc_prospective_cad_split_v1.json)。原CSV场按GRID对齐H5 vertices，保留signed U(mm)/S(MPa)原标签；旧H5-only结果已撤回。全部原节点、病例均值／最坏／绝对应力最高1%热点分别报告，teacher／grid往返／FEM误差分开。

先前同40遍冻结测试：FNO位移／应力22.39%／48.54%，全局Hash24.29%／48.44%；Hash热点34.05% vs FNO41.69%。这批FINAL已揭盲，未来迭代不再当盲测。[冻结测试](results/geobc_FINAL_v9/BRIEF.md) · [位移云图](results/geobc_FINAL_v9/plots_v2/field_U.png) · [应力云图](results/geobc_FINAL_v9/plots_v2/field_S.png)。40遍包含在整轮总耗时内：FNO16.61分钟；原全节点++约4.3小时是外推总时间，不能再乘40。

整体FNO几何／边界上下文控制的连续TCNN场已实现、训练并完成节点／分块／重排／输入梯度检查；连续接口一致不等于独立离节点FEM精度。[宏观对照](results/geobc_operator_hash_v5/BRIEF.md) · [连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

30项可恢复slides实验已完成，涵盖MLP、NGP、Hash、Transolver／++／3、残差、FNO及蒸馏。原实验目录、部分25特征schema／网格原点／50样本集合缺失，不能称精确数值复现。原200CAD获取在固定预算内input-negative；旧++作者制度中断，均保留失败成本与局限。

[一分钟结果](results/SOLVER_BRIEF.md) · [Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md) · [阶段历史](results/README_STATE_PRE_V20_20261011.md) · [输入负结果](results/geobc_coverage_input_negative/BRIEF.md) · [制度负结果](results/geobc_pp_regime_confirmation/BRIEF.md)。

## 运行与私人仓库

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存上限30.5GB（30,500,000,000 bytes），公共非阻塞GPU锁。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。研究工具在WSL项目根目录使用 `python -m tools.<模块名>`，按对应冻结计划／真实终态交接运行；不能因旧status或SOURCE-ready重复启动。

新克隆先执行 `python tools/restore_geobc_source_line_endings.py --check`，仅行尾不同可执行 `--restore`，恢复216个冻结运行依赖的LF／CRLF字节；未知内容拒绝写入。半小时监视器持续推进，健康后台训练只作轻量检查。

[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)最新已验收提交`272d437`，包含R128同5遍物理训练／新CAD读出及已验看报告：未改善，不追加网格长训练；并包含冻结FNO上的MLP／Hash残差合成CPU原型，尚无残差GPU训练结果。发布57.41秒，远端确切commit／tree和私人身份已核实。[发布状态](notes/github_publish_plan.json)。