# GeoBC：原版DeepJEB求解器筛选

从几何、材料、载荷及约束预测位移和signed应力，目标是共享模型迁移到新CAD，并提供由整体几何与边界条件控制的连续空间查询。使用原版DeepJEB。

## 当前结论

**目前优先保留FNO，Hash保留应力热点对照；绝对误差仍高，最终求解器尚未确定。** 已知单样本Hash小于1%是拟合能力，不是新CAD的迁移精度。

同100训练CAD×三载荷、物理损失、5遍／1500步，knownDEV20个新CAD×三载荷、全部原节点：

| 方法 | 位移误差 | 应力误差 | 热点误差 |
|---|---:|---:|---:|
| FNO R64 width16 | 29.03% | 54.45% | 42.81% |
| 全局Hash | 31.37% | 54.45% | 40.85% |
| 原版紧凑Transolver R16 | 52.06% | 93.20% | 93.36% |

原版是这一轮紧凑TS系列最好；++／3应力约94%。网格表示、容量和原生精度不同，不能视为架构单因素对照或架构上限。[三版TS](results/geobc_coarse_three_physical5_v21/BRIEF.md) · [宽FNO](results/geobc_wide_fno64_v25/BRIEF.md) · [全局Hash](results/geobc_global_hash_physical_v18/BRIEF.md)。

FNO width9从5遍继续到10遍，新CAD位移／应力为25.60%／52.14%，追加更新窗口97.18秒。这是增加曝光量的效果；不与fresh5当公平同训练量比较。[5→10遍](results/geobc_physical10_v16/BRIEF.md)。

最新短筛选冻结已有FNO5，只额外训练MLP／Hash残差各5遍：MLP新CAD位移／应力31.14%／55.87%，Hash31.52%／53.75%。Hash热点39.02%优于完整FNO10的41.93%，但综合误差仍是完整FNO10更好。两残差未通过既定推进门槛，不追加Hash长训练。新TRAIN整轮4.07分钟，knownDEV1.99分钟；原teacher历史成本另列。[残差一页报告](results/geobc_operator_residual_v46/BRIEF.md)。

R64→R128同参数／初值／顺序的5遍对照也未改善，纯训练窗口约1.90倍，不扩大网格。其他失败与局限保留于[阶段历史](results/README_STATE_PRE_V47_20261011.md)。

## 最新短筛选与下一步

全局BC分支已在FNO／原版TS各训练5遍并完成新CAD读出：FNO位移／应力31.15%／55.95%，TS50.05%／93.21%。相对各自原5遍，FNO改善0.46／0.15个百分点，TS位移改善2.01个百分点而应力基本不变；两者均未通过推进门槛，停止该分支。两arm训练整轮6.33分钟，knownDEV1.97分钟，CPU报告1.49秒；峰值及全节点资格见[一页报告](results/geobc_global_BC_v49/BRIEF.md)。

原版TS已从已有5遍延续至10遍：位移52.06%→40.33%，应力93.20%→91.50%，应力改善仅1.70个百分点，未通过推进门槛。同10遍FNO为25.60%／52.14%，当前优先保留FNO，暂不追加这套TS的长训练。新增5遍训练整轮4.81分钟、knownDEV65.38秒，Adam／RNG／曝光量保持连续。[TS与FNO一页对比](results/geobc_original_TS_v51/BRIEF.md)。

下一候选是已有GINO的当前物理损失／100CAD版本，而非重跑旧40CAD编码损失实验。物理反向传播适配器已通过CPU整体图梯度对照；原节点缓存已实现并通过metadata构造检查，缓存约7.53GB；40个几何CSR可复用，另60个需构建。尚未做本轮GPU全节点资格或训练，不能声称已得到新精度。沿用旧实际模型，最多先筛5遍，不默认40遍。[实施状态](notes/geobc_gino_physical5_proposal_v52.json)。

初期5遍筛选，有证据再追加少数领先方法，不默认40遍或24,000步。旧FNO40遍整轮16.61分钟；全节点++40遍约4.3小时是整段外推总时间，不能再乘40。[时间策略](notes/geobc_early_screening_budget_policy.json)。

## 数据与复现范围

固定100 TRAIN／20 knownDEV／20 FINAL CAD，按CAD隔离、每CAD三载荷；scaler只由旧TRAIN40拟合。原CSV场按GRID对齐H5 vertices，保留signed U(mm)/S(MPa)及全部原节点，teacher／grid往返／FEM指标分开。旧H5-only结果撤回。knownDEV已用于迭代；单seed、可用官方TRAIN来源，不能称无偏官方盲测。旧FINAL已揭盲，未来不再用来调参。[划分](notes/geobc_prospective_cad_split_v1.json)。

整体FNO上下文控制的连续TCNN场此前已实现并训练，通过节点／分块／重排／输入梯度检查。接口连续不等于独立离节点FEM精度；RBE2 CM是耦合成员标志，不假定为grounded SPC或据此夹紧位移。[连续接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。

30项可恢复slides实验已完成。原实验目录、部分25特征schema／网格原点／50样本集合缺失，不能称精确数值复现。原200CAD获取及旧++作者制度中断的失败成本均保留。[一分钟结果](results/SOLVER_BRIEF.md) · [Slides总表](ALL_METHODS_RESULTS.md) · [复现报告](results/REPORT.md)。

## 运行与私人仓库

WSL `cold-surge-ubuntu-24.04`，Python `/home/research/.venvs/geobc/bin/python`，RTX5090；显存上限30.5GB（30,500,000,000 bytes），公共非阻塞GPU锁。工具使用 `python -m tools.<模块名>`，按对应冻结计划及真实终态交接运行，不因旧status／SOURCE-ready重复启动。[依赖](requirements-lock-wsl.txt) · [TCNN安装](notes/install_tcnn.md)。

新克隆运行 `python tools/restore_geobc_source_line_endings.py --check`；仅行尾差异可用 `--restore` 恢复245个冻结依赖的LF／CRLF字节，未知内容拒绝写入。半小时监视器继续推进；健康后台任务只轻量检查。

[私人仓库](https://github.com/baiming-zhang/GeoBC-Neural-Operator)最新已验收提交`50cf0fe`，包含上一轮全局BC真实结果与原版TS5→10设计；本轮TS终态与GINO适配准备包正在形成新快照。发布61.10秒，45份文件、确切远端commit／tree及前中后私人身份均核实。[发布状态](notes/github_publish_plan.json)。
