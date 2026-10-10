# 已交付新增CAD的独立候选pilot：仅源码设计

**可准备一个额外验证pilot；目前只有源文件齐全证据，尚未验证三载荷缓存。** 本文与[draft](geobc_available_cad_pilot_design_draft.json)不授权准备、推理或标签读取。原540源文件／660输入／2400预测协议、20源码pins及input-negative均保持原样。

原完成收据317份全部来自additional TRAIN160：FieldMesh 107、Field CSV 105、SolverDeck 105；其中102个CAD三类齐全，均与oldTRAIN40、冻结DEV20互斥，DEV20齐全数为0。收据有实际字节数／本地SHA，未在本分析重新检查原文件。CSV字段解析及标签读取均为0，不能据此宣称ver/hor/tor列、GRID对齐或缓存已合格。已查coverage bootstrap仅oldTRAIN120，solver_selection45及既有50CAD multiload缓存与新增160交集为0。

**固定范围。** draft完整列出102 eligible ID，按CAD ID的UTF-8字符串升序取前20，得到60病例（ver/hor/tor）；不按标签挑选，不因输入不支持而替换。它们来自官方TRAIN池，仅对仍只训练old40的候选构成额外held CAD；数据可用性和字符串排序限制总体代表性，之后纳入200TRAIN会失去held资格。原冻结20DEV仍封存，不能将本pilot作为永久held或最终test。

**固定九候选。** 旧五方法4800、PP新制度4800／24000、FNO grid25／native25各4800。先行任务真实终止后，仅加载指定端点并冻结身份、成本、曝光。TRAIN推进门只决定科研优先级，不能删除有效端点或伪记失败；九候选全部预测，缺失／技术无效端点则逐60例提交真实失败。父input-negative也不构成新case失败。全原节点signed4、旧scaler不变；不同曝光／容量／成本不冒充相同预算比较。

**最少adapter。** 现入口是`geobc/coverage_comparison_data.py::_prepare_CAD`，无需原540complete wrapper。独立producer保留来源role=additional_train，显式设evaluation-only及`train_new160=[]`走无target分支，原源码／pins不改；照搬TRAIN角色会提前调用`_targets`。复用H5 vertices/cells、真实deck、CSV GRID坐标、`_deck_case`、`feature_arrays`及0.0002mm双射／旧cloud规则，只增独立身份、整批预算、缓存manifest和屏障。**该入口含CUDA：** 原recipe.device=cuda，源码显式`.to(device)`后用Torch `_stencil/_splat`构造R64并同步，不能在PP活着时作为CPU准备启动；未找到现有`_coverage_grid`函数。若拆CPU-only阶段须另实现、审源码后才能执行。

**屏障与预算。** 全9×60=540份新预测或真实失败提交后，才统一每CAD读一次三载荷12个U/S列并共享GT；完整60例才有均值／最坏／热点，不对子集重算分母。新540是本pilot的终端清单，绝非原540源文件完成声明。准备全段≤3600秒、独立readout全段≤3600秒，均计真实I/O、初始化、网格、持久化和失败成本；共用一次公共GPU锁、峰值24GiB，须在PP/FNO先行进程真实退出后开展。现函数内每CAD3600秒deck guard不能代替整批上限。20个target-free输入不齐则如实negative，GT仍0，不换ID或配置。

本次仅读源码与JSON metadata并产本文／draft；未执行模块或contract，未读H5／CSV／npy／PT／GT，未下载，未创建rootmarker。下一步由根冻结独立计划、决定是否实现并审阅最小producer／evaluator，之后才可实际释放。
