# 固定物理应力幅值诊断（proposal_only）

2026-10-10；仅 source/小 JSON 定位，数组、GT、模型、权重、GPU、诊断执行均0。原 raw6 的120更新 recipe已经结束；本文不训练、校准或复活该配方，须 root 独立冻结。

**问题与唯一公式。** 保持同 TRAIN `101_428,101_56,101_78`、DEV `102_249,102_38`，均 `/ver`，全原节点。
用未标准化的已保存物理张量六槽 `xx,yy,zz,xy,xz,yz`（tensor shear、MPa），仅计算
`q=sqrt(0.5*((xx-yy)^2+(yy-zz)^2+(zz-xx)^2)+3*(xy^2+xz^2+yz^2))`。
GPU FP64直接向量化；验证有限与非负，不clip/改scale，不fit、不试符号/主应力/trace候选。比较 `q` 和缓存原 signed S 的 `abs(S)`，只作 magnitude-only 误差，不覆写标签、不称q为原signed scalar预测。

**权威缓存路径和身份。** 父运行 `runs/geobc_exploration/elastic_stress_readout5_ver/summary.json` 必须completed且精确五case、两120端点及固定合同身份成立。
每case `summary.cases[key].arrays.sigma_MPa` 与 `.arrays.valid` 是主记录；独立 `input_cache/{cad}_ver/manifest.json` 应完全对应。
当前σ是 float64 `[N,6]`，valid是 bool `[N]`；不使用 normalized σ、神经stress预测或重新读取physical u来重构。
标签主记录为同run `label_cache/{cad}_ver.json`：`identity={key,scope:TRAIN|DEV,source,member:stress}`，`array={path,shape:[N],dtype:float64,bytes,mtime_ns,values_sha256}`。
读取该 `array.path`（`label_cache/{cad}_ver.npy`），只在新实际stage读缓存GT，不调用原 `read_label_once`，也不打开 canonical prepared NPZ。
read-start receipt必须表明本父阶段该stress源成员已读completed；committed JSON与父canonical身份/节点数一致。每次新实际加载检查缓存stat/shape/dtype，值SHA可从已加载连续bytes一次复核，避免另读整个文件。
原来源/算子合同只继承小JSON证据；禁 `run/train_arm/fit_scalers/load_case`，禁原stress/displacement、oldfinal、dia、sealed及weights。
可参考 `elastic_stress_readout.load_array(np,record,verify_values=True)` 的窄缓存校验语义；新入口保持metadata-only request_contract，科学读取仅root actualrelease后发生。

**已定位的规模（metadata，非新数组确认）。** 五N顺序为270276、221213、192353、211173、197514，总1092529。
父metadata五case均feature_valid=true、valid_nodes=N、missing_incident_nodes=0；新实际计算仍记录加载mask计数，不先假定两范围指标相等。
不手抄数组hash作为新身份；从上述JSON记录读取并锁summary/manifest/committed receipts小文件SHA与数组stat/valuesSHA。

**metrics与coverage。** 每case同时给allN与valid-only两组：relativeL2、RMSE/MAE/max MPa、ceil(1%所用节点数)最大|GT S|热点relativeL2/MAE/max；ties按原行ID升序。
fullN保留invalid节点；那里的缓存σ6=0是缺测，不能解释物理零应力。valid范围主用于判别proxy，全N只补充覆盖效应。当前mask全真若实际一致，两组应相等，并记录这个事实。
零GT范数或零热点范数：relative=null，绝对误差仍存；无分母floor，空valid集同样null。TRAIN3/DEV2均保持完整固定分母，缺case不均值伪补零。
结果保留每case、TRAIN/DEV mean/worst、worstIDs与热点定义，不挑最佳case/子集，不以训练误差改变公式。

**新阶段GT账本。** 缓存标签仍是GT；正常完整一次执行 `GT_reads=5, stress_reads=5, cached_signed_stress_reads=5`，分TRAIN3/DEV2。
`source_stress_member_reads=0, canonical_archive_reads=0, displacement_reads=0`；不能因为父run已读过而声称本次GT0。
每case实际读缓存前/后更新persistent read-attempt/read-completed receipt；resume跳过已提交诊断case，若必须重新加载则累计真实reads并明示，禁止把常数5写成实际值。
另列 `sigma_array_reads=5,valid_array_reads=5` 的正常完整规模和真实累计；模型/优化/训练/scaler fits/新解算/K guards均0。
建议一条独立CPU/GPU诊断job、总wall60秒（包含metadata/缓存IO/H2D/q/metrics/落盘，非预测实际耗时）；无toy、重复preflight或原算子构建。

**建议的粗分支（仅供root采纳，结果未知）。**
若valid-only两个DEV幅值L2均≤10%、DEV热点mean≤20%，并且TRAIN也无明显反常：幅值proxy有较强支持，下一宏观问题应是原scalar符号/恢复语义与可解释读出，不继续加原raw6训练步数；不能由此声称原signedlabels定义已确认。
若任一DEV幅值L2≥25%或DEV热点mean≥50%，尤其TRAIN同样差：proxy幅值/节点恢复本身仍是候选瓶颈，优先明确源stress exporter/recovery或物理stress恢复，而非救神经容量；不能断言唯一原因是物理解误差。
中间区间记inconclusive；只报告既有证据，由root选下一宏观问题，不自动参数扫、符号试验或新训练。
无论哪支，q的小误差只验证这个恢复proxy的幅值；不能区分所有积分、节点平均、材料/边界差异，也不能单凭幅值归因120step优化不足。
已有语义note仍为作者描述signed von Mises、精确sign/export/recovery未知；当前新q agreement没有测量，不预先称物理stress标签准确。

**建议结果schema。** `status/kind/config.contract/case_keys/cases[key]{group,num_nodes,valid_nodes,allN,valid_only,costs,read_receipt}/aggregate{TRAIN,DEV}/read_receipt/costs/interpretation`。
scope明确 `magnitude_only=true,predicts_source_sign=false,label_relabeling=false,training_updates=0`；数值诊断completed与物理/source语义结论分开。
