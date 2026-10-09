# signed stress 读出粗筛：冻结建议

2026-10-10；recommendation_only。只读既有提案、语义说明、源码与五 CAD 小元数据；未执行数组/GT/权重/模型/GPU任务。不改变当前 warm-start 规则或启动下一阶段。

**建议采用 root 的单次粗筛。** TRAIN 固定 `101_428/ver,101_56/ver,101_78/ver`，DEV 固定 `102_249/ver,102_38/ver`，不换失败 CAD。单 seed2026；每臂 40 个平衡 pass/120 个完整 case updates，Adam 1e-3、无 decay、节点 chunk32768。两臂均 fresh xyz PointNet1024→global32，加四层256/ReLU scalar MLP；输入67为 xyz3+原25+global32+六张量槽+valid1。参数、完整初始 trainable 状态、case 序列、优化器和预算相同；两臂自然初始输出不要求相同。actual 使用 σ6，zero 只清零这六槽，共用相同 valid1。拟 API `encode_geometry(cloud,mask)->context32; forward_points(xyz,features,context,tensor6,valid)->encoded_S`，完整 case scalar MSE 按节点数加权分块反传，每 case 仅一次 Adam；不新加 constitutive loss。

**必须只用物理预测 u，不能用 baseline 已读的 GT u。** 五 CAD 的已收敛 `runs/geobc_exploration/elastic_crosscad5_ver/summary.json` 中 `cases[key].prediction/physical_u_path` 是权威来源：`…/{cad}_ver/physical_u_mm.npy`。它在旧 stress/位移标签比较前已冻结；旧 `metrics` 不作为特征。核对该 prediction 的 stat/hash、原节点/GRID 身份及同 CAD input/operator contract。禁止打开 prepared 的 `displacement`，也禁止复用会加载 `physical/target` 的 `solver_selection_data.load_case/prepare_data`；不以 GT u 修正物理解、恢复方式或归一化。

输入定位由各 case 的 `input_manifest`、`operator_summary` 取，不靠全局101_479合同：`data/geobc_elastic_crosscad5/{cad}/input/manifest.json` 的 `cache_file` 提供 cells10、物理坐标、lame[λ,μ] 等；operator summary 的 `arrays.gradients/weights` 指向 `physical_gradients.npy[T,4,10,3]` 与 `quadrature_weights.npy[T,4]`。既有 `elastic_pilot_data.load_input` 可读 target-free input；新特征构建宜窄读所需成员及已保存梯度/权重，不实例化 K、不重复8个 K guards、不重解。建议新 cache 保存 `sigma_MPa[N,6]`, `valid[N]`, GRID/xyz 身份、预测/operator来源与统计；loader 只返回输入，标签另读。

**恢复规则一次固定。** FP64 物理 mm 梯度：`G=Σu_a⊗∇N_a; ε=sym(G); σ=2με+λtr(ε)I`。槽顺序 `xx,yy,zz,xy,xz,yz`，tensor shear、MPa。每 cell 以四 Q 的正物理权重求平均 σ，再按 `V_cell/10` 对 incident 全十节点归一恢复，`V_cell=ΣQweight`。零 incident mass 的 reference 节点 `valid=0,σ6=0`，保留原全N输出；不宣称那里是零应力。首次构建检查形状/finite、物理单位、对称性、正权、常数保持/归一、覆盖与精确节点身份，复用已有几何 guard。该恢复是新 proxy，不是 OptiStruct 节点外推/平均或 strain/stress GT。

**标签与 scaler。** TRAIN3 各仅读取一次 prepared `stress` 成员，保留原 signed MPa，并缓存用于120 updates；不读位移。scalar scaler 用 TRAIN3 全节点 `asinh(S/q90(|S|))` 的 mean/std；σ6 的 mean/std 仅 fit TRAIN3 的 valid 物理预测特征。两臂复用同两类 scaler；σ 标准化后 invalid 槽仍置0，zero 臂全部六槽置0。固定零方差处理并记录；不得把 std 的数值保护混入物理指标分母。两个 DEV 的两臂完整端点预测/FP64物理解码均落盘后，才每 CAD 一次读取 `stress` 算指标：总 stress reads/GT reads=5，displacement reads=0。无 DEV scaler/step选择，无 oldfinal/dia/sealedtest。

**收益和停止。** 主比较固定120端点，全原节点 signed S case-mean/worst relative L2、MPa RMS/max；热点恰 ceil(1%N) 个最大 |真S| 节点，报告 mean/worst 与固定 tie 规则。建议沿 root 门：两个 DEV L2 均<25%，mean 相对改善≥20%，worst≤zero臂1.05倍，热点 mean 相对改善≥10%。零参考范数时 relative=null、绝对指标保留，门不自动通过。只2 DEV/单 seed，不作统计显著或通用 GeoBC 结论。全新 stage900s、每臂训练最多300s；成本/RNG/断点累计不重置，预算不够不降低步数冒充完整端点。未过门停止此读出候选，不救宽度、符号公式或恢复参数。

完整 cold 成本另报 physics input/setup/solve + 张量恢复/IO + checkpoint/scaler加载 + 全N读出；历史物理冷成本约77–101秒/例是实测上下文，不能把复用缓存当 cold=0。σ-zero 直接 MLP 部署不需要物理解算，应同时列其独立冷成本，以及本次实验共用特征准备费用；不要求高精度混合路线比 FNO 单次冷推理更快，但不省略物理成本。

**仍未知。** [源语义说明](geobc_stress_semantics.json)确认论文称 signed von Mises，exporter 的符号函数、scalar先后与节点恢复未公开。原 signed label不取绝对值、不按 C:ε 反推出“真scalar”；新张量作为信息输入是否有效由粗筛决定。当前还没有上述 σ cache、两类 TRAIN3 scaler、新 scalar loader或读出模型，缺项须经 root 独立科学冻结后首次真实阶段完成，本文不授权访问。
