# 论文证据链：骨干选择 → 物理信息 → 字段消融 → Fieldwise

论文主线：在原版DeepJEB完整节点联合位移／signed应力任务中，先选定适合当前条件的工作骨干，再利用物理信息改善它，以字段消融解释紧凑架构的设计。

## 1. 历史骨干选择：为什么以FNO为工作骨干

[历史骨干比较](../results/backbone_comparison/README.md)恢复五组已有结果，保留各自划分、曝光、容量和成本口径：早期五CAD筛选覆盖MLP、FNO及三个Transolver家族；同R64输入／查询头对照分别保留原4800和后续12000端点；100TRAIN同20遍DEV20/60比较FNO、FNO+Hash与原版TS+Hash；原网格TS表示诊断作为TRAIN-only负结果单列。[五份原始报告档案](../results/research_history/README.md)保留了指定历史提交的原文，供追溯早期30项实验、几何／BC探索与物理求解研究。

其中原4800控制最直接约束表示和容量：FNO9与PP256使用共同R64输入和native25查询头，参数4,526,858／4,523,268。DEV U/S为27.930%/56.062%与41.437%/86.406%；训练UTC/monotonic约26.969/24.855分钟与144.359/132.357分钟。它比跨表示的早期大FNO对照更适合支撑骨干选择，但仍限于固定训练配方、单seed和已知开发集，不是家族上限。

同20遍历史结果：FNO U/S为20.647%/48.830%，原版TS连续解码器35.793%/59.381%；参数约4.53M与9.60M；60场推理UTC/monotonic为1.266/1.266秒与6.501/3.546秒。这支持当前条件下的效果和成本取舍，不能推广为Transolver家族上限。

FNO+Hash单seed误差20.375%/48.339%略好，这一事实应保留。后续[三种子确认](../results/neural_baselines/README.md)中，Hash相对plain FNO的primary优势仅0.186个百分点、S优势0.621个百分点，未达预声明3个百分点标准。因此选择更小的plain FNO作为工作基线，同时报告Hash的热点改善；不是宣称FNO必然在所有指标、成本或配置上更好。

历史PHYSICAL指物理单位目标和损失，不是后来新增U3/gradient9输入。不同组不构成一个等配置排行榜；早期FNO也比部分TS候选更大。

## 2. 信息增益：物理位移和梯度是否有效

[ZERO/ACTUAL匹配实验](../results/physical_information_learning/README.md)使用相同扩大后的解码器、初始化、参数量、病例顺序、训练曝光、损失和原TRAIN40 scaler。ACTUAL新增物理U3与1000倍恢复梯度9，ZERO对应列填零；不直接把物理U当模型输出。

100TRAIN/300、同20遍、三种子、DEV20/60：ZERO U/S为21.746%/49.419%，ACTUAL为15.361%/35.998%；primary改善9.903个百分点，三个配对种子均改善。它支持物理信息的作用，不能用历史TS和最新ACTUAL的跨实验差距代替。

## 3. 字段消融：不同输入分别帮助哪一个输出

[U_ONLY/GRAD_ONLY](../results/input_components/README.md)保持相同FNO、初始化与20遍曝光，分别屏蔽梯度9或位移3：

| 输入 | DEV U误差 | DEV signed S误差 |
|---|---:|---:|
| ZERO | 21.746% | 49.419% |
| ACTUAL：U3+gradient9 | 15.361% | 35.998% |
| U_ONLY | 11.717% | 48.652% |
| GRAD_ONLY | 17.058% | 35.471% |

U_ONLY比联合输入的U误差低3.644个百分点，却不保留大部分S收益；GRAD_ONLY保留S收益，但U不如U_ONLY。这为“U头用位移、S头用梯度”提供动机，尚未直接验证训练梯度冲突的因果机制，也不能证明任何联合U/S任务都必须采用Fieldwise。

## 4. 紧凑验证：Fieldwise保留收益并降低双模型成本

[共享主干实验](../results/shared_trunk_field_learning/README.md)用一个PointNet/FNO主干和两个字段输入头：U头用U3，S头用gradient9。三种子DEV U/S为11.569%/34.819%，通过距字段路由参考各不超过1个百分点的预声明标准。4,554,762参数约为双模型9,056,788的一半；比联合输入模型4,528,394多0.58%，不是严格同参数消融。

双模型参考U11.717%/S35.471%及primary23.594%由各字段标量组合，不是新增独立验证。Fieldwise相对联合ACTUAL的primary改善2.486个百分点，不能误称通过另一个3个百分点门槛。结论是字段选择有开发证据、共享主干紧凑实现保留收益，而非架构具有普遍必要性。

## 5. 最终冻结评价与可调用接口

架构、六个端点和三种子在FINAL读出前固定。[FINAL20/60](../results/final_evaluation/README.md)上，Fieldwise U/S为11.629%/34.247%，ZERO为21.255%/47.003%；primary改善11.191个百分点、3/3种子胜，预声明标准通过。360完整预测先保存，再共享读取20个原CSV的60载荷GT，无FINAL训练、scaler拟合、最佳seed选择或阈值调整。

[坐标查询交付](../results/prepared_response_delivery/README.md)另验证固定首种子一个模型会话覆盖20CAD、1,200行新U/S响应、60个域外拒绝，GT读取0。它是接口／成本证据，不是独立离节点FEM精度。

历史DEV、最新物理增强DEV与FINAL分开展示；三种子共享20CAD，不能视作60独立CAD。评价来自官方TRAIN池内、身份隔离且受源可用性限制的划分，未证明shape-family隔离或工程适用性。原signed标签直接使用，原signedVM导出契约仍未知。源构建、物理求解、训练与缓存推理成本分开报告。
