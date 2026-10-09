# GeoBC：一分钟结果

**六家族筛选、双种子固定成本确认及一次冻结最终验证已完成。FNO是主候选，MLP是低成本对照；尚未证明工程可用。**

- **完成：**原30项slides复现；后续91个阶段包含训练、诊断、输入准备和最终评估，并非求解器种数。
- **新CAD：**FNO的应力误差较低且两seed稳定；MLP一个seed位移误差更低，另一个明显退化，不能称FNO所有指标都胜出。
- **组合载荷：**两条路线都明显变差，且seed波动较大。节点拟合或known-load结果不足以证明GeoBC泛化。
- **后续规则：**不据旧final调参、选择seed或重训本轮模型；新结构使用另外冻结、仍封存的test5。

凝练结果见[求解器简报](../SOLVER_BRIEF.md)，完整冻结known3／dia结果见[最终评估](../solver_final_eval/BRIEF.md)。
当前选型限于共同数据、配方与预算；自主研究与30分钟监视继续。

- **局部连续能力：**固定TRAIN32 cells／504查询连续诊断完成（3.718s）：节点精度不代表P2梯度准确，FNO原生查询未消除该局部proxy差距；MLP native未实现。C0通过只表明结构接口，非solver strainGT或全CAD结论。
- **新结构输入：**GINO单TRAIN radius输入图可行（2.245s，6,547,680 pairs，179,316点全覆盖）；仅构图资源，尚无GINO模型／BC／连续场精度证据。
- **新结构资源：**GINO全输入/512输出模型资源通过（4.985s、8/8 checks），peak allocated 1.829 GB；零labels/optimizer更新，训练与首例全N输出仍待验，尚无精度证据。
- **新结构初筛：**GINO480主端点U/S均值=54.424%/72.456%，实测train=740.1s；四遍历完成，较旧screen及600秒确认的成本背景另报，未凭更多耗时选定solver。
- **新周期test：**独立test5身份已冻且字段封存；获取completed（15/15文件），原始字节已获取，尚未评估或用于模型开发。

固定480的FNO响应基结构配对与DEV组合载荷结果见[FNO结构简报](../fno_response/BRIEF.md)；历史direct4仅上下文，未据此改旧final或新sealed test。

共同BC22输入、固定480的DeepONet/concat-MLP与DEV组合载荷结果见[结构初筛简报](../deeponet_screen/BRIEF.md)；heads/容量不同，单seed早期结果不是工程资格或full-model初始化匹配。

一个完整节点的多尺度网格求解器固定480结果见[网格结构简报](../multiscale_mesh/BRIEF.md)；参数/拓扑输入不同，仅早期筛选，未据此改旧final或新sealed test。

固定480的完整节点PointConv与开发组合载荷见[点卷积简报](../pointconv/BRIEF.md)；纯几何KDE/FPS和BN适配均公开，非官方分割模型或同容量因果复现。

单TRAIN几何的有界近似弹性基线见[物理求解简报](../elastic_pilot/BRIEF.md)；数值收敛、标签误差与实际成本分列，不代表神经跨CAD成绩或原软件精确重放。