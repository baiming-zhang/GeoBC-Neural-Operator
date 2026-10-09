# 物理张量＋signed stress读出

固定粗筛门：**FAIL**。

| 读出臂 | DEV mean / worst S L2% | 热点 mean L2% | 单请求冷成本s |
|---|---:|---:|---:|
| 直接 MLP／六零槽 | 94.62 / 95.82 | 97.57 | 0.3266–0.3402 |
| 物理 σ6／同 validity | 31.76 / 38.54 | 53.8 | 92.19–101.8 |

TRAIN3/DEV2、仅ver、单seed2026、固定120个完整case更新。比较原始signed MPa；两臂相同布局/初始状态，σ6来自物理预测位移，validity共同。热点为每case最高|真S|的1%原节点；完整MPa误差、最差ID、实际读取次数与节点曝光见metrics.json。

mean相对改善66.43%、worst比值0.4022、热点mean相对改善44.86%；每DEV<25%的检查False。

真实累计：stress读取5次，displacement读取0次；scaler fit记录2。resume增加的实际读取按receipt保留，不固定写成5。

冷成本是已测baseline物理费用＋本阶段恢复/读出的组成核算，并非一次fresh联合计时。直接MLP不计物理解算或σ恢复；共同拓扑validity与自身输入/加载/推理仍计费。训练、实验共用准备、缓存增量延迟分别保存，不与推理冷成本混合。

该门只决定是否值得扩大这个粗筛；未通过则结束此固定recipe。源scalar符号/恢复仍未知，张量proxy不是源stress GT；单seed/两个DEV不证明工程资格、泛化或整个solver族失败。没有把旧位移GT精度移到新应力端点。
