# GeoBC 求解器：当前判断

**尚未选定求解器。已统一筛选七类方法，当前以FNO为主候选、MLP为低成本对照；Hash专项小幅优化已暂停。**

目前FNO的应力预测较好且两个seed较稳，MLP保留为低成本位移对照。一次冻结新CAD评估的全节点误差如下（两seed范围）：

| 方法 | 位移误差 | signed stress误差 |
|---|---:|---:|
| FNO | 33–36% | 49–50% |
| MLP | 27–49% | 64–73% |

MLP一个seed的位移优于FNO；组合载荷dia两者均明显退化。Hash小样本拟合强，当前跨CAD证据不足。短预算不等于慢方法已收敛，现有结果不能宣布某个solver工程可用。

节点误差也不足以评判连续场：固定TRAIN局部的位移派生P2代理误差仍很大，FNO约507–610%、MLP约413–421%。这是局部运动学代理，非solver应变GT或全CAD结论。

新增GINO训练已完成，总用时14.4分钟。同480更新下，验证U/S误差54.4%/72.5%，FNO为47.5%/66.7%；GINO训练耗时约7.4倍，本配置暂不优先。短预算结果不代表整个GINO家族无效。

最新结构对照已完成（两臂训练合计约5分钟）：开发集组合载荷的U从普通FNO的55.9%降至约29.8%，S从70.4%升至73.8–74.6%。两种响应结构几乎相同，强制主干与载荷无关没有额外收益；不能把位移改善单独归因于这个约束。

**下一步：**设计几何／BC感知的DeepONet与共享输入MLP对照，检验不同场表示。新test5标签继续封存，旧final不用于调参。

详细证据：[七类比较](gino_screen/BRIEF.md) · [固定成本确认](solver_confirmation/BRIEF.md) · [冻结评估](solver_final_eval/BRIEF.md) · [连续能力](solver_capability/BRIEF.md) · [响应结构结果](fno_response/BRIEF.md) · [后续结构](../notes/solver_next_structures.md)。
