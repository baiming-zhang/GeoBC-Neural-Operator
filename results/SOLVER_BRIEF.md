# GeoBC 求解器：当前判断

**尚未选定求解器。已统一筛选六类方法，GINO正在加入比较；Hash专项小幅优化已暂停。**

目前FNO的应力预测较好且两个seed较稳，MLP保留为低成本位移对照。一次冻结新CAD评估的全节点误差如下（两seed范围）：

| 方法 | 位移误差 | signed stress误差 |
|---|---:|---:|
| FNO | 33–36% | 49–50% |
| MLP | 27–49% | 64–73% |

MLP一个seed的位移优于FNO；组合载荷dia两者均明显退化。Hash小样本拟合强，当前跨CAD证据不足。短预算不等于慢方法已收敛，现有结果不能宣布某个solver工程可用。

节点误差也不足以评判连续场：固定TRAIN局部的位移派生P2代理误差仍很大，FNO约507–610%、MLP约413–421%。这是局部运动学代理，非solver应变GT或全CAD结论。

**正在做：**GINO同40TRAIN/5VAL、相同原节点signed四场损失、相同480更新；已对齐实际频带，首个32.5万节点完整更新通过。只训练一个候选，记录真实精度与成本，不重复已有六类实验。新test5已取得且标签封存；旧final不用于调参。

详细证据：[六类初筛](solver_selection/BRIEF.md) · [固定成本确认](solver_confirmation/BRIEF.md) · [冻结评估](solver_final_eval/BRIEF.md) · [连续能力](solver_capability/BRIEF.md) · [GINO适配范围](../notes/gino_vs_fno_implementation_scope.md)。