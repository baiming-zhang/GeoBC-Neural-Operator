# GeoBC 求解器探索：阶段结论

使用原版DeepJEB：六家族初筛后，FNO、MLP、Transolver++各做双seed、600累计训练秒确认，再预先冻结FNO/MLP最终验证。下表是预算端点的两seed case-mean物理相对L2范围；U为signed XYZ位移，S为源signed scalar stress。

| 方法 | 未见5 CAD×已知3载荷：U / S % | 同5 CAD×dia：U / S % |
|---|---:|---:|
| FNO | 33.09–36.17 / 49.23–50.39 | 46.43–76.51 / 68.78–96.31 |
| MLP | 26.53–49.40 / 64.20–73.49 | 57.71–131.70 / 84.19–98.90 |

FNO的应力误差更低、已知载荷均值较稳定；但MLP seed2027的U为26.53%，低于两个FNO seed，不能说FNO所有指标都赢。dia组合载荷误差明显较大且seed不稳，MLP一个seed的U超过零场基准。当前仅有条件候选依据，尚非工程合格；双seed不构成统计显著性证据。

结果覆盖全原始节点；已知载荷为ver/hor/tor，dia单独报告。VAL-best仅作预先选定的次结果，成本、最差case及完整指标见[冻结评估报告](solver_final_eval/BRIEF.md)。不按final改路线、挑seed或反馈本轮训练/调参。

已计划独立TRAIN连续能力检查；节点误差尚未证明连续查询或BC满足。之后新的架构适配须采用独立、未使用的评估集。
