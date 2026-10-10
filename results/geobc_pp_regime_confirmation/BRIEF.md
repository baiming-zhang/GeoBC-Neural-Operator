++ 作者训练制度确认：未通过完整实验资格。实际更新 23166/24000；原生状态 completed / terminal_budget_or_backend_or_nonfinite。
完整TRAIN120物理指标改善幅度门：未通过；它与成本/过程完整性资格分别报告。
原DeepJEB，原40CAD×ver/hor/tor=120；signed UXYZ(mm)/S(MPa)，seed2026。下表相对L2均为无量纲，热点使用原top1% |S|。

| 固定TRAIN120端点 | U均值 | S均值 | 热点均值 | 最坏S | 最坏热点S | 完整finite |
|---|---:|---:|---:|---:|---:|---|
| old_constant4800 | 37.64% | 77.59% | 81.83% | 91.40% | 93.43% | True |
| new_bundle4800 | 44.20% | 80.08% | 83.02% | 94.67% | 92.71% | True |
| new_bundle24000 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 | False |
| FNO_reference4800 | 13.98% | 34.86% | 38.32% | 43.87% | 55.96% | True |

闭合monotonic clock：训练 19717.2s/28800s，非训练 126.2s/1200s，总计 19843.4s/30000s；守卫monotonic launch→exit 缺失/30000s。
守卫记录UTC起止跨度 23596.3s/30000s；UTC减守卫monotonic 缺失。分phase的UTC耗时未记录，保留原monotonic值，不按比例补算；两轴差异原因未确定。
检查点恢复后，原未闭合guard记录保持原样；整段守卫monotonic及精确计算次数无法恢复。回滚更新、已记录额外计算与unknown区间单列；只展示实际完整的固定端点，24000缺失时列为空，严格成本资格不强行通过。
最终24000门槛：S均值≤0.8×old、U均值≤1.05×old、最坏S及最坏热点S≤old；并须完整120、有限、原生完成gate、预算、24GiB峰值及零禁读receipt全部通过。
未通过项：both_new_full120_finite, final24000_thresholds, native_success_and_gate, actual_outer_guard_budget_and_success, native_memory24GiB, native_TRAIN_only_read_receipt, full200passes120, fixed_endpoint_identity, saved_loss24000_finite_and_TRAIN_only, summary_endpoint_copies_match_native。
新4800对旧constant4800是同曝光制度组合比较；新24000同时改变制度与曝光，不能作单独scheduler因果结论。FNO4800仅作原fullTRAIN参考，容量和时间不同。
本轮没有DEV/held5选优、泛化或最终solver胜者结论。缺失、technicalfailed或科学负结果如实保留为false。
曲线：保存的loss/LR/beta1/preclip gradient共 23166/24000 行；PNG/SVG可用 True；缺点/坏行见metrics与manifest。
