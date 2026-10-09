# 固定物理幅值＋学习符号因子（proposal_only）

2026-10-10；只基于已有源码/小JSON证据。未读新数组/GT/权重/模型，未执行检查/GPU/训练；不冻结或派发任务，不改 coarse 候选。

**值得一次有界筛选，理由有限。** 固定q幅值诊断 TRAIN/DEV mean L2约10.77%/10.10%、热点约17.58%/17.00%，是 useful-but-imperfect 的幅值证据；strict10%门仍inconclusive。它与旧signed raw6的31.76%不是同一目标，不能当因果或solver排名；旧raw6已经关闭。

固定同TRAIN3 `101_428,101_56,101_78` 与DEV2 `102_249,102_38`，仅ver/allN。q完全由既有raw σ6按已冻结唯一VM幅值公式计算；不套trace/principal/sign规则，不校准q，不假定q=absGT。新的fresh分类/因子头输出logit ℓ，最终 `S_hat=q*tanh(ℓ)`；固定幅值来自FE，神经头不用预测完整自由scalar。tanh因子允许低置信度衰减幅值，因此称bounded sign-factor而非纯hard-sign classifier。

训练派生 `y=sign(S)`、binary label `(y+1)/2`，只从原committed signed S cache生成，原field标签保持不变。S=0的分类项忽略，field项仍保留且鼓励预测0。不得以推测的物理sign作教师。
建议一个固定联合目标（无loss sweep）：每个完整TRAINcase
`L_field=Σ_allN(q*tanh(ℓ)-S)^2 / Σ_allN S^2`；
`L_class=Σ_{S!=0} S^2*softplus(-y*ℓ) / Σ_allN S^2`；`L=L_field+0.1*L_class`。
field相对能量项直接对应case signed L2²，分类项按真实应力能量而非节点数量，避免低应力海量节点主导分类。按原行chunk累加精确case sum，再一次Adam；零case参考能量明确unsupported/null，不造分母floor。热点另作固定评价，不额外改变采样/loss。

保持可解释成本：fresh old xyz-PointNet1024/global32，固定四层256ReLU因子MLP；query输入xyz3+原25+global32+TRAIN标准化σ6+固定q槽+valid1（68维）。tensor/q归一只fit TRAIN3物理输入，invalid归零；q生成最终物理幅值时用未标准化MPa。不加可学习幅值支路。不是容量/完整初始状态matched机制对照，也不是现raw6的继续训练。
建议单seed2026、20 balanced passes/60完整case更新，Adam1e-3/no decay、chunk32768；新stage总180s、训练最多120s，完整端点一次DEV，无best/第二seed/宽度/radius/loss/sign/correction救。预算不够明确incomplete，不补步/换case；实际梯度和资源只在唯一正式stage验。

只读父run已提交sigma6/valid与label_cache五case，身份取committed JSON/父contract/节点row/stat；不重开canonical/位移/oldfinal/dia/sealed。新stage实际缓存GT读取正常5次（TRAIN3/DEV2），source stress_member_reads=0、GTu=0；resume按真实访问累计，不因缓存而称GT0。两DEV预测冻结后再读其缓存标签评价；训练标签独立读并用于上述loss，不进入模型输入。

表示限制必须报告：若q<abs(S)，输出界[-q,q]使完美signed回归不可能。可在同stage算解析range-floor `sqrt(Σmax(abs(S)-q,0)^2/ΣS^2)`（TRAIN训练前，DEV仅端点后），只作给定q的可表示性下界，不是oracle sign预测、q准确证明或调参依据；不另开实验/改变公式。

**一次继续门建议。** 必须完整60步、两DEV allN signed S L2均<25%、DEV热点mean L2<25%、按GT应力能量加权的符号误判率≤5%（S=0忽略，ℓ=0算未正确）；缺case/null/incomplete不通过。评价还列MPa RMSE/max、worstID/原行热点及TRAIN读出，但不以它们改门。若fail，结束此固定representation，不延长原或新训练；若pass只支持扩大验证，不宣布工程或通用GeoBC solver。

部署仍需要完整被支持的FE接口：真实材料/载荷/约束、输入/算子构建、1e-8物理求解、σ恢复、q、模型加载和全N因子推理。完整cold账本包含这些实测或继承实测阶段，缓存增量另列；训练费用单列，不能把新头廉价称为整体solver廉价。约10%的q幅值误差、未知原scalar sign/export/recovery及仅两DEV都限制结论；改善不唯一归因“120步不足”或“符号才是唯一瓶颈”。
