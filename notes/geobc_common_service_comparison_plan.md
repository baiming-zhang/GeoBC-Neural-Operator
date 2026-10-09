# GeoBC 共同 DEV2 服务比较：当前实际范围

`source_preparation`，当前实际范围已选定；源码未审查释放前不执行。本次仅 completed 小 JSON；没有训练、scaler/GT/数组/权重读取、GPU、final 或 sealed 测试。下一目标是**一次单请求的完整成本与交付能力**，不继续 Hash、coarse 参数或失败 stress head 微调。

共同请求固定 `102_249/ver`、`102_38/ver`。主端点固定 seed2026 累计 600s 的 `last.pt`：FNO step2880、MLP step7080；不使用 validation-best 或按结果挑 seed。seed2027 只引用既存精度稳定性。

| 主端点；全原节点、物理单位 | 102_249/ver：U / S / S-hot L2% | 102_38/ver：U / S / S-hot L2% |
|---|---:|---:|
| FNO2026 | 12.43 / 63.55 / 50.13 | 20.64 / 58.54 / 48.31 |
| MLP2026 | 35.93 / 77.23 / 77.00 | 22.41 / 70.81 / 71.07 |

2027 对应 U/S：FNO 14.93/61.22、21.18/56.19%；MLP 40.11/75.70、17.19/71.85%。来源为 `runs/geobc_exploration/solver_confirm_{fno,mlp}_seed{2026,2027}_600s/summary.json` 的 `final_metrics.validation.per_case`；四项 completed，最后 record 与 endpoint_step 一致。逐例 U RMS/max、S RMSE/MAE/hotspot 数亦可直接引用 JSON，不重新评价。

**物理证据单列。** 同前缀 `elastic_crosscad5_ver/{102_249_ver,102_38_ver}/summary.json` 旧 zero 端点 GT U L2 是 0.0162207% / 0.0162243%。coarse 新端点来源 `elastic_coarse2_ver/102_249_ver/summary.json` 与 `elastic_coarse5_ver_extension/102_38_ver/summary.json`，true residual <1e-8、旧数值解 L2 差 7.998e-11 / 3.611e-10；GT0，不给新端点套旧精度。coarse charged-cold 为 36.4921 / 34.4204s（降63.88% /62.36%），包含继承 fine setup、新 case wall、参考/检查点；`fresh_combined_cold_remeasured=false`。

**接口与需求。** 学习接口：节点 xyz +25 BC/material +PointNet32（FNO 另需64³网格）；物理：完整 TET10、材料和真实 SPC/RBE2/RBE3。信息不同，仅比较服务，不做架构因果排名。需求暂保留 U-only（原节点 ux/uy/uz mm）与完整 U/S（另含原 CSV signed stress MPa）。物理 signed-S 缺口见 `notes/geobc_signed_output_interface.md`：CSV selector/recovery/exporter 未知，不能手工定 sign、以 tensor/proxy 替代标签或救训 factored60 head。

**当前实际范围已收窄为必要缺项。** 仅四个独立干净进程：FNO2026、MLP2026各对固定DEV2/ver交付一次全原节点场；不重新回放物理，2027不重复推理。物理旧GT精度、新coarse同数值解和accounted cold分别引用，fresh联合请求仍未重测。源准备后根必要审核才释放；禁止资源/toy/科学预验。

请求起点明确为已给的无目标节点xyz与25条件。实际NPZ只读输入member；不能用会打开stress/displacement的旧loader。独立PointNet1024 cloud、FNO64³网格、输入IO/H2D、模型与CUDA启动、checkpoint/scaler载入、全N推理、原decode及输出持久化全部计时；parent launch到child退出全程费用不遗漏。原CAD/deck到25条件生成不在这一接口内，明确missing_upstream与raw_CAD_end_to_end=false，不宣称rawCAD端到端。完成本接口冷流程才可记complete_cold_cost_accounted=true；cached模型/网格不免费。

累计900秒、每请求180秒、峰值24GiB；resume保留真实attempt、累计时间和失败ledger，已完成请求不重放。零GT/训练/scaler拟合/oldfinal/DEVdia/sealed；仍读取固定模型checkpoint与实际输入并准确计账。固定原signedS MPa及uXYZ mm全节点落盘，精度只引用原completed JSON，不重新评价。

**输出与推进。** 一张共同请求表，分别给 U-only/完整U-S的能力、既存误差、实测服务成本、输入前置条件及 missing项。完整FE允许且只需U时保留coarse候选；点云接口或完整U/S需求则按固定FNO/MLP实测服务判断，物理signed缺口未解前不宣布完整服务胜出。用户接口/精度预算确定后，才冻结 CAD×load 扩展；sealed 全服务协议固定后一次测试，不拆开反复调模型。
