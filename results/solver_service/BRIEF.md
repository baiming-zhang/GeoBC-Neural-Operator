# 共同 DEV2 完整服务回放

固定2026累计600s last端点：FNO2880、MLP7080；四个独立进程各一次，零训练/新scaler/GT。

| 方法 / ver CAD | 实测服务完整进程 wall(s) | 既存 U / S L2% | 本次交付 |
|---|---:|---:|---|
| fno:102_249/ver | 3.678 | 12.43 / 63.55 | completed |
| fno:102_38/ver | 2.620 | 20.64 / 58.54 | completed |
| mlp:102_249/ver | 2.037 | 35.93 / 77.23 | completed |
| mlp:102_38/ver | 2.053 | 22.41 / 70.81 | completed |

请求起点是已给 canonical xyz/25条件及可用边界faces、固定模型；完整包含进程启动/import/CUDA、input IO/cloud/grid/H2D/model、全N推理/query、物理decode/persist与退出。

原始deck、归一坐标/边界提取与25条件生成在上游：这不是 raw-CAD 端到端。缺失或budget行保留null，partialcold及阶段账本见metrics.json。

旧 zero 端点 GT U：0.016221% / 0.016224%；只属于旧端点。
新coarse仅引用既存同解/true residual证据，charged cold为36.492s / 34.420s，physical fresh仍未测，本次没有物理回放。

2027只引用既存精度；本次不重评价精度，GPU原子splat重建可能有FP32舍入差。完整FE与点/条件输入接口不同，不能拼成架构因果胜出。

U-only 与 U/S 能力分开：物理 signed scalar exporter缺口仍在。final/dia/sealed保持关闭；无warm benchmark或新增资源/数学探测。
