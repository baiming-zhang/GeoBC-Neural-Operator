# Elastic pilot：控制卡官方语义

2026-10-10。仅查询 Altair primary 文档；未读 deck/科学数组、未运行 GPU，`geobc/elastic_pilot.py` 保持冻结。以下“无 contact/gap、只选择 ver 线性静态，NormalModes954 未选”来自 root 已完成的卡名/子工况元数据确认，并非本次再次检查。首次 `FORMAT H3D` 被 reader 拒绝是覆盖范围问题，不是弹性模型的物理失败。

| 已有设置 | 官方含义与本轮分类 |
|---|---|
| `FORMAT H3D` | 指定 Hyper3D 二进制结果输出；可明确排除为输出格式。[FORMAT](https://help.altair.com/hwsolvers/os/topics/solvers/os/format_io_r.htm) |
| `PARAM CHECKEL NO` | 关闭单元几何质量检查，但仍做数学有效性检查。**检查策略，不是纯输出**；新模型保留自身冻结 J/有限性门，不模拟原 quality-check 策略。[CHECKEL](https://help.altair.com/hwsolvers/os/topics/solvers/os/param_checkel_bulk_r.htm) |
| `PARAM RENUMOK YES` | 允许自动重排反向、inside-out 的可接受 solid 节点序列。**输入几何/数值预处理，不是纯输出**；新模型使用已验证的原始全10节点对应与自身方向检查，不声称重放原自动重排。[RENUMOK](https://help.altair.com/hwsolvers/os/topics/solvers/os/param_renumok_bulk_r.htm) |
| `PARAM PRINFACC 1` | 将惯性释放刚体力和加速度写入 `.out`；这是输出开关，本身不启用惯性释放。[PRINFACC](https://help.altair.com/hwsolvers/os/topics/solvers/os/param_prinfacc_bulk_r.htm) |
| `PARAM CONTFEL AUTO` | 非线性静态/瞬态中的 contact-friendly element 设置；AUTO 针对 N2S 接触的二阶 solid/gasket。基于已确认的无接触、线性静态 scope，判断本轮不激活；这是**条件推断**，不是一般性“输出卡”。[CONTFEL](https://help.altair.com/hwsolvers/os/topics/solvers/os/param_contfel_bulk_r.htm) |
| `GAPPRM GAPCMPL NO` | 从 global compliance 计算中排除 gap/contact compliance；不等于关闭接触力学。当前无 gap/contact 时没有相应贡献，故本轮不生效（基于 root 的 scope 推断）。[GAPCMPL](https://2021.help.altair.com/2021.1/hwsolvers/os/topics/solvers/os/gapprm_gapcmpl_bulk_r.htm) |
| 未选 `METHOD 954 / EIGRL 954` | METHOD 选择实特征值抽取，EIGRL 定义 Lanczos 振动/屈曲特征值计算；不是所选静态载荷方程。必须记录未选工况，不把它归为输出。[METHOD](https://help.altair.com/hwsolvers/os/topics/solvers/os/method_sub_r.htm)、[EIGRL](https://help.altair.com/hwsolvers/os/topics/solvers/os/eigrl_bulk_r.htm) |

**`DEBUG,LGSHPSNS,0.0` 的边界：**官方 [DEBUG 页面](https://help.altair.com/hwsolvers/os/topics/solvers/os/debug_io_r.htm) 只说明 DEBUG 是 Altair Support 的内部问题调试入口，给出 name/option 格式；未定义 `LGSHPSNS`，本次限定检索也未找到该 option 的公开官方公式或取值表。因而**不能确认其准确含义、不能断言 0 禁用、不能称纯输出或已证明在 static 中 inactive**。不根据缩写猜测 shape sensitivity/刚度行为。

若 root 继续明确的新近似模型，可将该**准确名称与值**记录为“原求解器内部设置，本模型不模拟，其对原软件行为的影响未知”，而不是伪造 disabled 语义。上述 CHECKEL/RENUMOK 同样单列未重放的预处理策略。此排除只能支持公开的 approximate-model scope，不能支持 OptiStruct exact replay 或将之后的误差归因于材料/载荷唯一因素；具体 parser allowlist/source 变更仍由 root/data author 负责，本次不改科学源码。
