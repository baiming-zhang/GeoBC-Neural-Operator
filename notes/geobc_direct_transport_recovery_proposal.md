# 剩余223份原始输入：独立传输恢复建议

状态：**proposal_only，当前不释放下载或输入准备**。原540获取已按旧期限终态为317/540、budget negative；其clock、失败、unknown费用与summary永久保留，不重启或覆盖。十臂科学配方、20源pins、固定CAD清单和DEV标签屏障不变。

本次已有明确的新传输组合：私有GitHub出版的Windows .NET直连、HTTP/1.1 exact实际成功；旧HF直连是Python urllib，旧.NET则强制本地代理。这足以支持一次不同组合的有限HEAD，不能把GitHub可达当作HF可达。

**唯一新HEAD结果：失败，20.053秒。** 对既有官方HF revision `160424ca2a5095315ff9ead8a5fc36d2667fd3c6` 的FieldMesh公共源，使用Windows .NET、`UseProxy=false`、`RequestVersionExact=1.1`；20秒连接期限/45秒总上限、不重定向。未收到HTTP状态，安全异常链为TaskCanceledException→TimeoutException。HEAD1、GET0、正文读取0、负载写盘0；TLS/请求头网络字节未测，不能称总网络字节0，也不能认定DNS、TLS或配额是哪一层失败。记录：[独立HEAD receipt](geobc_hf_direct_http11_HEAD_receipt.json)。没有第二次探针。

因此，**当前不能证明有可执行的223文件恢复路线，不值得授权约127GB的HF顺序gzip交付**。旧Drive quota HTML、后续MCP客户端HTTP发送错误和HF超时是不同观测；清理本调用代理不自动消除服务端配额或连接器故障。保留317个实际成功binary与receipt，维持本轮已训练证据，等待明确的新代理/网络或连接器状态变化，避免在相同路线耗费请求。

若未来出现可核实的新路线，根可另行冻结一次小额传输阶段：固定一个仍缺的官方文件/固定HF revision，独立180秒、最多2MiB新正文、无自动重试；HF只在官方HTTPS重定向与精确Range/Content-Range/大小都确认后收取范围，Drive则要求真实binary而非quota页面。只报实际正文bytes与墙钟、保留HTTP/错误类别，不解析H5/CSV/FEM字段。失败即关停该路线；成功只证明这份有限交付，不证明223份或全档案完成，需另冻结完整预算、磁盘和恢复协议。

**进入旧准备器还有真实契约缺口。** `coverage_comparison_data.py:155–199`固定读取原raw summary，要求540全部completed、每份URL等于原Drive清单，并与canonical receipt的SHA/stat完全一致。独立HF产物不能假装Drive下载，也不能将旧negative改成completed。必须由根另审独立来源adapter：旧317仅只读引用；新223记录真实HF archive/revision、member路径、实际字节SHA/stat和传输账本，单独生成新完整inventory并映射原540逻辑ID。提取时可比较317重叠member与旧receipt SHA；同名不等于byte-identical，出现版本/字节差异应阻塞，不换CAD、不换DeepJEB++。来源适配不能自动授权DEV标签解析或续训。

本次仅小JSON/源码读取与这一HEAD；科学数组、标签、权重、GPU、档案下载/解压均0。当前建议是**不再探针、不启动恢复；待真实外部路线变化后再决定一个独立小额阶段**。
