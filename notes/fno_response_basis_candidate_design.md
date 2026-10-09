# FNO 响应基候选（proposal_only；未实施/未派发）

**建议下一结构对照检验载荷因子化，而非继续调 Hash/GINO。**现有 [45CAD载荷guard](geobc_load_basis_45.json) 的45/45均记录 rank3、共享规范基、同CAD四load相同零SPC/材料/loaded interface、STATICS且无已列非线性卡；这支持固定线性刚度的设计假设，不证明工程小变形适用性或所有导出场严格叠加。保证仅针对已有45CAD；新CAD须另有相同metadata guard。此次只读既有source/JSON，未读场、权重或封存集。

**真实载荷坐标。**规范基按 ver/hor/tor：`b_v=(0,0,35585.77;0,0,0)`、`b_h=(-37809.9,0,0;0,0,0)`、`b_t=(0,0,0;0,0,564924.2)`，分号前单位N、后Nmm。因此 `α=(Fz/35585.77,−Fx/37809.9,Mz/564924.2)`；也可用guard保存的 `feature_to_alpha_matrix_3x6` 作用于原features7:13（F/42300、M/565000）。分别检查力/矩重构残差，不混单位拟合。dia的metadata系数为 `(0.8824848285080245,0.7478522316567883,0)`，不是凸组合；Fy/Mx/My等基外载荷必须拒绝，不声称任意六维载荷求解。

**唯一候选结构。**保留现有R64、hidden16/modes12/4层四象限FNO、PointNet1024/global32和splat/trilinear前端。共享latent只看geometry/support/material/load-location：原25中7:13先清零（缓存29格点对应11:17），其余19槽、occupancy/xyz及global32保持；不向PointNet加入当前力。latent16分成 `16→128→9` 位移基head和 `(16+当前归一化F/M六槽)→128→1` 非线性应力head，后者显式广播真实载荷。位移基保留全部signed分量；source signed scalar stress直接监督，其未知sign/export/recovery语义见 [source说明](geobc_stress_semantics.json)，不线性合成、不取abs、不从strain推导GT。

**共享scaler的仿射偏移必须只加一次。**令已有TRAIN-only位移均值/标准差为 `μ_j,s_j`；九输出 `V_kj` 表示零中心基响应 `B_kj=s_j V_kj`（mm），不是各基的常规centered编码。先在原节点查询9通道，再 `u_j=s_j Σ_k α_k V_kj`、`z_uj=Σ_k α_k V_kj−μ_j/s_j`，与原四场target比较。零载荷必得physical `u=0`，encoded为 `−μ/s`；直接相加centered基会产生 `(1−Σα)μ` 伪偏置。stress head输出既有asinh/center/std编码，节点插值后才按共享scaler解码。[FieldScaler](../geobc/benchmark.py#L29)

**最小可信对照。**两fresh arms同上述九基+应力head、参数布局/初始化、Adam.001、seed2026、同120案例四shuffle passes/480完整案例更新、全原节点四通道MSE/4；唯一差异是control共享FNO保留当前六载荷槽，candidate清零，两边相同真实α合成与stress载荷支路。control因此为load-dependent basis；mask降低有效条件容量，虽参数存储相同也不宣称同有效容量。初始自然输出无需相同，首步两head与encoder梯度/物理decode须正确。旧direct4 FNO只作历史上下文，旧Hash-u9仅启发，均不能冒充新matched control。每次更新重编码，不跨step缓存latent；同complete-case曝光，实际训练/全节点query秒与显存另报，不强行认为480等成本。checkpoint仅用known15共同U/S分数，端点主报、best次报，同时列mean/worst/hotspot；不改loss权重救某head。

**边界与停止条件。**实施前唯一需要的TRAIN-input检查是同CAD known3的xyz、非载荷features/格点模板一致及α与metadata吻合；不重扫SolverDeck，也不开dia/final。known3本身不能数值证明载荷线性，模型只强制假设下的u组合；应力组合外推仍无保证，几何泛化/BC精度也可能未改善。旧final绝不用于开发，新sealed test继续封存；任何后续组合载荷检验须root另行冻结独立未用评估范围。此note不授权训练、loss扫描或新数据审计。
