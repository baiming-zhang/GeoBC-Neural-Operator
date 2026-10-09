"""A compact, evidence-driven report for the autonomous GeoBC experiments."""
from pathlib import Path
import json
import math
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geobc.train import write_json,utc_now

def percent(value):
    if not math.isfinite(float(value)): raise ValueError('Nonfinite metric')
    return f'{100*value:.3f}%'

def main():
    plan=json.loads((ROOT/'notes/geobc_exploration_plan.json').read_text())
    dest=ROOT/'results/geobc_exploration'
    dest.mkdir(parents=True,exist_ok=True)
    summaries={}
    checkpoints={}
    jobs=plan.get('historical_jobs',[])+plan['jobs']
    for job in jobs:
        source=ROOT/job['output']
        path=source/'summary.json'
        if not path.exists(): continue
        summary=json.loads(path.read_text())
        if summary.get('status')!='completed': continue
        summaries[job['name']]=summary
        checkpoint_files={}
        for filename in ('best.pt','last.pt','endpoint_250.pt','endpoint_10000.pt'):
            checkpoint=source/filename
            if checkpoint.is_file():
                stat=checkpoint.stat()
                checkpoint_files[filename]={'local_path':str(checkpoint),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if checkpoint_files:
            checkpoints[job['name']]={'best_step':summary.get('best_step'),'files':checkpoint_files,
                                      'code_sha256':summary.get('config',{}).get('contract',{}).get('code_sha256',{}),
                                      'selection':summary.get('config',{}).get('checkpoint_selection')}
        target=dest/'runs'/job['name']; target.mkdir(parents=True,exist_ok=True)
        for filename in ('summary.json','config.json','metrics_best.json','metrics_latest.json','metrics.jsonl','parent_comparison.json','metrics_endpoint_250.json','metrics_endpoint_10000.json'):
            if (source/filename).exists(): shutil.copyfile(source/filename,target/filename)
    write_json(dest/'checkpoint_index.json',{'weights_published':False,'weights_location':'local training directories',
                                            'identity_policy':'paths/bytes/mtime plus small config source hashes; no redundant checkpoint scans',
                                            'runs':checkpoints})
    manifests=dest/'data_manifests'; manifests.mkdir(exist_ok=True)
    for directory in ('prepared_geobc_transfer','prepared_geobc_multiload','geobc_local_geometry','geobc_frozen_u_response','geobc_frozen_strain_response','geobc_gradient_supervision'):
        source=ROOT/'data'/directory/'manifest.json'
        if source.is_file():shutil.copyfile(source,manifests/(directory+'.json'))
    lines=['# GeoBC 探索：当前结论与下一步','',
           '研究对象：原版 DeepJEB；使用逐节点坐标对齐的原始有符号应力。','',
           '## 当前结论','',
           '**Hash 的单案例表示能力很强；当前共享模型的跨几何、跨载荷组合迁移很差。**','']
    geo=summaries.get('hash50_unseen_geometry_pilot')
    load=summaries.get('hash50_unseen_load_combinations')
    parent=json.loads((ROOT/'runs/slides_reproduction/hash50_full1000ep/metrics_best.json').read_text())
    if geo and load:
        lines += [f"- 原共享模型训练集平均应力 L2：{percent(parent['case_mean_relative_l2'])}。",
                  f"- 两个未参与该模型训练的 CAD：平均 {percent(geo['metrics']['case_mean_relative_l2'])}。",
                  f"- 三个已见 CAD 的九个新载荷组合：平均 {percent(load['metrics']['case_mean_relative_l2'])}。",'']
    comparisons=[]
    for cad in ('101_479','107_320','114_202'):
        summary=summaries.get('independent_'+cad+'_tor')
        if summary:
            item=summary['parent_comparison'][cad+'/tor']
            comparisons.append((cad,item['parent_shared50']['relative_l2'],item['independent_fit']['relative_l2']))
    if comparisons:
        lines+=['| 案例 | 原共享模型 | 独立拟合 |','|---|---:|---:|']
        lines += [f'| {cad}/tor | {percent(shared)} | {percent(independent)} |' for cad,shared,independent in comparisons]
        lines+=['','这些对照固定了架构、输入和父模型 scaler；独立模型增加了总容量，不能单独证明某个几何模块必需。','']
    lines+=['## 几何条件能否改善泛化','',
            '使用同一 40-CAD 训练／5-CAD 验证划分。每组对照的 decoder 布局、随机采样、预算相同；scaler 只拟合训练标签。每项训练10,000次更新，不能与原50案例的1000个完整epoch直接等同。',
            '训练与选模型只使用 ver/hor/tor；五个官方测试 CAD 保持隔离。验证CAD的dia曾作一次模型冻结后的诊断，后续不能再称全新盲测。','']
    paired=[
        ('整体几何／宽度64','hash_bc_no_geometry_10k','hash_bc_pointnet_10k'),
        ('整体几何／宽度128','hash_bc_no_geometry_width128_10k','hash_bc_pointnet_width128_10k'),
        ('局部几何／宽度128，均有PointNet','hash_bc_pointnet_local_zero_width128_10k','hash_bc_pointnet_local_actual_width128_10k'),
        ('局部几何／同配置，种子2027','hash_bc_pointnet_local_zero_width128_seed2027_10k','hash_bc_pointnet_local_actual_width128_seed2027_10k'),
    ]
    available=[(label,summaries.get(zero),summaries.get(actual)) for label,zero,actual in paired]
    available=[row for row in available if row[1] or row[2]]
    if available:
        lines+=['| 同配置对照 | 零几何特征：验证误差 | 实际几何特征：验证误差 |','|---|---:|---:|']
        metric=lambda summary:percent(summary['best_metrics']['validation']['case_mean_relative_l2']) if summary else '待完成'
        lines += [f'| {label} | {metric(zero)} | {metric(actual)} |' for label,zero,actual in available]
        lines += ['', '整体PointNet的收益不足0.04个百分点，加宽解码器后最佳验证误差仍约65.3%；目前不能认为这些改动改善了泛化。',
                  '局部几何对照使用13维离散边界距离、方向和邻域形状特征；只从几何构造，无标签拟合。同组零特征模型保留相同输入槽与参数布局。']
    zero=summaries.get('hash_bc_pointnet_local_zero_width128_10k')
    actual=summaries.get('hash_bc_pointnet_local_actual_width128_10k')
    if zero and actual:
        gap=100*(zero['best_metrics']['validation']['case_mean_relative_l2']-actual['best_metrics']['validation']['case_mean_relative_l2'])
        lines += [f'首个种子的局部特征收益为{gap:.2f}个百分点；第二个种子用于检查收益是否稳定，高误差仍未解决。']
        z2=summaries.get('hash_bc_pointnet_local_zero_width128_seed2027_10k')
        a2=summaries.get('hash_bc_pointnet_local_actual_width128_seed2027_10k')
        if z2 and a2:
            gap2=100*(z2['best_metrics']['validation']['case_mean_relative_l2']-a2['best_metrics']['validation']['case_mean_relative_l2'])
            lines += [f'第二个种子的收益为{gap2:.2f}个百分点。两次均改善平均与热点误差，但最差案例变差；不能称为全面改善。']
    lines += ['', '## 位移的载荷响应基','',
              '已实现ver/hor/tor位移响应的线性组合，并与同容量、允许响应基依赖载荷的控制模型比较。几何输入、训练预算和随机采样保持相同。',
              '45个训练/验证CAD的求解配置支持该设定；源FEM叠加残差见下方冻结诊断。训练只拟合原始uXYZ，使用训练集的零中心线性尺度；应力保留独立监督。']
    physics=[]
    for mode in ('linear_basis','load_dependent_basis'):
        summary=summaries.get('displacement_'+mode+'_local_actual_10k')
        if summary:
            physics.append((mode,summary['best_metrics']['train']['case_mean_relative_l2'],summary['best_metrics']['validation']['case_mean_relative_l2']))
    if physics:
        lines += ['', '| 位移响应基 | 最佳点训练误差 | 验证误差 |','|---|---:|---:|']
        lines += [f"| {'载荷无关，强制线性' if mode=='linear_basis' else '载荷依赖控制'} | {percent(train)} | {percent(val)} |" for mode,train,val in physics]
    relative=[]
    for mode in ('linear_basis','load_dependent_basis'):
        summary=summaries.get('displacement_'+mode+'_relative_case_10k')
        if summary:
            relative.append((mode,summary['best_metrics']['validation']['case_mean_relative_l2'],summary['best_metrics']['validation']['worst_case_relative_l2']))
    if relative:
        lines += ['', '训练集证据：相同相对误差下，扭转载荷仅占原共同RMS损失约1.3%的权重。改为训练案例RMS归一化损失，物理解码尺度保持不变：']
        lines += [f"- {'线性响应基' if mode=='linear_basis' else '载荷依赖对照'}：验证平均{percent(val)}，最差{percent(worst)}。" for mode,val,worst in relative]
        lines += ['训练扭转拟合明显改善，但验证平均收益小、最差案例变差，幅值权重不是全部泛化瓶颈。']
    probe=summaries.get('frozen_validation_dia_probe')
    if probe:
        u=probe['displacement']
        lines += ['', '冻结后仅评估五个验证CAD的dia组合（933,951个原节点）：',
                  f"- 位移平均误差：线性响应基{percent(u['linear_basis']['case_mean_relative_l2'])}，载荷依赖对照{percent(u['load_dependent_basis']['case_mean_relative_l2'])}。",
                  f"- 源FEM位移叠加残差平均{percent(u['oracle']['case_mean_relative_l2'])}，支持近似线性；不能说数值残差为零或已确定其来源。",
                  f"- 独立应力模型的dia平均误差仍为{percent(probe['stress']['case_mean_relative_l2'])}，组合载荷应力预测仍是主要瓶颈。",
                  '这是验证集载荷组合诊断，未读取最终测试CAD。线性模型的最大绝对位移误差仍可能高于对照，完整物理指标保留在原始summary。']
    uz=summaries.get('stress_u_condition_zero_width128_10k')
    ua=summaries.get('stress_u_condition_actual_width128_10k')
    lines += ['', '## 冻结预测位移作为应力条件','',
              '按常规载荷验证均值选择并冻结案例归一化损失的线性位移教师（29.178%）。每CAD只推断一次三组响应，再按当前载荷组合成三维预测位移；应力模型的对应三个槽使用预测值或全零。两者均从头训练，相同41维局部输入、32维整体描述、采样与10k预算。',
              '预测条件生成不读取真实位移或应力；教师本身使用额外的训练位移监督，因此整个pipeline不能声称与stress-only模型有相同监督量或总成本。两个候选教师训练成本合计约223秒，缓存生成和应力训练成本另列。']
    if uz or ua:
        lines += ['', '| 位移条件 | 验证应力均值 | 最差案例 | 热点均值 | 训练耗时 |','|---|---:|---:|---:|---:|']
        for mode,summary in [('零槽',uz),('冻结预测位移',ua)]:
            if summary:
                v=summary['best_metrics']['validation']
                lines += [f"| {mode} | {percent(v['case_mean_relative_l2'])} | {percent(v['worst_case_relative_l2'])} | {percent(v['case_mean_hotspot_relative_l2'])} | {summary['elapsed_seconds']:.1f}s |"]
            else:
                lines += [f'| {mode} | 待完成 | — | — | — |']
    else:
        lines += ['成对实验已排队，等待实现及共享响应缓存。此轮不打开dia或最终测试CAD。']
    if uz and ua:
        lines += ['位移条件改善了均值与最差案例，但热点变差；未通过预定的复验门。下一轮固定预测u3，比较廉价mesh恢复应变六槽与同布局全零槽，检验局部变形变化是否提供不同的信息。它是corner-P1离散代理，不是精确TETRA10应变，也不以本构公式替换源应力标签。']
    sz=summaries.get('stress_u_actual_strain_zero_width128_10k')
    sa=summaries.get('stress_u_actual_strain_actual_width128_10k')
    if sz or sa:
        lines += ['', '## 固定预测位移，增加局部应变条件','',
                  '双方保持同一预测u3、几何、应力loss、采样和预算。TETRA10前四角点计算cell常梯度，以体积向原十节点平均；六个tensor应变分量只使用训练CAD预测应变的共同零中心RMS缩放。47维局部条件加32维整体描述，zero/actual布局和初始化匹配。该代理忽略二次单元内应变变化。',
                  '', '| 应变条件 | 验证应力均值 | 最差案例 | 热点均值 | 训练耗时 |','|---|---:|---:|---:|---:|']
        for mode,summary in [('零槽',sz),('预测P1应变',sa)]:
            if summary:
                v=summary['best_metrics']['validation']
                lines += [f"| {mode} | {percent(v['case_mean_relative_l2'])} | {percent(v['worst_case_relative_l2'])} | {percent(v['case_mean_hotspot_relative_l2'])} | {summary['elapsed_seconds']:.1f}s |"]
            else:
                lines += [f'| {mode} | 待完成 | — | — | — |']
    quality_path=ROOT/'notes/geobc_fixed_train_strain_quality.json'
    quality=json.loads(quality_path.read_text()) if quality_path.is_file() else None
    if quality and quality.get('status')=='completed':
        q=quality['aggregate']
        lines += ['', '## 节点位移准确，不等于局部梯度准确','',
                  f"固定训练CAD101_479的三载荷平均位移误差为{percent(q['u_mean_vector_relative_l2'])}，由相同corner-P1算子恢复的tensor应变误差为{percent(q['strain_mean_symmetric_Frobenius_relative_l2'])}，约为位移相对误差的{q['mean_error_ratio_strain_to_u']:.1f}倍。预测应变量级为参考的约9–11倍。CPU诊断耗时{quality['cost']['elapsed_seconds']:.2f}s，没有读取应力、验证、dia或最终测试字段。",
                  '这支持当前冻结教师的导数质量不足，不能把应变条件无整体收益解释成应变信息无用。参考应变由已有训练u派生，是四角点P1代理，不是solver输出的真应变张量。结论目前仅限该固定训练CAD。',
                  '', '![固定训练CAD的位移与恢复应变误差](strain_quality_diagnosis.png)', '',
                  '下一组保持原载荷线性位移模型，两侧使用相同的节点/单元角点混合采样与51,200次前向查询，仅比较是否增加训练u派生的体积加权P1梯度监督。选checkpoint仍只看已知载荷验证位移均值；最终测试及dia继续隔离。设计与后续判据提前保存于notes/geobc_gradient_teacher_plan.json。',
                  '函数值与导数共同监督已有 [Sobolev Training](https://papers.nips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html) 工作。本实验使用位移派生的离散P1应变目标，验证其对GeoBC教师及跨CAD迁移的作用；加入梯度损失本身尚不构成新算法结论。']
        precision_path=ROOT/'notes/geobc_u_output_precision.json'
        if precision_path.is_file():
            precision=json.loads(precision_path.read_text())
            fp16=precision['methods']['fp16_oracle_roundtrip']
            lines += [f"同一训练CAD的真实u按冻结教师共同尺度做FP16往返舍入，u误差为{percent(fp16['u_mean_vector_relative_l2'])}、恢复应变误差为{percent(fp16['strain_mean_symmetric_Frobenius_relative_l2'])}。这远不足解释当前约941%的应变误差；这是oracle舍入敏感性诊断，不是严格误差下界或因果分解，也不能排除内部算术等其他因素。"]
    gz=summaries.get('displacement_gradient_zero_10k')
    ga=summaries.get('displacement_gradient_actual_10k')
    if gz or ga:
        lines += ['', '## 匹配的位移梯度监督','',
                  '两侧使用同一load-linear模型、训练u共同尺度、相同混合节点/角点查询和初始化。相较历史教师，两侧共同将一半查询改为均匀单元的角点；内部对照仅改变梯度目标的系数0或1。梯度参考来自训练u，按单元体积及完整训练案例Frobenius能量归一化，验证目标不拟合损失权重。各checkpoint仍只由验证位移均值选择。',
                  '', '| 梯度损失 | 验证位移均值 | 验证位移最差 | 验证P1应变均值 | 训练耗时 |',
                  '|---|---:|---:|---:|---:|']
        for mode,summary in [('系数0',gz),('系数1',ga)]:
            if summary:
                v=summary['best_metrics']['validation']
                lines += [f"| {mode} | {percent(v['case_mean_relative_l2'])} | {percent(v['worst_case_relative_l2'])} | {percent(v['strain']['case_mean_relative_l2'])} | {summary['elapsed_seconds']:.1f}s |"]
            else:
                lines += [f'| {mode} | 待完成 | — | — | — |']
        gate_path=ROOT/'notes/geobc_gradient_teacher_pair_result.json'
        if gz and ga and gate_path.is_file():
            gate=json.loads(gate_path.read_text())
            lines += [f"在同一u-selected checkpoint，验证应变误差相对减少{100*gate['validation_strain_relative_error_reduction']:.1f}%，固定训练101_479减少{100*gate['fixed_train_101_479_strain_relative_error_reduction']:.1f}%；验证位移均值变差{gate['validation_u_mean_deterioration_percentage_points']:.2f}个百分点、最差变差{gate['validation_u_worst_deterioration_percentage_points']:.2f}个百分点。预定后续判据{'通过' if gate['clear_gain_gate_passed'] else '未通过'}。"]
            if not gate['clear_gain_gate_passed']:
                lines += ['随后检查已导出的匹配代表案例是否有支撑边界整体位移偏移。只用预测u在已知零位移约束处的均值来计算偏移，真实u用于评价；常量平移不改变应变。这是选定代表的CPU诊断，完整验证与后续模型变更另行声明。']
    boundary_path=ROOT/'notes/geobc_boundary_rigid_offset.json'
    if boundary_path.is_file():
        boundary=json.loads(boundary_path.read_text())
        if boundary.get('status')=='completed':
            reps=boundary['per_method']['gradient_actual']
            values=[(key,reps[key]['raw']['vector_relative_l2'],reps[key]['corrected_using_zero_BC']['vector_relative_l2']) for key in ('107_91/hor','102_249/tor')]
            text='；'.join(f"{key}：{percent(raw)} → {percent(corrected)}" for key,raw,corrected in values)
            lines += ['', '## 边界均值与刚体偏移的代表诊断','',
                      f"将预测位移在原SPC节点上的分量均值归零，两个已保存验证代表的位移误差为{text}。训练代表有改善，但验证没有一致收益；未扩为全45CAD校正。CPU耗时{boundary['elapsed_seconds']:.2f}s，常量偏移保持P1梯度不变。GT只用于指标，未用GT误差计算偏移；零均值也不等于逐节点严格满足Dirichlet约束。",
                      '后续固定一个训练历史校准系数：取零梯度基线step6000–10000五个既定训练minibatch的Lu/裸Lstrain中位数，得到0.000410665；没有验证调权重或额外乘数。独立入口仅改变scope/parser，计算函数与原版本逐函数AST及文本相同，原零控制复用。它是一次loss尺度控制，不是最优权重或参数梯度平衡。']
    gc=summaries.get('displacement_gradient_calibrated_10k')
    if gc:
        v=gc['best_metrics']['validation']
        lines += ['', f"固定训练系数结果：验证位移均值{percent(v['case_mean_relative_l2'])}，最差{percent(v['worst_case_relative_l2'])}，P1应变均值{percent(v['strain']['case_mean_relative_l2'])}；u-selected step{gc['best_step']}，耗时{gc['elapsed_seconds']:.1f}s。配对控制与预定判据见notes/geobc_gradient_calibrated_result.json。"]
        calibrated_path=ROOT/'notes/geobc_gradient_calibrated_result.json'
        if calibrated_path.is_file():
            calibrated= json.loads(calibrated_path.read_text())
            lines += [f"均值有改善，但最差位移误差退步{calibrated['validation_u_worst_deterioration_percentage_points']:.2f}个百分点，预定综合标准未通过；P1误差仍高于预测全零应变的100%基准。因此没有进入第二种子或应力教师阶段，也没有继续扫损失权重。"]
    scale_audit_path=ROOT/'notes/geobc_physical_scale_input_audit.json'
    if scale_audit_path.is_file():
        scale_audit=json.loads(scale_audit_path.read_text())
        if scale_audit.get('status')=='complete':
            lengths=scale_audit['length_statistics']['all45']
            lines += ['', '## 下一阶段：整体边界角色条件','',
                      f"元数据核验确认原25通道已包含L_mm/200；45CAD尺寸仅{lengths['min_mm']:.2f}–{lengths['max_mm']:.2f}mm，暂缓显式尺寸缩放。当前PointNet只汇聚坐标；新对照在相同1024点云附加固定节点并集和加载节点两个输入角色。零角色与真实角色都采用5输入PointNet，新增64参数在两侧相同，不能复用旧3输入模型作控制。",
                      '保持既定位移／梯度损失、采样和预算，以新的零角色模型为基线比较全节点物理位移及P1代理应变。最差案例保护与不超过全零应变100%基准的门槛已预定；未通过时不自动增加种子或扫参数。设计见notes/geobc_bc_role_plan.json。']
    bz=summaries.get('displacement_bc_role_zero_10k')
    ba=summaries.get('displacement_bc_role_actual_10k')
    bc_result_path=ROOT/'notes/geobc_bc_role_pair_result.json'
    if bz or ba:
        lines += ['', '| 整体BC角色 | 验证位移均值 | 验证P1应变均值 | 训练耗时 |',
                  '|---|---:|---:|---:|']
        for mode,summary in [('零角色',bz),('真实角色',ba)]:
            if summary:
                v=summary['best_metrics']['validation']
                lines += [f"| {mode} | {percent(v['case_mean_relative_l2'])} | {percent(v['strain']['case_mean_relative_l2'])} | {summary['elapsed_seconds']:.1f}s |"]
            else:
                lines += [f'| {mode} | 进行中 | — | — |']
        if bz and ba and bc_result_path.is_file():
            bc_result=json.loads(bc_result_path.read_text())
            lines += [f"匹配控制通过。真实角色使验证位移均值改善{bc_result['validation_u_mean_improvement_percentage_points']:.2f}个百分点，最差位移变化{bc_result['validation_u_worst_deterioration_percentage_points']:+.2f}个百分点（正数为退步）；P1误差相对改善{100*bc_result['validation_P1_relative_error_reduction']:.1f}%。预定综合门槛{'通过' if bc_result['clear_gain_gate_passed'] else '未通过'}。",
                      f"同一u-selected checkpoint的训练位移均值改善{bc_result['train_u_mean_improvement_percentage_points']:.2f}个百分点；训练P1误差相对改善{100*bc_result['train_P1_relative_error_reduction']:.1f}%。该结果限已反复使用的开发验证，不是最终盲测。"]
    if (ROOT/'results/geobc_exploration/gradient_tradeoff.png').is_file():
        lines += ['', '![位移与P1梯度精度的取舍](gradient_tradeoff.png)',
                  '图中每点均取该模型按位移验证均值选择的同一checkpoint；同形状标记区分各组对照。BC增益仅比较两个新方块，不能把旧三输入校准点与新五输入零角色点的差异归于BC信息。']
    specialist_design_path=ROOT/'notes/geobc_specialist_plan.json'
    if specialist_design_path.is_file():
        specialist_design=json.loads(specialist_design_path.read_text())
        if specialist_design.get('status')=='selected_after_failed_BC_gate':
            lines += ['', '## 下一阶段：固定训练CAD独立拟合','',
                      '固定已观察的TRAIN101_479/ver、hor、tor，采用原xyz3模型与同一校准损失，共享模型的scaler和案例权重不重拟合。报告固定250与10000步，不用验证或应变挑选最优checkpoint。主要比较共享模型固定10000步的该CAD：位移16.85%、P1应变293.24%。',
                      '单CAD每步四个载荷槽有放回抽样，这是显式诊断变化。250步仅匹配共享10000步的期望案例曝光，完整10000步约有40倍曝光；独立模型也增加总参数。即使独立拟合成功，也不能单独证明共享容量不足或获得泛化能力。设计见notes/geobc_specialist_plan.json。']
    lines += ['', '## 解释边界','',
              '两个 CAD 的结果是迁移 pilot，存在其他历史实验使用；不是最终盲测。九个组合的四类载荷均已在其他训练 CAD 出现，不能称为全局未见载荷类别。新模型的最终测试CAD仅指未参与其训练/选模，原50案例历史实验包含这些CAD。',
              '迁移尾部误差可能被 asinh 逆变换放大；不截断预测、不更改标签，也不把训练误差称为泛化误差。',
              '候选模型组合已有的 [PointNet](https://arxiv.org/abs/1612.00593) 点集汇聚与 [Instant-NGP](https://nvlabs.github.io/instant-ngp/) hash 表示；当前尚无新算法或泛化成功的结论。','',
              f"本页随实验更新；累计完成 {len(summaries)}/{len(jobs)} 项，自主研究继续。",'']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    brief=['# GeoBC：一分钟结果','',
           '**目前的主要问题是共享场的泛化，而非单个案例能否拟合。**','',
           '- 难例单独训练：应力误差约0.1%；原共享Hash换CAD或载荷组合：约92%。',
           '- 整体几何编码和解码器加宽没有明显收益；局部边界几何仅将应力验证误差从约65%改善到约64%。']
    if zero and actual:
        brief += ['  局部几何收益在两个种子上重复，但最差案例变差，不能称为全面改善。']
    if probe:
        brief += ['- 组合载荷诊断：强制位移线性响应优于对照（28.4% vs 33.0%），但应力仍约87%；位移物理结构有用，应力迁移仍未解决。']
    if bz and ba and bc_result_path.is_file():
        bc_result=json.loads(bc_result_path.read_text())
        z=bz['best_metrics']['validation'];a=ba['best_metrics']['validation']
        brief += ['', f"**最新对照：**整体点云加入固定／加载角色，位移验证均值{100*z['case_mean_relative_l2']:.1f}% → {100*a['case_mean_relative_l2']:.1f}%，最差{100*z['worst_case_relative_l2']:.1f}% → {100*a['worst_case_relative_l2']:.1f}%，P1应变代理误差仍约{100*a['strain']['case_mean_relative_l2']:.0f}%。综合门槛{'通过' if bc_result['clear_gain_gate_passed'] else '未通过'}。下一步固定一个训练CAD独立拟合位移与应变，定位共享训练的瓶颈；最终测试继续保留。"]
    elif gc and gz:
        z=gz['best_metrics']['validation'];a=gc['best_metrics']['validation']
        delta=100*(a['worst_case_relative_l2']-z['worst_case_relative_l2'])
        brief += ['', f"**最新对照：**固定训练损失尺度后，位移验证均值{100*z['case_mean_relative_l2']:.1f}% → {100*a['case_mean_relative_l2']:.1f}%，P1应变误差{100*z['strain']['case_mean_relative_l2']:.1f}% → {100*a['strain']['case_mean_relative_l2']:.1f}%。最差位移退步{delta:.1f}个百分点，仍未通过综合标准。下一步检验整体点云中的固定／加载位置条件；最终测试继续保留。"]
    elif gz and ga:
        z=gz['best_metrics']['validation'];a=ga['best_metrics']['validation']
        brief += ['', f"**最新对照：**梯度监督使验证P1应变误差{100*z['strain']['case_mean_relative_l2']:.1f}% → {100*a['strain']['case_mean_relative_l2']:.1f}%，位移误差{100*z['case_mean_relative_l2']:.1f}% → {100*a['case_mean_relative_l2']:.1f}%。导数明显改善，位移变差；支撑均值校正也没恢复验证精度。正在测试一个按训练损失尺度计算的固定系数。应变是位移派生的P1代理，最终测试继续保留。"]
    elif quality and quality.get('status')=='completed':
        q=quality['aggregate']
        brief += ['', f"**关键新发现：**固定训练CAD的位移误差约{100*q['u_mean_vector_relative_l2']:.0f}%，恢复的应变误差却约{100*q['strain_mean_symmetric_Frobenius_relative_l2']:.0f}%。当前Hash教师的局部梯度不可靠，增加应变条件没有整体收益。下一步直接对照梯度监督，检验能否修复这一问题。此应变为训练u派生的P1代理，结论限该CAD。"]
    elif sz and sa:
        z=sz['best_metrics']['validation'];a=sa['best_metrics']['validation']
        brief += ['', f"最新对照：固定预测位移后，增加局部应变条件，应力验证均值{percent(z['case_mean_relative_l2'])} → {percent(a['case_mean_relative_l2'])}，热点{percent(z['case_mean_hotspot_relative_l2'])} → {percent(a['case_mean_hotspot_relative_l2'])}。只是开发验证结果，最终测试继续保留。"]
    elif uz and ua:
        z=uz['best_metrics']['validation'];a=ua['best_metrics']['validation']
        brief += ['', f"最新对照：预测位移条件将应力验证均值{percent(z['case_mean_relative_l2'])} → {percent(a['case_mean_relative_l2'])}，最差也改善；热点{percent(z['case_mean_hotspot_relative_l2'])} → {percent(a['case_mean_hotspot_relative_l2'])}，反而变差。下一步检验局部应变条件，最终测试继续保留。"]
    else:
        brief += ['', '正在检验：冻结模型预测的位移，能否帮助应力预测；与相同输入布局的零条件做成对比较。']
    brief += ['', f"完成{len(summaries)}/{len(jobs)}项探索；原30项slides复现已完成。当前研究继续，五个最终测试CAD未参与新模型训练/选择。",'',
              '数据、控制变量及解释范围见 [完整探索报告](REPORT.md)。','']
    (dest/'BRIEF.md').write_text('\n'.join(brief),encoding='utf8')
    write_json(dest/'manifest.json',{'scope':'GeoBC exploration; independent of the completed slides suite',
                                   'completed':len(summaries),'planned':len(jobs),
                                   'phase':'ongoing' if len(summaries)<len(jobs) else 'stage_completed',
                                   'research_status':'ongoing','current_stage':plan['stage'],
                                   'summaries':list(summaries),'raw_data_checkpoints_predictions':'retained locally; not included in Git payload',
                                   'updated_at_utc':utc_now()})
    if geo and load and comparisons:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        values=[100*parent['case_mean_relative_l2'],100*geo['metrics']['case_mean_relative_l2'],100*load['metrics']['case_mean_relative_l2']]
        axes[0].bar(['Train (50)','Unseen CAD (2)','New loads (9)'],values,color=['#507eaa','#d78c59','#d78c59'])
        for i,v in enumerate(values):axes[0].text(i,v+2,f'{v:.1f}%',ha='center')
        axes[0].set_ylim(0,max(values)*1.2);axes[0].set_title('Frozen shared field: migration pilot')
        axes[0].set_ylabel('Case-mean stress relative L2 (%)')
        x=np.arange(len(comparisons))
        axes[1].bar(x-.18,[100*b for _,b,_ in comparisons],.36,label='Shared50')
        axes[1].bar(x+.18,[100*c for _,_,c in comparisons],.36,label='Independent')
        axes[1].set_yscale('log');axes[1].set_xticks(x,[a for a,_,_ in comparisons])
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
        for i,(_,shared,independent) in enumerate(comparisons):
            axes[1].text(i-.18,100*shared*1.05,f'{100*shared:.1f}%',ha='center',fontsize=9)
            axes[1].text(i+.18,100*independent*1.12,f'{100*independent:.3f}%',ha='center',fontsize=9)
        axes[1].set_ylim(.035,65)
        axes[1].set_title('Identical field architecture; separate fitting')
        axes[1].set_ylabel('All-node stress relative L2 (%)')
        fig.savefig(dest/'diagnosis.png',dpi=180)
        plt.close(fig)
    print(json.dumps({'completed':len(summaries),'planned':len(jobs),'report':str(dest/'REPORT.md')}))

if __name__=='__main__':main()
