"""A compact, evidence-driven report for the autonomous GeoBC experiments."""
from pathlib import Path
from datetime import datetime,timezone
import json
import math
import shutil

ROOT=Path(__file__).resolve().parents[1]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def write_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)

def percent(value):
    if not math.isfinite(float(value)): raise ValueError('Nonfinite metric')
    return f'{100*value:.3f}%'

def stage_counts(jobs,summaries):
    synthetic={job['name'] for job in jobs if job.get('kind','').startswith('synthetic_')}
    scientific={job['name'] for job in jobs}-synthetic
    completed=set(summaries)
    return {'scientific_completed':len(scientific&completed),'scientific_planned':len(scientific),
            'synthetic_acceptance_completed':len(synthetic&completed),'synthetic_acceptance_planned':len(synthetic)}

def job_state(job,summaries):
    if job['name'] in summaries:return '已完成'
    path=ROOT/job['output']/'status.json'
    status=json.loads(path.read_text()) if path.is_file() else {}
    phase=status.get('status',status.get('phase',''))
    if phase=='failed':return '失败，等待主协调器处理'
    if status:return '进行中' if phase!='completed' else '等待完成摘要'
    return '等待／排队'

def continuous_geometry_text(result):
    gate=result['geometry_gate']; counts=result['counts']; scan=gate['conformity']
    checks={item['name']:item for item in gate['checks']}
    replay_checks=[checks.get(name) for name in ('direct_FP64_nodal_descriptor_replay','inverse_nodal_descriptor_replay')]
    replay=('通过' if all(item and item['passed'] for item in replay_checks) else
            '未通过' if any(item and not item['passed'] for item in replay_checks) else '未执行')
    exact=result['receipt']['array_identity']['source_float32_reproduction_exact']
    statuses='、'.join(f'{name}={count}' for name,count in result['LOCATOR_status_counts'].items()) or '面一致性停止后未执行数值查询'
    verdict='通过' if result['gate_pass'] else '未通过'
    report=(f"固定TRAIN101_479的CPU几何接口gate{verdict}，耗时{result['elapsed_seconds']:.2f}s；源hash已验证，float32节点缓存{'精确复现' if exact else '未精确复现'}，全十节点描述符的直接／逆定位重放{replay}。"
            f"全CAD的{scan['cell_count']:,}个单元、{scan['interior_face_count']:,}对共享面完成ID一致性扫描，非流形／中点ID不一致{scan['nonmanifold_face_count']}/{scan['midside_mismatch_count']}。"
            f"数值范围为{counts['selected_original_cells']}个单元、{counts['selected_shared_faces']}对面，{counts['logical_query_occurrences']}次逻辑查询（{counts['unique_physical_points_exact_float64']}个不同物理点）；定位状态：{statuses}。"
            '这是输入与固定LN/tanh描述符的有符号二次插值验收；没有GT、模型或GPU，不证明模型精度、泛化、全域Jacobian／单射性或native FP16连续性。正式结果见notes/geobc_continuous_geometry_result.json。')
    brief=(f"单TRAIN几何的CPU连续查询接口gate{verdict}（{result['elapsed_seconds']:.1f}s、{counts['logical_query_occurrences']}次查询），源／节点重放{replay}；这只验收有界输入插值和定位，尚无模型精度、全域可逆性或泛化结论。")
    return report,brief

def quadratic_kinematics_text(result):
    kin=result['kinematics']; metrics=kin['metrics']; verdict='通过' if result['gate_pass'] else '未通过'
    scope=f"同一{len(kin['selected_original_rows'])}单元／{kin['Gauss_points_checked']}个Gauss点"
    limits='这只描述既定局部子集；P2是位移派生代理，曲单元四点积分不是精确连续积分，不代表全CAD、solver应变GT或泛化，也不替代历史全节点恢复P1。'
    if metrics is None:
        return (f"固定TRAIN101_479/ver专用模型10000步保存场的CPU二次运动学诊断已完成，数值gate{verdict}，耗时{result['elapsed_seconds']:.2f}s；{scope}的Gauss几何检查未通过，未计算部分目标指标。"+limits,
                f"冻结TRAIN保存场的局部二次运动学诊断完成，数值gate{verdict}，Gauss检查停止后没有部分精度结果；尚无全CAD或泛化结论。")
    def relative(item):
        return percent(item['value']) if item['value'] is not None else f"未定义（{item['reason']}）"
    p1=relative(metrics['P1']['relative_L2_to_truth']); p2=relative(metrics['P2']['relative_L2_to_truth'])
    mismatch=relative(metrics['truth_P2_minus_P1']['relative_L2_to_same_P2'])
    energy=metrics['error_decomposition']
    mid_fraction=f"{100*energy['Emid']/energy['E2']:.1f}%" if energy['E2'] else '未定义（E2为零）'
    report=(f"固定TRAIN101_479/ver专用模型10000步保存场，在{scope}上完成源、几何和分解检查，数值gate{verdict}，耗时{result['elapsed_seconds']:.2f}s。"
            f"P1／P2代理相对误差为{p1}／{p2}，各用自身真值Frobenius参考分母；真值P2−P1差异相对P2为{mismatch}。"
            f"误差能量E1／Emid／有符号cross／E2={energy['E1']:.4g}／{energy['Emid']:.4g}／{energy['signed_cross']:+.4g}／{energy['E2']:.4g} mm³，恒等式残差{energy['identity_relative_residual']:.2g}；中边项约为E2的{mid_fraction}，交叉项单列，不能解释为因果错误占比。"
            f"真值／预测中边残余位移RMS为{metrics['truth_midside_residual_displacement_vector_RMS_mm']:.4g}／{metrics['prediction_midside_residual_displacement_vector_RMS_mm']:.4g} mm。"+limits+
            '正式结果见notes/geobc_quadratic_kinematics_result.json。')
    brief=(f"冻结专用模型在{scope}的P1／P2派生代理误差{p1}／{p2}（各自真值分母），数值gate{verdict}；这是局部诊断，不代表全CAD或泛化，不与历史全节点P1直接比较。")
    return report,brief

def fullcad_quadratic_kinematics_text(result):
    kin=result['kinematics']; metrics=kin['metrics']; geometry=kin['geometry']
    verdict='通过' if result['gate_pass'] else '未通过'
    scope=f"同一TRAIN101_479/ver保存的专用模型10000步场，覆盖{geometry['centers_checked']:,}单元／{geometry['Gauss_points_checked']:,}个Gauss点"
    limits='范围仍是单TRAIN/ver与专用模型；四点P2是位移派生代理，不是solver应变GT或曲单元精确积分，不证明全域Jacobian、泛化或因果，也不替代历史全节点恢复P1。'
    if metrics is None:
        failed_stage='几何' if not geometry['passed'] else '数值或子集parity'
        return (f"GPU FP64全CAD诊断已完成：{scope}，耗时{result['elapsed_seconds']:.2f}s，数值gate{verdict}；{failed_stage}检查未通过，全CAD目标指标未发布。"+limits,
                f"同TRAIN全CAD二次运动学诊断完成，数值gate{verdict}，全量指标未发布；仍不能给出全CAD精度或泛化结论。")
    def relative(item):
        return percent(item['value']) if item['value'] is not None else f"未定义（{item['reason']}）"
    p1=relative(metrics['P1']['relative_L2_to_truth']); p2=relative(metrics['P2']['relative_L2_to_truth'])
    mismatch=relative(metrics['truth_P2_minus_P1']['relative_L2_to_same_P2'])
    energy=metrics['error_decomposition']; subset=kin['subset32']
    report=(f"GPU FP64确认：{scope}，中心／四角／Gauss几何检查及原{subset['cells']}单元GPU/CPU parity{'通过' if subset['parity']['passed'] else '未通过'}，耗时{result['elapsed_seconds']:.2f}s，数值gate{verdict}。"
            f"全单元P1／P2代理相对误差{p1}／{p2}（各自真值分母），真值P2−P1相对P2差异{mismatch}。"
            f"E1／Emid／有符号cross／E2={energy['E1']:.4g}／{energy['Emid']:.4g}／{energy['signed_cross']:+.4g}／{energy['E2']:.4g} mm³，闭合残差{energy['identity_relative_residual']:.2g}；负交叉项保留，中边项占比不能作因果解释。"
            f"真值／预测中边残余位移RMS为{metrics['truth_midside_residual_displacement_vector_RMS_mm']:.4g}／{metrics['prediction_midside_residual_displacement_vector_RMS_mm']:.4g} mm。"+limits+
            '正式结果见notes/geobc_fullcad_quadratic_kinematics_result.json。')
    brief=(f"同TRAIN/ver专用模型的全CAD四点积分P1／P2代理误差{p1}／{p2}（各自真值分母），GPU FP64及子集parity通过；中边梯度现象延伸至该CAD全单元，仍非solver应变GT或泛化结论。")
    return report,brief


def solver_capability_text(result):
    counts=result['counts']; reports=[]
    for route in result.get('results') or []:
        common=route.get('common_nodal_full10')
        if common is None:
            reports.append(f"{route['method']}/{route['seed']}指标未定义")
            continue
        kin=common['kinematics_proxy']; p2=kin['P2']['relative_L2_to_truth']['value']
        native=route.get('native') or {}; item=native.get('P2_reference_strain_proxy',{}).get('relative_L2',{})
        p2_text=percent(p2) if p2 is not None else '零参考分母未定义'
        native_text=percent(item['value']) if item.get('value') is not None else native.get('status','未执行')
        reports.append(f"{route['method']}/{route['seed']}共同P2／原生={p2_text}／{native_text}")
    checks=result['checks']; verdict='通过' if checks.get('all_routes_numeric_guards_passed') else '未全部通过'
    scope=f"固定TRAIN101_479/ver的32 cells／8 faces／{counts['logical_queries']} logical queries，{counts.get('unique_original_nodes',0)} unique原节点／128 Gauss"
    report=(f"四个已锁预算端点完成连续能力诊断，{scope}，耗时{result['elapsed_seconds']:.3f}s；几何与数值guards{verdict}。"+
            '；'.join(reports)+'。'+
            '共同full10由物理解码的原节点预测构建；FNO原生为同预测grid上的直接物理坐标三线性查询和解析梯度，另栏使用相同GT-P2分母。MLP原25 offnode输入延拓未定义，native未实现，不虚构BC权重。'+
            'P1/P2只来自GT节点运动学proxy，非solver应变GT、全CAD或final结果；C0是结构接口性质，不证明学习精度。绝对RMS、有符号cross、最差cell、节点舍入重放和实际成本见results/solver_capability/metrics.json；不按此结果挑seed或重训。')
    brief=f"固定TRAIN32 cells／504查询连续诊断完成（{result['elapsed_seconds']:.3f}s）：节点精度不代表P2梯度准确，FNO原生查询未消除该局部proxy差距；MLP native未实现。C0通过只表明结构接口，非solver strainGT或全CAD结论。"
    return report,brief


def gino_radius_text(result):
    stats=result['stats']; point=stats['point_queries']; recipe=result['config']['contract']['recipe']
    elapsed=result['costs']['elapsed_seconds']; feasible=result.get('feasible') is True
    coverage=stats.get('point_queries_all_covered') is True
    report=(f"GINO候选的固定TRAIN{recipe['case_key']}输入邻域准备completed，耗时{elapsed:.3f}s，可行性{'通过' if feasible else '未通过'}。"
            f"R{recipe['resolution']}、in/out radius={recipe['in_radius']}/{recipe['out_radius']}，{point['edges']:,}个相同pairs形成两种CSR方向，{point['queries']:,}点{'全部covered' if coverage else '未全部covered'}；"
            f"缓存{result['cache_bytes']:,} B，实测构图peak allocated={result['memory']['peak_allocated_bytes']:,} B。"
            '仅xyz输入、枚举／排序／CSR／缓存成本，零labels/features/weights/models/backward；不是GINO模型实验或模型前反向显存。未测BC角色保持、核函数、连续query截止行为或精度，不能把现有point-to-grid FNO改名GINO。')
    brief=(f"GINO单TRAIN radius输入图{'可行' if feasible else '未通过'}（{elapsed:.3f}s，{point['edges']:,} pairs，{point['queries']:,}点{'全覆盖' if coverage else '覆盖不全'}）；仅构图资源，尚无GINO模型／BC／连续场精度证据。")
    return report,brief


def gino_resource_text(result):
    checks=result['checks']; passed=sum(value is True for value in checks.values())
    feasible=result.get('feasible') is True and passed==len(checks)
    costs=result['costs']; memory=result['memory']['main_encode_query_backward']
    api=result['api']; params=result['parameter_counts']; receipt=result['receipt']
    report=(f"GINO正式有界模型资源检查completed，{'通过' if feasible else '未通过'}（{passed}/{len(checks)} checks），耗时{costs['elapsed_seconds']:.3f}s。"
            f"固定TRAIN101_479/ver的{api['input_points']:,}点全输入encode={costs['full_input_encode_seconds']:.3f}s，512原节点query={costs['query512_seconds']:.4f}s，backward={costs['backward_seconds']:.4f}s；"
            f"{receipt['main_backwards']}次无目标主反向和{receipt['backend_autograd_grad_calls']}次小真实CSR后端autograd核对，labels／optimizer更新={receipt['labels_read']}/{receipt['optimizer_steps']}。"
            f"实际Torch peak allocated/reserved={memory['peak_allocated_bytes']:,}/{memory['peak_reserved_bytes']:,} B；参数{params['total']:,}（含PointNet，complex tensor按numel计）。"
            '这已是模型资源实验，但只覆盖全输入/512输出，首例全N输出训练资源尚未验；无训练或准确度结果。旧FNO约14.18M且频率实现不同，不是容量匹配对照，不能把未来差异归因单GNO组件；不证明BC满足或连续场能力。')
    brief=(f"GINO全输入/512输出模型资源{'通过' if feasible else '未通过'}（{costs['elapsed_seconds']:.3f}s、{passed}/{len(checks)} checks），"
           f"peak allocated {memory['peak_allocated_bytes']/1e9:.3f} GB；零labels/optimizer更新，训练与首例全N输出仍待验，尚无精度证据。")
    return report,brief


def gino_training_text(result):
    exposure=result['case_exposure_counts']; costs=result['costs']
    if result.get('endpoint_step')!=480 or costs.get('train_case_exposures')!=480 or len(exposure)!=120 or any(v!=4 for v in exposure.values()):
        raise ValueError('Completed GINO report requires480 updates and120 cases each4 exposures')
    final=result['final_metrics']; best=result['best_metrics']; v=final['validation']; bv=best['validation']
    primary=f"U均值／最差={percent(v['case_mean_u_relative_l2'])}/{percent(v['worst_case_u_relative_l2'])}，signed S均值／最差={percent(v['case_mean_stress_relative_l2'])}/{percent(v['worst_case_stress_relative_l2'])}"
    report=(f"GINO单seed2026 fixed480完整case初筛completed，四次TRAIN120遍历，{primary}；可比train_seconds={costs['train_seconds']:.2f}s，总wall={result['elapsed_seconds']:.2f}s。"
            f"共同VAL-score best次结果step{best['step']}，同点U/S均值={percent(bv['case_mean_u_relative_l2'])}/{percent(bv['case_mean_stress_relative_l2'])}；主结果仍锁480，step0若为best只标未训练基线。"
            '共享原25/PointNet32、TRAIN-only scaler与旧480 case顺序；CSR首建/save另报，cached IO/H2D计入可比训练时间。'
            '官方modes24/24/22与旧FNO保留频带相当，但复杂参数numel/实自由度、lifting/skip/projection及Adam foreach/旧fused不同，不能单GNO归因或凭更新数称成本胜出。'
            '完整旧六screen同480点和FNO/MLP两seed600秒端点只作真实成本背景，不插值或外推等时精度。见results/gino_screen/BRIEF.md和metrics.json；未读取final/dia/新test，不证明收敛、连续梯度、BC满足或工程精度。')
    brief=(f"GINO480主端点U/S均值={percent(v['case_mean_u_relative_l2'])}/{percent(v['case_mean_stress_relative_l2'])}，实测train={costs['train_seconds']:.1f}s；四遍历完成，较旧screen及600秒确认的成本背景另报，未凭更多耗时选定solver。")
    return report,brief


def sealed_acquisition_text(result):
    contract=result['config']['contract']; total=len(contract['files']); done=result.get('completed_files',0)
    failed=result.get('failed_files',0); status=result['status']
    errors=list(dict.fromkeys(item.get('error','') for item in result.get('files',[]) if item.get('error')))
    reason=errors[0] if errors else '见status.json的当前获取状态'
    seal=result.get('labels_sealed') is True and result.get('Field_CSV_values_parsed') is False and result.get('scientific_array_reads')==0
    report=(f"下一周期独立test5身份已冻结：{'、'.join(contract['test_CADs'])}。获取状态{status}，{done}/{total}文件完成、{failed}失败，耗时{result.get('elapsed_seconds',0):.1f}s。"
            + (f"首个错误：{reason}。" if status!='completed' else '仅下载／收到字节校验完成。')+
            f"字段{'仍封存，未解析labels' if seal else '封存receipt需核对'}，没有替换CAD或修改旧raw来源。"
            '身份独立不等于已证明形状族隔离；接收字节SHA没有独立上游checksum。新路线、预算、两seed/checkpoint与指标规则须在Field值解析前锁定，旧final不作开发集。下载失败不是科学模型负结果，也不计为completed阶段。')
    brief=(f"独立test5身份已冻且字段{'封存' if seal else '状态待核对'}；获取{status}（{done}/{total}文件），"+
           ('下载路径待修复，不换CAD、不解析labels。' if status!='completed' else '原始字节已获取，尚未评估或用于模型开发。'))
    return report,brief

def plan_next_text(plan,summaries,*,compact=False):
    current_jobs=plan.get('jobs',[])
    complete=bool(current_jobs) and all(job['name'] in summaries for job in current_jobs)
    text=('当前正在设计下一阶段，尚未放行数据或训练。' if not current_jobs else
          '当前计划阶段已完成，后续决策仍由主计划确定。' if complete else
          '当前主计划：'+ '；'.join(f"{job['name']}（{job_state(job,summaries)}）" for job in current_jobs)+'。')
    stages=plan.get('next_stages',[])
    if isinstance(stages,dict):stages=[stages]
    if compact:
        if stages and isinstance(stages[0],dict) and stages[0].get('brief_action'):
            return stages[0]['brief_action']
        name=stages[0].get('stage',plan.get('stage','未指定')) if stages and isinstance(stages[0],dict) else plan.get('stage','未指定')
        completed=sum(job['name'] in summaries for job in current_jobs)
        return f"主计划阶段{plan.get('stage','未指定')}已完成{completed}/{len(current_jobs)}项；"+('当前阶段已完成，' if complete else '其余等待主协调器完成，')+f"后续计划阶段为{name}，具体动作以notes/geobc_exploration_plan.json的证据规则为准。"
    if stages:
        stage=stages[0]
        if isinstance(stage,str):return text+' 下一阶段：'+stage+'。'
        text+=f" 计划阶段：{stage.get('stage',plan.get('stage','未指定'))}。"
        if stage.get('design'):text+=f" 设计：{stage['design']}。"
        if stage.get('if_pass'):text+=f" 若通过：{stage['if_pass']}。"
        if stage.get('if_fail'):text+=f" 若未通过：{stage['if_fail']}。"
        if stage.get('after'):text+=f" 完成后：{stage['after']}。"
    return text

def main(*, preserve_existing_plots=False):
    plan=json.loads((ROOT/'notes/geobc_exploration_plan.json').read_text())
    dest=ROOT/'results/geobc_exploration'
    dest.mkdir(parents=True,exist_ok=True)
    summaries={}
    checkpoints={}
    observed_stages={}
    observed_receipts={}
    metadata_kinds={'solver_continuous_capability','gino_radius_input_pilot','gino_native_resource','gino_training_screen','sealed_test_input_acquisition','fno_response_screen','fno_response_eval','deeponet_bc_cloud_input','deeponet_screen','deeponet_development_dia_eval','multiscale_mesh_input','multiscale_mesh_screen','multiscale_mesh_development_dia_eval'}
    jobs=plan.get('historical_jobs',[])+plan['jobs']
    for job in jobs:
        source=ROOT/job['output']
        path=source/('manifest.json' if job.get('kind') in ('solver_final_input_cache','deeponet_bc_cloud_input','multiscale_mesh_input') else 'summary.json')
        if not path.exists(): continue
        summary=json.loads(path.read_text())
        completed=summary.get('status')=='completed'
        if not completed and job.get('kind')!='sealed_test_input_acquisition': continue
        if completed:summaries[job['name']]=summary
        if job.get('kind') in metadata_kinds:observed_stages[job['name']]=summary
        checkpoint_files={}
        for filename in ('best.pt','last.pt','endpoint_250.pt','endpoint_10000.pt','endpoint_0000.pt','endpoint_1000.pt'):
            checkpoint=source/filename
            if checkpoint.is_file():
                stat=checkpoint.stat()
                checkpoint_files[filename]={'local_path':str(checkpoint),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if checkpoint_files:
            checkpoints[job['name']]={'best_step':summary.get('best_step'),'files':checkpoint_files,
                                      'code_sha256':summary.get('config',{}).get('contract',{}).get('code_sha256',summary.get('config',{}).get('contract',{}).get('source_sha256',{})),
                                      'selection':summary.get('config',{}).get('checkpoint_selection')}
        target=dest/'runs'/job['name']; target.mkdir(parents=True,exist_ok=True)
        filenames=('summary.json','config.json','metrics_best.json','metrics_latest.json','metrics.jsonl','parent_comparison.json','metrics_endpoint_250.json','metrics_endpoint_10000.json','metrics_step0000.json','metrics_step1000.json','first_backward_check.json')
        if job.get('kind') in ('real_geometry_interface_acceptance','frozen_field_diagnostic','solver_screen','solver_confirmation','solver_final_input_cache','solver_final_evaluation'):filenames+=('status.json',)
        if job.get('kind') in metadata_kinds:filenames=('summary.json','config.json','status.json')
        if job.get('kind')=='gino_training_screen':filenames+=('metrics_best.json','metrics_latest.json','metrics.jsonl','first_update_gradient_check.json')
        if job.get('kind')=='fno_response_screen':filenames+=('metrics_best.json','metrics_latest.json','metrics.jsonl','first_update_gradient_check.json','first_actual_check.json')
        if job.get('kind')=='deeponet_screen':filenames+=('metrics_best.json','metrics_latest.json','metrics.jsonl','first_update_gradient_check.json','first_actual_check.json')
        if job.get('kind')=='multiscale_mesh_screen':filenames+=('metrics_best.json','metrics_latest.json','metrics.jsonl','first_update_gradient_check.json','first_actual_check.json')
        if job.get('kind')=='deeponet_bc_cloud_input':filenames+=('manifest.json',)
        if job.get('kind')=='multiscale_mesh_input':filenames+=('manifest.json',)
        if job.get('kind')=='solver_final_input_cache':filenames+=('manifest.json',)
        if job.get('kind') in ('solver_screen','solver_confirmation'):filenames+=('first_update_gradient_check.json',)
        for filename in filenames:
            if (source/filename).exists(): shutil.copyfile(source/filename,target/filename)
        if job.get('kind') in metadata_kinds:
            observed_receipts[job['kind']]={'run':job['name'],'status':summary.get('status'),
                'source_output':job['output'],'completed_stage':completed,
                'exported_files':[str(Path('runs')/job['name']/name).replace('\\','/') for name in filenames if (source/name).exists()],
                'export_policy':'Small JSON only; no NPY/CSR/raw/weights copied',
                'cache_payloads_exported':False,'receipt_location':'summary.json:receipt or config.contract'}
    counts=stage_counts(jobs,summaries)
    next_text=plan_next_text(plan,summaries)
    brief_next_text=plan_next_text(plan,summaries,compact=True)
    current_stage_complete=bool(plan['jobs']) and all(job['name'] in summaries for job in plan['jobs'])
    write_json(dest/'checkpoint_index.json',{'weights_published':False,'weights_location':'local training directories',
                                            'identity_policy':'paths/bytes/mtime plus small config source hashes; no redundant checkpoint scans',
                                            'runs':checkpoints})
    manifests=dest/'data_manifests'; manifests.mkdir(exist_ok=True)
    for directory in ('prepared_geobc_transfer','prepared_geobc_multiload','geobc_local_geometry','geobc_frozen_u_response','geobc_frozen_strain_response','geobc_gradient_supervision','geobc_graph_context','geobc_solver_selection','geobc_solver_final_eval'):
        source=ROOT/'data'/directory/'manifest.json'
        if source.is_file():shutil.copyfile(source,manifests/(directory+'.json'))
    lines=['# GeoBC 探索：当前结论与下一步','',
           '研究对象：原版 DeepJEB；使用逐节点坐标对齐的原始有符号应力。原30项slides复现已完成，本页的后续自主研究仍在继续。','',
           '## 当前结论','',
           '**Hash 的单案例表示能力很强；当前共享模型的跨几何、跨载荷组合迁移很差。**','']
    if plan.get('stage','').startswith('solver_'):
        lines += ['**选型状态：尚未确定求解器，优先六家族跨CAD广筛，暂缓Hash局部优化。**',
                  '当前共同协议与简洁结果见 [选型计划](../../notes/geobc_solver_selection_plan.json)；下文保留此前Hash探索证据，不代表跨求解器排名。','']
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
            lines += ['', '## 整体边界角色条件对照','',
                      f"元数据核验确认原25通道已包含L_mm/200；45CAD尺寸仅{lengths['min_mm']:.2f}–{lengths['max_mm']:.2f}mm，暂缓显式尺寸缩放。这一对照之前PointNet只汇聚坐标；新对照在相同1024点云附加固定节点并集和加载节点两个输入角色。零角色与真实角色都采用5输入PointNet，新增64参数在两侧相同，不能复用旧3输入模型作控制。",
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
    specialist=summaries.get('specialist_101_479_joint_u_P1_10k')
    specialist_design_path=ROOT/'notes/geobc_specialist_plan.json'
    if specialist_design_path.is_file():
        specialist_design=json.loads(specialist_design_path.read_text())
        if specialist_design.get('status')=='selected_after_failed_BC_gate':
            lines += ['', '## 固定训练CAD独立拟合','',
                      '固定已观察的TRAIN101_479/ver、hor、tor，采用原xyz3模型与同一校准损失，共享模型的scaler和案例权重不重拟合。报告固定250与10000步，不用验证或应变挑选最优checkpoint。主要比较共享模型固定10000步的该CAD：位移16.85%、P1应变293.24%。',
                      '单CAD每步四个载荷槽有放回抽样，这是显式诊断变化。250步仅匹配共享10000步的期望案例曝光，完整10000步约有40倍曝光；独立模型也增加总参数。即使独立拟合成功，也不能单独证明共享容量不足或获得泛化能力。设计见notes/geobc_specialist_plan.json。']
    if specialist:
        reference=specialist['reference_final_10000']
        early=specialist['endpoint_metrics']['250']['train']
        final=specialist['endpoint_metrics']['10000']['train']
        lines += ['', '| TRAIN101_479三个载荷 | 位移均值 | P1应变均值 |', '|---|---:|---:|',
                  f"| 共享模型固定10000步 | {percent(reference['case_mean_relative_l2'])} | {percent(reference['strain']['case_mean_relative_l2'])} |",
                  f"| 独立模型固定250步 | {percent(early['case_mean_relative_l2'])} | {percent(early['strain']['case_mean_relative_l2'])} |",
                  f"| 独立模型固定10000步 | {percent(final['case_mean_relative_l2'])} | {percent(final['strain']['case_mean_relative_l2'])} |",'',
                  f"独立训练耗时{specialist['elapsed_seconds']:.1f}s，最终最差位移{percent(final['worst_case_relative_l2'])}、最差P1{percent(final['strain']['worst_case_relative_l2'])}。同一场与损失在该CAD能联合拟合；250步案例曝光接近共享10000步仍有改善，但优化轨迹与专用容量不同，不能单独证明共享容量或编码是原因。",
                  '后续回到40/5共享训练，匹配新增查询上下文32的零／真实分支：对同一PointNet几何／BC点token按固定Gaussian距离加权，σ=.15归一化长度。原全局32保留，新增投影和native输入容量在两侧相同；这是整个分支消融，不能称纯信息因果控制。设计见notes/geobc_query_context_plan.json。']
    qz=summaries.get('displacement_query_context_zero_10k')
    qa=summaries.get('displacement_query_context_actual_10k')
    query_result_path=ROOT/'notes/geobc_query_context_pair_result.json'
    query_result=json.loads(query_result_path.read_text()) if query_result_path.is_file() else None
    if qz or qa:
        lines += ['', '## 查询位置相关几何与BC上下文','',
                  '同一五通道PointNet的64维点token经共同64→32投影，再按查询点距离以固定σ=.15的Gaussian权重汇聚；保留整体32维描述和原38维查询输入。两侧拥有相同新投影、native条件102、初始化和训练配方，零组在计算后将查询上下文置零。旧BC模型不能充当该分支的匹配控制。',
                  '', '| 查询上下文 | 验证位移均值 | 最差位移 | 验证P1代理均值 | 训练耗时 |', '|---|---:|---:|---:|---:|']
        for mode,summary in [('零槽',qz),('距离汇聚的实际点token',qa)]:
            if summary:
                v=summary['best_metrics']['validation']
                lines += [f"| {mode} | {percent(v['case_mean_relative_l2'])} | {percent(v['worst_case_relative_l2'])} | {percent(v['strain']['case_mean_relative_l2'])} | {summary['elapsed_seconds']:.1f}s |"]
            else:
                lines += [f'| {mode} | 进行中 | — | — | — |']
        if qz and qa and query_result:
            lines += [f"配对控制已核对。实际上下文的位移均值改善{query_result['validation_u_mean_improvement_percentage_points']:.2f}个百分点、最差位移变化{query_result['validation_u_worst_deterioration_percentage_points']:+.2f}个百分点（正数为退步），P1误差相对改善{100*query_result['validation_P1_relative_error_reduction']:.1f}%。预定综合门槛{'通过' if query_result['clear_gain_gate_passed'] else '未通过'}。",
                      '每模型的位移与P1指标来自同一个按开发验证位移选择的checkpoint。分支激活改变有效容量，结果只检验整个分支，不是缺失局部信息的纯因果证明；空间汇聚也不等于弹性载荷传播。最终测试未用于本轮。']
        if (dest/'query_context_comparison.png').is_file():
            lines += ['', '![整体角色与查询上下文的两个匹配对照](query_context_comparison.png)',
                      '同形状的两个点才是匹配对照；不能把两个组之间的差异归于一种输入。左下方更好，100%虚线为预测全零应变的参考。']
    learning=summaries.get('query_learning_frozen_10k')
    if learning:
        readout=learning['readout'];probe=readout['actual_same_weights_final_query_zero']
        intact=probe['mean_intact_losses'];masked=probe['mean_masked_losses']
        lines += ['', '## 固定模型的查询学习与梯度诊断','',
                  f"固定两个最终10000步checkpoint，在4个已见训练CAD的三种载荷上使用同一12800行查询witness；{learning['case_backwards']}次backward、{learning['optimizer_updates']}次参数更新，计算耗时{learning['elapsed_seconds']:.1f}s。不使用验证、dia或最终测试字段。",
                  f"实际模型只将最终query32置零，保留权重、整体描述与其他输入：12/12案例的位移损失及P1损失都升高，平均分别为完整输入的{masked['relative_Lu']/intact['relative_Lu']:.2f}倍和{masked['normalized_volume_P1_Frobenius_loss']/intact['normalized_volume_P1_Frobenius_loss']:.2f}倍。该分支在这些训练查询上有实际作用；遮蔽可能离开训练分布，不能据此证明信息充分或泛化。",
                  f"实际模型的6个CAD梯度对中有{readout['actual']['full_model_negative_CAD_pairs']}对反向，零分支模型有{readout['zero']['full_model_negative_CAD_pairs']}对；双方均没有CAD与四CAD平均梯度点积为负。部分方向竞争存在，但这个探针未显示共同平均方向会立即增大某个CAD损失。它也不是Adam实际更新或泛化失败的因果证明。",
                  '此处P1损失是原训练配方的体积加权、归一化cell平方损失，不能与前文全节点P1相对L2百分比直接比较。这一冻结诊断提供了后续候选的设计依据，没有执行梯度手术或参数扫描。']
    native=summaries.get('hash_modulation_native_smoke')
    native_passed=False
    if native:
        checks=native['native_and_gate_checks']
        native_passed=(native['paired_initial_state_exact_equal'] and native['paired_step0_prediction_exact_equal'] and
                       native['no_GT'] and native['dataset_or_checkpoint_files_opened']==0 and native['optimizer_updates']==0 and
                       all(checks[mode][group]['present_and_finite'] and checks[mode][group]['gradient_L1']>0
                           for mode in ('zero','actual') for group in ('native_encoding_gradient','native_network_gradient')) and
                       all(checks[mode]['gate_head_gradient']['present_and_finite'] for mode in ('zero','actual')) and
                       checks['zero']['gate_head_gradient']['gradient_L1']==0 and checks['actual']['gate_head_gradient']['gradient_L1']>0)
        evidence=('两侧初始参数与step0输出严格相同；独立Hash编码和MLP的两个native参数buffer有有限非零梯度，调制头的零组梯度精确为零、实际组非零。' if native_passed else '该记录没有通过完整的配对初始化、native梯度与调制头梯度检查。')
        lines += ['', '## Hash调制的native合成验收','',
                  f"合成接口验收{'通过' if native_passed else '未通过'}，耗时{native['elapsed_seconds']:.2f}s，{native['backwards']}次backward、{native['optimizer_updates']}次参数更新。{evidence}",
                  '使用共享合成输入，没有读取数据集或checkpoint，也没有GT标签。这只证明新split-native接口和梯度可用；没有物理精度、泛化或有效学习结论。它单独计作合成验收，不计入科学数据实验。']
    hash_result_path=ROOT/'notes/geobc_hash_modulation_pair_result.json'
    hash_result=json.loads(hash_result_path.read_text()) if hash_result_path.is_file() else None
    if hash_result and hash_result.get('status')!='completed':hash_result=None
    hash_jobs=[job for job in plan['jobs'] if job['name'].startswith('displacement_hash_modulation_')]
    if hash_result or hash_jobs:
        lines += ['', '## 几何与BC条件直接调制共享Hash特征','',
                  '两侧均保留实际query32和整体BC32，用共同64→24调制头生成1+tanh倍率；W/b零初始化。原生编码与MLP拆成两个模块，两侧从同一新布局初始化。对照只改变是否启用Hash倍率，旧查询模型不能作为匹配基线；分支激活仍改变有效容量，不能当作纯信息因果控制。']
        if hash_result:
            hz=hash_result['runs']['zero'];ha=hash_result['runs']['actual']
            lines += [f"按已完成配对分析，开发验证位移均值{percent(hz['validation_u_mean_relative_l2'])} → {percent(ha['validation_u_mean_relative_l2'])}，最差位移{percent(hz['validation_u_worst_relative_l2'])} → {percent(ha['validation_u_worst_relative_l2'])}，P1代理均值{percent(hz['validation_P1_mean_relative_l2'])} → {percent(ha['validation_P1_mean_relative_l2'])}。两侧均按位移验证均值选模，所报P1来自同一checkpoint；选择步分别{hz['best_step']}／{ha['best_step']}。",
                      f"匹配检查{'通过' if hash_result['matched_controls_passed'] else '未通过'}；预定综合门槛{'通过' if hash_result['clear_gain_gate_passed'] else '未通过'}。位移均值改善{hash_result['validation_u_mean_improvement_percentage_points']:.2f}个百分点，最差变化{hash_result['validation_u_worst_deterioration_percentage_points']:+.2f}个百分点（正数为退步），P1误差相对改善{100*hash_result['validation_P1_relative_error_reduction']:.1f}%。",
                      f"训练位移均值{percent(hz['train_u_mean_relative_l2'])} → {percent(ha['train_u_mean_relative_l2'])}，训练P1均值{percent(hz['train_P1_mean_relative_l2'])} → {percent(ha['train_P1_mean_relative_l2'])}；计算耗时{hz['elapsed_seconds']:.1f}s／{ha['elapsed_seconds']:.1f}s。结果来源notes/geobc_hash_modulation_pair_result.json；最终测试未用于本轮。"]
        else:
            lines += ['主计划阶段状态：'+'；'.join(f"{job['name']}（{job_state(job,summaries)}）" for job in hash_jobs)+'。配对分析尚未完成，此处不引用单侧物理指标。']
        if hash_result and (dest/'hash_modulation_comparison.png').is_file():
            lines += ['', '![新split匹配对照的位移和P1验证结果](hash_modulation_comparison.png)',
                      '只比较本组新split布局的零／实际调制模型；位移与P1均来自同一个u-selected checkpoint。两侧纵轴从零起，P1的100%虚线表示预测全零应变；没有混入历史Composite模型。']
    hash_use_path=ROOT/'notes/geobc_hash_modulation_use_result.json'
    hash_use=json.loads(hash_use_path.read_text()) if hash_use_path.is_file() else None
    if hash_use and (hash_use.get('status')!='completed' or not hash_use.get('formal_checks_passed')):hash_use=None
    if hash_use:
        use=hash_use['actual']
        lines += ['', '## 冻结模型中的Hash调制使用','',
                  f"固定两个最终{hash_use['fixed_step']}步模型，在4个已见TRAIN CAD×3载荷的相同混合查询witness上执行{hash_use['basis_forward_calls']}次basis forward、{hash_use['backwards']}次backward、{hash_use['optimizer_updates']}次更新，耗时{hash_use['elapsed_seconds']:.2f}s；没有读取验证、dia或最终测试字段。这不是前述u-selected checkpoint的全节点评估。",
                  f"对实际模型保持权重和其他输入，仅把γ设为1关闭Hash调制：{use['cases_Lu_masking_harms']}/{use['sampled_cases']}个采样案例的位移损失上升，平均Lu为原来的{use['Lu_masked_to_natural_ratio']:.2f}倍；{use['cases_P1_cell_loss_masking_improves']}/{use['sampled_cases']}个案例的P1 cell平方损失下降，平均为原来的{use['P1_cell_squared_loss_masked_to_natural_ratio']:.4f}倍。这里的P1是体积加权、归一化cell平方损失，不是全节点P1相对L2百分比，也不是solver GT应变。",
                  f"相同训练查询上的γ≤0.05或≥1.95占{100*use['near_endpoint_fraction_mean4']:.2f}%，精确为0或2的比例{100*use['exact_endpoint_fraction_mean4']:.2f}%；四CAD平均的空间中心化RMS为{use['gamma_spatial_centered_RMS_mean4']:.5f}。接近端点不等于硬数值饱和，混合查询也不代表全节点分布。调制已被拟合位移场使用，但没有改善本轮共享开发验证；关闭是可能离分布的同权重干预，不能据此确定泛化失败原因。结果来源notes/geobc_hash_modulation_use_result.json。"]
    graph_ready_path=ROOT/'notes/graph_context_data_ready.json'
    graph_ready=json.loads(graph_ready_path.read_text()) if graph_ready_path.is_file() else None
    graph_native=summaries.get('graph_context_native_smoke')
    graph_result_path=ROOT/'notes/geobc_graph_context_pair_result.json'
    graph_result=json.loads(graph_result_path.read_text()) if graph_result_path.is_file() else None
    if graph_result and not (graph_result.get('status')=='completed' and graph_result.get('matched_controls_passed')):graph_result=None
    if graph_ready and graph_ready.get('complete'):
        lines += ['', '## 同一边界界面的距离表示对照','',
                  '用相同的四个支撑member组和实际loaded member组，比较到界面的欧氏距离与沿原四面体网格边的最短路径距离。两组五通道均启用，使用同一长度归一化、完整初始化、fused容量、query32/global32、训练损失和采样；历史小容量模型不作匹配对照。不同距离分布也改变数值条件，因此该对照检验表示效用，不能单独证明拓扑的因果作用。',
                  '45个TRAIN／开发验证几何的双距离缓存已准备完成，没有读取位移或应力标签、拟合scaler或打开最终测试。固定网格路径依赖原离散网格，不是弹性刚度或精确连续测地距离。缓存准备不计作训练实验。']
        if graph_native:
            lines += [f"真实native合成验收完成，耗时{graph_native['elapsed_seconds']:.2f}s，{graph_native['basis_forward_calls']}次basis forward、{graph_native['backwards']}次backward、零参数更新。两组完整初始参数和同距离fixture输出相同；自然输入不同，不要求初始输出相等。实测每组公共native参数7770496、完整模型7789376；没有推断flat buffer分区。这只验证接口和梯度，物理收益依据后续配对数据实验。"]
        if graph_result:
            e=graph_result['runs']['euclidean'];g=graph_result['runs']['graph']
            lines += [f"在相同15个known-load开发验证案例上，欧氏／网格路径的位移均值{percent(e['validation_u_mean_relative_l2'])} → {percent(g['validation_u_mean_relative_l2'])}、最差位移{percent(e['validation_u_worst_relative_l2'])} → {percent(g['validation_u_worst_relative_l2'])}、P1代理均值{percent(e['validation_P1_mean_relative_l2'])} → {percent(g['validation_P1_mean_relative_l2'])}；选择步{e['best_step']}／{g['best_step']}，P1来自同一位移选模checkpoint。",
                      f"匹配控制通过，预先声明的综合门槛{'通过' if graph_result['clear_gain_gate_passed'] else '未通过'}。位移均值改善{graph_result['validation_u_mean_improvement_percentage_points']:.2f}个百分点，最差退步{graph_result['validation_u_worst_deterioration_percentage_points']:+.2f}个百分点，P1误差相对改善{100*graph_result['validation_P1_relative_error_reduction']:.1f}%；两组计算共{(e['elapsed_seconds']+g['elapsed_seconds'])/60:.1f}分钟。最终测试和dia未用于本轮。正式结果见notes/geobc_graph_context_pair_result.json。"]
            if (dest/'graph_context_comparison.png').is_file():
                lines += ['', '![相同边界界面的两种距离表示](graph_context_comparison.png)']
        else:
            lines += ['两侧数据实验和正式配对分析尚未全部完成，此处不使用单侧结果作为结论。']
    error_gradient_path=ROOT/'notes/geobc_error_gradient_result.json'
    error_gradient=json.loads(error_gradient_path.read_text(encoding='utf8')) if error_gradient_path.is_file() else None
    if error_gradient and error_gradient.get('status')!='completed':error_gradient=None
    if error_gradient:
        if (error_gradient['case_key']!='10_58/tor' or error_gradient['group']!='train'
                or error_gradient['selected_step_both']!=10000):
            raise ValueError('Error-gradient report requires the frozen TRAIN10_58/tor pair at10000 steps')
        de=error_gradient['models']['euclidean'];dg=error_gradient['models']['graph']
        ep=de['all_node_P1'];gp=dg['all_node_P1']
        energy_share=lambda record,key:100*record[key]/record['raw_error_energy']
        lines += ['', '## 固定训练代表的误差与梯度分解','',
                  '只分析两组当前距离模型已导出的TRAIN10_58/tor、10000步checkpoint。用真实误差拟合并扣除整体仿射项，是依赖GT的描述诊断，不能作为可部署修正，也不代表全体验证结果。',
                  f"欧氏／网格路径模型的位移参考RMS为{de['u']['reference_RMS']:.6g}／{dg['u']['reference_RMS']:.6g}mm；原始／仿射／残余误差RMS分别为{de['u']['raw_error_RMS']:.6g}／{de['u']['affine_error_RMS']:.6g}／{de['u']['residual_error_RMS']:.6g}mm与{dg['u']['raw_error_RMS']:.6g}／{dg['u']['affine_error_RMS']:.6g}／{dg['u']['residual_error_RMS']:.6g}mm。全节点P1参考Frobenius RMS为{ep['reference_RMS']:.6g}／{gp['reference_RMS']:.6g}，原始误差RMS为{ep['raw_error_RMS']:.6g}／{gp['raw_error_RMS']:.6g}（均无量纲）。",
                  f"去除仿射误差后，保留的全节点P1误差能量为原来的{100*ep['residual_over_raw_error_energy']:.2f}%／{100*gp['residual_over_raw_error_energy']:.2f}%；仿射能量／原始能量为{energy_share(ep,'affine_error_energy'):.2f}%／{energy_share(gp,'affine_error_energy'):.2f}%，有符号两倍交叉项／原始能量为{energy_share(ep,'signed_two_affine_residual_cross'):+.2f}%／{energy_share(gp,'signed_two_affine_residual_cross'):+.2f}%。这些项包含交叉作用，不能当作互相独立的因果贡献。",
                  f"预先声明的非仿射保留标记{'达到' if error_gradient['readout']['non_affine_dominance_flag'] else '未达到'}，小单元集中标记{'达到' if error_gradient['readout']['small_cell_concentration_flag'] else '未达到'}；后者只描述最小输入h四分位中的体积加权cell误差集中，与全节点能量不能混加。它不证明Hash频率振荡、全体验证瓶颈或新的精度收益。P1仍是角节点恢复代理，不是solver应变GT。正式结果见notes/geobc_error_gradient_result.json。"]
    bandwidth_result_path=ROOT/'notes/geobc_hash_bandwidth_pair_result.json'
    bandwidth_result=json.loads(bandwidth_result_path.read_text()) if bandwidth_result_path.is_file() else None
    if bandwidth_result and not (bandwidth_result.get('status')=='completed' and bandwidth_result.get('matched_controls_passed') and bandwidth_result.get('pair_controls_passed')):
        bandwidth_result=None
    if bandwidth_result:
        bf=bandwidth_result['runs']['full12'];bc=bandwidth_result['runs']['coarse8']
        lines += ['', '## 固定Hash空间层数的匹配对照','',
                  '完整12层与只保留前8层共用Euclidean43、query32、global32、训练正常器、损失、初始化和预算。两侧均计算完整12层Encoding，后者把最后8个特征乘零；采用新分离Encoding/Network布局，历史模型不作匹配基线。存储参数数相同，但有效空间容量和梯度路径不同。',
                  f"开发验证位移均值{percent(bf['validation_u_mean_relative_l2'])} → {percent(bc['validation_u_mean_relative_l2'])}，最差{percent(bf['validation_u_worst_relative_l2'])} → {percent(bc['validation_u_worst_relative_l2'])}；同一位移选模检查点的P1代理均值{percent(bf['validation_P1_mean_relative_l2'])} → {percent(bc['validation_P1_mean_relative_l2'])}。选择步{bf['best_step']}／{bc['best_step']}，两组耗时{(bf['elapsed_seconds']+bc['elapsed_seconds'])/60:.1f}分钟。",
                  f"训练位移均值{percent(bf['train_u_mean_relative_l2'])} → {percent(bc['train_u_mean_relative_l2'])}，训练P1均值{percent(bf['train_P1_mean_relative_l2'])} → {percent(bc['train_P1_mean_relative_l2'])}。匹配检查通过，四项预定综合标准{'通过' if bandwidth_result['clear_gain_gate_passed'] else '未通过'}；没有调整门槛。",
                  '这是单种子的开发验证和表示效用对照，不能单独识别频率原因，也不保证场带限或加速。P1为位移导出的代理，最终测试和dia未用于本组。完整分析见notes/geobc_hash_bandwidth_pair_result.json。']
        if (dest/'hash_bandwidth_comparison.png').is_file():
            lines += ['', '![完整12层和前8层的位移与P1验证结果](hash_bandwidth_comparison.png)']
    path_result_path=ROOT/'notes/geobc_hash_condition_path_result.json'
    path_result=json.loads(path_result_path.read_text()) if path_result_path.is_file() else None
    if path_result and path_result.get('status')!='completed':path_result=None
    if path_result:
        witness=path_result['sampling']
        lines += ['', '## 固定训练几何的Hash与条件路径诊断','',
                  f"固定TRAIN101_479，两组自身位移选模best9000，至多1024个输入分层单元；实际{witness['sample_cells']}个，BC位角点变化／不变各{witness['strata']['BC_change']['sample']}／{witness['strata']['no_BC_change']['sample']}个。先提交选点再读同TRAIN参考，耗时{path_result['elapsed_seconds']:.2f}s，无反向传播或更新。变化层在原网格只有{witness['strata']['BC_change']['population']}/{witness['population_cells']}个单元，因此该过采样合并结果不能当作全CAD估计。",
                  '从实际公共decoder输入捕获masked Hash24与全部条件107；用corner0为锚的四次交叉配方，按对称约定分配交互，保留真实交叉项。自然输出精确重放，位移差先转FP64，再计算物理P1。',
                  '| 载荷 | Hash路径RMS：12层→8层 | 条件路径RMS：12层→8层 | 抽样cell P1相对误差：12层→8层 |',
                  '|---|---:|---:|---:|']
        for load_name in ('ver','hor','tor'):
            pf=path_result['models']['full12']['per_load'][load_name]['combined']
            pc=path_result['models']['coarse8']['per_load'][load_name]['combined']
            lines += [f"| {load_name} | {pf['H_RMS']:.4g}→{pc['H_RMS']:.4g} | {pf['C_RMS']:.4g}→{pc['C_RMS']:.4g} | {percent(pf['sampled_cell_relative_l2'])}→{percent(pc['sampled_cell_relative_l2'])} |"]
        lines += ['', '两条路径的P1 RMS、交互RMS以及两个分层内的误差均下降；H/C交叉项为负，存在抵消，不能把各项能量当成互相独立的错误占比。全部条件包括离散BC、局部几何、距离、平滑query与固定global，不可把C统一称作边界噪声。',
                  '这是单TRAIN几何、输入分层样本与可能离流形的交叉输入描述；不能识别因果、代表全CAD/验证集，或部署成纠正场。RMS无量纲，P1仍为位移导出代理。正式结果见notes/geobc_hash_condition_path_result.json。']
    topology_result_path=ROOT/'notes/geobc_topology_context_feasibility_result.json'
    topology_result=json.loads(topology_result_path.read_text()) if topology_result_path.is_file() else None
    if topology_result and topology_result.get('status')!='completed':topology_result=None
    balanced_result_path=ROOT/'notes/geobc_topology_balanced_feasibility_result.json'
    balanced_result=json.loads(balanced_result_path.read_text()) if balanced_result_path.is_file() else None
    if balanced_result and balanced_result.get('status')!='completed':balanced_result=None
    if topology_result:
        lines += ['', '## 网格拓扑上下文的输入可行性','',
                  f"首次四TRAIN输入准备完成科学可行性诊断，耗时{topology_result['elapsed_seconds']:.2f}s，实际完成缓存{len(topology_result['completed_CADs'])}/4，没有读取标签、模型或初始化CUDA。"]
        for cad,record in topology_result.get('blocked_CADs',{}).items():
            e=record['evidence']
            lines += [f"{cad}的bounded BFS在分配{e['assigned_vertices']:,}/{e['vertices']:,}节点时形成{e['observed_aggregate_count']:,}个组，超过{e['hard_maximum']:,}硬上限；角色诱导组件下界仅{e['role_components']['lower_bound']}。这是所选贪心粗化的碎片化，而非必要的BC组件过多；其他三个几何尚未处理。"]
        lines += ['随后另行冻结实际边上的小组合并方案，保持角色、规模上限、完整P/R和平滑公式；旧结果和源码保留。这是输入工程可行性，不是模型精度或泛化收益。']
    if balanced_result:
        lines += [f"新合并方案耗时{balanced_result['elapsed_seconds']:.2f}s，可行性{'通过' if balanced_result['feasibility_pass'] else '未通过'}，完成{len(balanced_result['completed_CADs'])}/4缓存，共{balanced_result['cache_bytes']/2**20:.1f}MiB。没有自动放行训练。"]
        if balanced_result['feasibility_pass']:
            lines += ['', '| TRAIN CAD | 原节点 | 中间／粗节点 | P01非零元 |','|---|---:|---:|---:|']
            for cad,record in balanced_result['CADs'].items():
                f=record['facts']
                lines += [f"| {cad} | {f['N0']:,} | {f['N1']:,}/{f['N2']:,} | {f['maps']['P01']['nnz']:,} |"]
        lines += ['正式记录见notes/geobc_topology_balanced_feasibility_result.json；节点排序依赖、P1代理质量、任意坐标连续查询与重网格泛化仍未验证。']
    topology_message_path=ROOT/'notes/geobc_topology_message_pair_result.json'
    topology_message= json.loads(topology_message_path.read_text()) if topology_message_path.is_file() else None
    if topology_message and topology_message.get('status')=='completed':
        lines += ['', '## 四TRAIN几何：学习邻居消息的固定预算试验','',
                  '两个fresh模型均使用coarse8 Hash、同一完整网格聚合／平滑、Euclidean43与整体BC；唯一差异为消息层使用自身或真实邻居加权平均。12个TRAIN案例，固定1000步端点，无验证选模。自身组也含共同R/P空间混合，不能称为无拓扑基线。','',
                  '| 消息 | TRAIN位移均值／最差 | TRAIN全节点P1均值 | 物理P1误差RMS案例均值 | 耗时 |',
                  '|---|---:|---:|---:|---:|']
        for mode,label in [('self_only','自身'),('neighbors','邻居')]:
            m=topology_message['arms'][mode]['metrics']
            lines += [f"| {label} | {percent(m['train_u_mean_relative_l2'])}／{percent(m['train_u_worst_relative_l2'])} | {percent(m['train_P1_mean_relative_l2'])} | {m['train_P1_mean_case_physical_tensor_error_RMS']:.6g} | {m['elapsed_seconds']:.1f}s |"]
        passed=sum(g['passed'] for g in topology_message['clear_gain_gate'].values())
        lines += ['',f"完整初始化、采样和配置匹配核对通过；联合门限{'通过' if topology_message['clear_gain_gate_passed'] else '未通过'}（{passed}/4）。P1为位移导出的corner-FE代理，其物理误差RMS含shear平方权重2且不除以6。这里只检验四TRAIN拟合效用，不能推断跨几何或任意坐标连续查询。结果见notes/geobc_topology_message_pair_result.json。"]
    continuous_geometry_path=ROOT/'notes/geobc_continuous_geometry_result.json'
    continuous_geometry=json.loads(continuous_geometry_path.read_text()) if continuous_geometry_path.is_file() else None
    if continuous_geometry and continuous_geometry.get('status')!='completed':continuous_geometry=None
    continuous_geometry_receipt=None
    if continuous_geometry:
        continuous_report,continuous_brief=continuous_geometry_text(continuous_geometry)
        lines += ['', '## 有界连续坐标的CPU几何接口验收','',continuous_report]
        geometry_jobs=[job for job in jobs if job['name']=='continuous_query_geometry_train101_479' and job['name'] in summaries]
        if geometry_jobs:
            geometry_job=geometry_jobs[-1]
            exported=Path('runs')/geometry_job['name']
            continuous_geometry_receipt={'run':geometry_job['name'],'exported_files':[str(exported/name).replace('\\','/') for name in ('config.json','status.json','summary.json')],
                                         'receipt_location':'summary.json:receipt','status':continuous_geometry['status'],
                                         'scientific_outcome':continuous_geometry['scientific_outcome'],'gate_pass':continuous_geometry['gate_pass']}
    quadratic_kinematics_path=ROOT/'notes/geobc_quadratic_kinematics_result.json'
    quadratic_kinematics=json.loads(quadratic_kinematics_path.read_text()) if quadratic_kinematics_path.is_file() else None
    if quadratic_kinematics and quadratic_kinematics.get('status')!='completed':quadratic_kinematics=None
    quadratic_kinematics_receipt=None
    if quadratic_kinematics:
        quadratic_report,quadratic_brief=quadratic_kinematics_text(quadratic_kinematics)
        lines += ['', '## 固定保存场的二次运动学诊断','',quadratic_report]
        quadratic_jobs=[job for job in jobs if job['name']=='quadratic_kinematics_specialist101_479_ver' and job['name'] in summaries]
        if quadratic_jobs:
            quadratic_job=quadratic_jobs[-1]; exported=Path('runs')/quadratic_job['name']
            quadratic_kinematics_receipt={'run':quadratic_job['name'],'exported_files':[str(exported/name).replace('\\','/') for name in ('config.json','status.json','summary.json')],
                                          'receipt_location':'summary.json:receipt','status':quadratic_kinematics['status'],
                                          'scientific_outcome':quadratic_kinematics['scientific_outcome'],'gate_pass':quadratic_kinematics['gate_pass']}
    fullcad_kinematics_path=ROOT/'notes/geobc_fullcad_quadratic_kinematics_result.json'
    fullcad_kinematics=json.loads(fullcad_kinematics_path.read_text()) if fullcad_kinematics_path.is_file() else None
    if fullcad_kinematics and fullcad_kinematics.get('status')!='completed':fullcad_kinematics=None
    fullcad_kinematics_receipt=None
    if fullcad_kinematics:
        fullcad_report,fullcad_brief=fullcad_quadratic_kinematics_text(fullcad_kinematics)
        lines += ['', '## 同一TRAIN保存场的全CAD确认','',fullcad_report]
        fullcad_jobs=[job for job in jobs if job['name']=='fullcad_quadratic_kinematics_specialist101_479_ver' and job['name'] in summaries]
        if fullcad_jobs:
            fullcad_job=fullcad_jobs[-1]; exported=Path('runs')/fullcad_job['name']; receipt=fullcad_kinematics['receipt']
            fullcad_kinematics_receipt={'run':fullcad_job['name'],'exported_files':[str(exported/name).replace('\\','/') for name in ('config.json','status.json','summary.json')],
                                      'receipt_location':'summary.json:receipt','status':fullcad_kinematics['status'],
                                      'scientific_outcome':fullcad_kinematics['scientific_outcome'],'gate_pass':fullcad_kinematics['gate_pass'],
                                      'backend':receipt['backend'],'device':receipt['device'],'dtype':receipt['dtype'],
                                      'GPU_event_elapsed_seconds':receipt['GPU_event_elapsed_seconds'],
                                      'GPU_peak_allocated_bytes':receipt['GPU_peak_allocated_bytes'],'GPU_peak_reserved_bytes':receipt['GPU_peak_reserved_bytes']}
    def latest_observed(kind):
        return next((observed_stages[job['name']] for job in reversed(jobs)
                     if job.get('kind')==kind and job['name'] in observed_stages),None)
    capability_result=latest_observed('solver_continuous_capability')
    gino_radius_result=latest_observed('gino_radius_input_pilot')
    gino_resource_result=latest_observed('gino_native_resource')
    gino_training_result=latest_observed('gino_training_screen')
    sealed_acquisition_result=latest_observed('sealed_test_input_acquisition')
    capability_brief=gino_radius_brief=gino_resource_brief=gino_training_brief=sealed_acquisition_brief=None
    if capability_result:
        capability_report,capability_brief=solver_capability_text(capability_result)
        lines += ['', '## 锁定FNO/MLP端点的局部连续能力','',capability_report]
    if gino_radius_result:
        gino_radius_report,gino_radius_brief=gino_radius_text(gino_radius_result)
        lines += ['', '## GINO候选的输入半径图可行性','',gino_radius_report]
    if gino_resource_result:
        gino_resource_report,gino_resource_brief=gino_resource_text(gino_resource_result)
        lines += ['', '## GINO有界模型资源','',gino_resource_report]
    if gino_training_result:
        gino_training_report,gino_training_brief=gino_training_text(gino_training_result)
        lines += ['', '## GINO固定480更新初筛','',gino_training_report]
    if sealed_acquisition_result:
        sealed_acquisition_report,sealed_acquisition_brief=sealed_acquisition_text(sealed_acquisition_result)
        lines += ['', '## 下一周期独立test5的封存获取','',sealed_acquisition_report]
    lines += ['', '## 主计划中的下一阶段','',next_text]
    lines += ['', '## 解释边界','',
              '两个 CAD 的结果是迁移 pilot，存在其他历史实验使用；不是最终盲测。九个组合的四类载荷均已在其他训练 CAD 出现，不能称为全局未见载荷类别。新模型的最终测试CAD仅指未参与其训练/选模，原50案例历史实验包含这些CAD。',
              '迁移尾部误差可能被 asinh 逆变换放大；不截断预测、不更改标签，也不把训练误差称为泛化误差。',
              '候选模型组合已有的 [PointNet](https://arxiv.org/abs/1612.00593) 点集汇聚与 [Instant-NGP](https://nvlabs.github.io/instant-ngp/) hash 表示；当前尚无新算法或泛化成功的结论。','',
              f"本页随实验更新；科学训练／评估／冻结诊断完成{counts['scientific_completed']}/{counts['scientific_planned']}项，合成验收完成{counts['synthetic_acceptance_completed']}/{counts['synthetic_acceptance_planned']}项。原30项slides复现已完成，自主研究继续。",'']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    brief=['# GeoBC：一分钟结果','',
           '**Hash能拟合单个几何，当前难点是让一个共享模型可靠地预测新几何。**','']
    if specialist:
        f=specialist['endpoint_metrics']['10000']['train']
        brief += [f"- **已有能力：**一个训练CAD的位移／P1应变代理误差{100*f['case_mean_relative_l2']:.2f}%／{100*f['strain']['case_mean_relative_l2']:.1f}%。这是专用拟合，不能代表泛化。"]
    else:
        brief += ['- **已有能力：**难例单独训练的应力误差约0.1%；共享预测新几何仍很差。']
    if bandwidth_result:
        brief += [f"- **最新进展：**完整12层／前8层Hash的验证位移{100*bf['validation_u_mean_relative_l2']:.2f}% → {100*bc['validation_u_mean_relative_l2']:.2f}%，P1代理{100*bf['validation_P1_mean_relative_l2']:.1f}% → {100*bc['validation_P1_mean_relative_l2']:.1f}%；两组耗时{(bf['elapsed_seconds']+bc['elapsed_seconds'])/60:.1f}分钟。",
                  f"- **判断：**最差位移{100*bf['validation_u_worst_relative_l2']:.1f}% → {100*bc['validation_u_worst_relative_l2']:.1f}%，预定综合标准{'通过' if bandwidth_result['clear_gain_gate_passed'] else '未通过'}。两组存储容量相同、有效容量不同；这项对比不单独证明频率原因，跨几何可靠性仍需后续证据。"]
    elif graph_result:
        e=graph_result['runs']['euclidean'];g=graph_result['runs']['graph']
        brief += [f"- **最新进展：**同边界欧氏／网格路径距离的验证位移{100*e['validation_u_mean_relative_l2']:.2f}% → {100*g['validation_u_mean_relative_l2']:.2f}%，{'达到' if graph_result['clear_gain_gate_passed'] else '未达到'}预定综合门槛；两组计算共{(e['elapsed_seconds']+g['elapsed_seconds'])/60:.1f}分钟。",
                  f"- **当前瓶颈：**网格路径模型的P1应变代理误差{100*g['validation_P1_mean_relative_l2']:.0f}%，最差位移{100*g['validation_u_worst_relative_l2']:.1f}%。这是一个种子的开发验证；距离表示的帮助须结合全项门槛判断，尚未使用最终测试。"]
        if error_gradient:
            brief[-1]=f"- **当前瓶颈：**固定训练代表10_58/tor去除整体仿射误差后，欧氏／网格路径模型仍保留{100*ep['residual_over_raw_error_energy']:.1f}%／{100*gp['residual_over_raw_error_energy']:.1f}%的P1误差能量。这是依赖GT的单案例描述，不能据此断定频率原因或全体验证瓶颈。"
    elif hash_result:
        hz=hash_result['runs']['zero'];ha=hash_result['runs']['actual']
        brief += [f"- **最新进展：**Hash调制配对的验证位移{100*hz['validation_u_mean_relative_l2']:.2f}% → {100*ha['validation_u_mean_relative_l2']:.2f}%，{'达到' if hash_result['clear_gain_gate_passed'] else '未达到'}预定综合门槛；两组计算合计{(hz['elapsed_seconds']+ha['elapsed_seconds'])/60:.1f}分钟。"]
        if hash_use:
            brief += [f"- **当前瓶颈：**调制已被模型使用，但共享验证没有改善。约{100*hash_use['actual']['near_endpoint_fraction_mean4']:.0f}%的采样通道倍率接近范围端点；关闭调制可能离分布，不能用该诊断确定泛化失败原因。"]
        else:
            brief += [f"- **当前瓶颈：**实际分支的P1代理误差{100*ha['validation_P1_mean_relative_l2']:.0f}%，最差位移{100*ha['validation_u_worst_relative_l2']:.1f}%；预定综合门槛{'通过' if hash_result['clear_gain_gate_passed'] else '未通过'}。这是开发验证，跨几何可靠性尚需按主计划判断。"]
    elif qz and qa and query_result:
        z=qz['best_metrics']['validation'];a=qa['best_metrics']['validation']
        brief += [f"- **最新进展：**局部查询上下文将验证位移误差{100*z['case_mean_relative_l2']:.1f}% → {100*a['case_mean_relative_l2']:.1f}%；两组计算合计{(qz['elapsed_seconds']+qa['elapsed_seconds'])/60:.1f}分钟。",
                  f"- **当前瓶颈：**P1应变代理虽下降{100*query_result['validation_P1_relative_error_reduction']:.0f}%，仍有{100*a['strain']['case_mean_relative_l2']:.0f}%误差。预定综合标准{'通过' if query_result['clear_gain_gate_passed'] else '未通过'}，跨几何可靠性仍未解决。"]
    else:
        brief += ['- **当前瓶颈：**共享模型的位移验证约24%，P1代理仍约584%。',
                  '- **当前证据：**科学结果以已完成配对分析为准，阶段状态见主计划；合成接口验收没有泛化指标。']
    if plan['stage']=='matched_fixed_hash_bandwidth':
        bandwidth_native_path=ROOT/'notes/hash_bandwidth_native_ready.json'
        if bandwidth_native_path.is_file():
            bandwidth_native=json.loads(bandwidth_native_path.read_text())
            brief += ['', '本组Hash层数对照的合成接口记录已生成；模型表现以完成的配对训练分析为准。']
        else:
            brief += ['', '本组Hash层数对照正在准备模型与训练入口，合成接口尚未验收；已有距离缓存直接复用。']
    elif bandwidth_result:
        pass  # Completed training analysis already establishes this pair's scope.
    elif graph_native:
        brief += ['',f"上一组距离对照的native接口验收已通过（{graph_native['elapsed_seconds']:.2f}s、零参数更新），只证明接口可用。"]
    elif native:
        brief += ['',f"新native合成验收{'已通过' if native_passed else '未通过'}（{native['elapsed_seconds']:.2f}s、零参数更新），不代表物理表现。"]
    if path_result:
        brief += ['', f"冻结诊断（{path_result['elapsed_seconds']:.1f}秒）显示两条路径的空间变化都减小，不能把瓶颈归于单一分支；这是单TRAIN几何的分层抽样结果。"]
    if topology_result and not topology_result['feasibility_pass'] and not balanced_result:
        brief += ['首次粗网格准备发现贪心粗化碎片化，尚无缓存完成；现保持约束与上限，改用真实边上的同角色小组合并。']
    elif balanced_result:
        brief += [f"粗网格输入可行性{'通过' if balanced_result['feasibility_pass'] else '未通过'}（{balanced_result['elapsed_seconds']:.1f}秒，{len(balanced_result['completed_CADs'])}/4几何）；训练收益尚待配对检验。"]
    if topology_message and topology_message.get('status')=='completed':
        tm0=topology_message['arms']['self_only']['metrics'];tm1=topology_message['arms']['neighbors']['metrics']
        brief[-1]=f"四几何固定预算拟合：自身→邻居消息，位移{100*tm0['train_u_mean_relative_l2']:.2f}%→{100*tm1['train_u_mean_relative_l2']:.2f}%，P1代理{100*tm0['train_P1_mean_relative_l2']:.1f}%→{100*tm1['train_P1_mean_relative_l2']:.1f}%；联合门限{'通过' if topology_message['clear_gain_gate_passed'] else '未通过'}。这是TRAIN拟合，尚无新增泛化结论。"
    if continuous_geometry:brief += [continuous_brief]
    if quadratic_kinematics:brief += [quadratic_brief]
    if fullcad_kinematics:brief += [fullcad_brief]
    brief += ['',f'下一步（来自主计划）：{brief_next_text} 测试使用状态以冻结评估与新周期封存计划为准。',
              '',f"科学训练／评估／冻结诊断完成{counts['scientific_completed']}/{counts['scientific_planned']}项，合成验收完成{counts['synthetic_acceptance_completed']}/{counts['synthetic_acceptance_planned']}项；原30项slides复现已完成。自主研究与30分钟监视继续。",'',
              '完整指标、控制变量和解释范围见 [探索报告](REPORT.md)。','']
    if plan.get('stage','').startswith('solver_'):
        brief=['# GeoBC：一分钟结果','',
               '**尚未选定求解器。当前优先做跨架构泛化筛选，暂缓Hash局部优化。**','',
               f'- **完成：**原30项slides复现与私人仓库交付；后续探索已完成{len(summaries)}个阶段。这些阶段包含训练、诊断和输入准备，不是求解器种数。',
               '- **已知：**Hash小样本节点拟合强；已有跨CAD结果仍不足以证明它适合GeoBC。Transformer历史结果主要是overfit，也不足以淘汰这些路线。',
               '- **评价问题：**一个保存训练场的全CAD四角／十节点应变代理误差22.4%／187.3%，说明仅看位移或四角梯度会漏掉问题；它不是solver应变GT。',
               f'- **当前方向：**{brief_next_text} 六家族初筛结果见[选型简表](../solver_selection/BRIEF.md)；确认使用相同几何/BC输入、训练集归一化和全原节点物理指标。',
               '- **决策标准：**未见CAD平均与最差精度、稳定性、训练及推理成本。短预算结果用于筛选，下降明显的慢路线保留确认机会。','',
               '最终测试继续保留；自主研究与30分钟监视继续。',
               '配置见 [选型计划](../../notes/geobc_solver_selection_plan.json)，历史证据见 [探索报告](REPORT.md)。','']
    if summaries.get('solver_final_validation', {}).get('status') == 'completed':
        brief=['# GeoBC：一分钟结果','',
               '**六家族筛选、双种子固定成本确认及一次冻结最终验证已完成。FNO是主候选，MLP是低成本对照；尚未证明工程可用。**','',
               f'- **完成：**原30项slides复现；后续{len(summaries)}个阶段包含训练、诊断、输入准备和最终评估，并非求解器种数。',
               '- **新CAD：**FNO的应力误差较低且两seed稳定；MLP一个seed位移误差更低，另一个明显退化，不能称FNO所有指标都胜出。',
               '- **组合载荷：**两条路线都明显变差，且seed波动较大。节点拟合或known-load结果不足以证明GeoBC泛化。',
               '- **后续规则：**不据旧final调参、选择seed或重训本轮模型；新结构使用另外冻结、仍封存的test5。','',
               '凝练结果见[求解器简报](../SOLVER_BRIEF.md)，完整冻结known3／dia结果见[最终评估](../solver_final_eval/BRIEF.md)。',
               '当前选型限于共同数据、配方与预算；自主研究与30分钟监视继续。','']
    if capability_brief:brief += ['- **局部连续能力：**'+capability_brief]
    if gino_radius_brief:brief += ['- **新结构输入：**'+gino_radius_brief]
    if gino_resource_brief:brief += ['- **新结构资源：**'+gino_resource_brief]
    if gino_training_brief:brief += ['- **新结构初筛：**'+gino_training_brief]
    if sealed_acquisition_brief:brief += ['- **新周期test：**'+sealed_acquisition_brief]
    fno_response_metrics=ROOT/'results/fno_response/metrics.json'
    if fno_response_metrics.is_file() and json.loads(fno_response_metrics.read_text(encoding='utf8')).get('status')=='completed':
        brief += ['', '固定480的FNO响应基结构配对与DEV组合载荷结果见[FNO结构简报](../fno_response/BRIEF.md)；历史direct4仅上下文，未据此改旧final或新sealed test。']
    deeponet_metrics=ROOT/'results/deeponet_screen/metrics.json'
    if deeponet_metrics.is_file() and json.loads(deeponet_metrics.read_text(encoding='utf8')).get('status')=='completed':
        brief += ['', '共同BC22输入、固定480的DeepONet/concat-MLP与DEV组合载荷结果见[结构初筛简报](../deeponet_screen/BRIEF.md)；heads/容量不同，单seed早期结果不是工程资格或full-model初始化匹配。']
    mesh_metrics=ROOT/'results/multiscale_mesh/metrics.json'
    if mesh_metrics.is_file() and json.loads(mesh_metrics.read_text(encoding='utf8')).get('status')=='completed':
        brief += ['', '一个完整节点的多尺度网格求解器固定480结果见[网格结构简报](../multiscale_mesh/BRIEF.md)；参数/拓扑输入不同，仅早期筛选，未据此改旧final或新sealed test。']
    deeponet_ready_exports=[]
    if any(job.get('kind') in ('deeponet_bc_cloud_input','deeponet_screen','deeponet_development_dia_eval') for job in jobs):
        for ready_name in ('deeponet_field_source_ready.json','deeponet_data_source_ready.json','deeponet_train_source_ready.json','deeponet_report_source_ready.json'):
            ready_source=ROOT/'notes'/ready_name
            if ready_source.is_file():
                ready_target=dest/'source_ready'/ready_name;ready_target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ready_source,ready_target)
                deeponet_ready_exports.append(str(ready_target.relative_to(dest)).replace('\\','/'))
    mesh_ready_exports=[]
    if any(job.get('kind') in ('multiscale_mesh_input','multiscale_mesh_screen','multiscale_mesh_development_dia_eval') for job in jobs):
        for ready_name in ('multiscale_mesh_field_source_ready.json','multiscale_mesh_data_source_ready.json','multiscale_mesh_train_source_ready.json','multiscale_mesh_report_source_ready.json'):
            ready_source=ROOT/'notes'/ready_name
            if ready_source.is_file():
                ready_target=dest/'source_ready'/ready_name;ready_target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ready_source,ready_target)
                mesh_ready_exports.append(str(ready_target.relative_to(dest)).replace('\\','/'))
    (dest/'BRIEF.md').write_text('\n'.join(brief),encoding='utf8')
    write_json(dest/'manifest.json',{'scope':'GeoBC exploration; independent of the completed slides suite',
                                   'completed':len(summaries),'planned':len(jobs),
                                   **counts,
                                   'phase':'design_in_progress' if not plan['jobs'] else 'ongoing' if len(summaries)<len(jobs) else 'stage_completed',
                                   'research_status':'ongoing','current_stage':plan['stage'],
                                   'current_stage_complete':current_stage_complete,
                                   'original_slides_completed':plan.get('prior_completed_slides_experiments',30),
                                   'original_slides_planned':30,'original_slides_status':'completed',
                                   'counting_policy':'scientific counts include training/evaluation/frozen real-data diagnostics; synthetic native validation is separate and has no generalization result; completed/planned retain total stage counts',
                                   'next_stage_source':'notes/geobc_exploration_plan.json',
                                   'hash_modulation_pair_analysis':str(hash_result_path.relative_to(ROOT)) if hash_result else None,
                                   'hash_modulation_use_analysis':str(hash_use_path.relative_to(ROOT)) if hash_use else None,
                                   'graph_context_pair_analysis':str(graph_result_path.relative_to(ROOT)) if graph_result else None,
                                   'error_gradient_analysis':str(error_gradient_path.relative_to(ROOT)) if error_gradient else None,
                                   'hash_bandwidth_pair_analysis':str(bandwidth_result_path.relative_to(ROOT)) if bandwidth_result else None,
                                   'hash_condition_path_analysis':str(path_result_path.relative_to(ROOT)) if path_result else None,
                                   'topology_input_analysis':str(topology_result_path.relative_to(ROOT)) if topology_result else None,
                                   'balanced_topology_input_analysis':str(balanced_result_path.relative_to(ROOT)) if balanced_result else None,
                                   'topology_message_pair_analysis':str(topology_message_path.relative_to(ROOT)) if topology_message else None,
                                   'continuous_query_geometry_analysis':str(continuous_geometry_path.relative_to(ROOT)) if continuous_geometry else None,
                                   'continuous_query_geometry_cpu_receipt':continuous_geometry_receipt,
                                   'quadratic_kinematics_analysis':str(quadratic_kinematics_path.relative_to(ROOT)) if quadratic_kinematics else None,
                                   'quadratic_kinematics_cpu_receipt':quadratic_kinematics_receipt,
                                   'fullcad_quadratic_kinematics_analysis':str(fullcad_kinematics_path.relative_to(ROOT)) if fullcad_kinematics else None,
                                   'fullcad_quadratic_kinematics_gpu_receipt':fullcad_kinematics_receipt,
                                   'solver_continuous_capability_receipt':observed_receipts.get('solver_continuous_capability'),
                                   'gino_radius_input_pilot_receipt':observed_receipts.get('gino_radius_input_pilot'),
                                   'gino_native_resource_receipt':observed_receipts.get('gino_native_resource'),
                                   'gino_training_screen_receipt':observed_receipts.get('gino_training_screen'),
                                   'sealed_test_input_acquisition_receipt':observed_receipts.get('sealed_test_input_acquisition'),
                                   'deeponet_bc_cloud_input_receipt':observed_receipts.get('deeponet_bc_cloud_input'),
                                   'deeponet_screen_receipt':observed_receipts.get('deeponet_screen'),
                                   'deeponet_development_dia_eval_receipt':observed_receipts.get('deeponet_development_dia_eval'),
                                   'deeponet_source_ready_exports':deeponet_ready_exports,
                                   'multiscale_mesh_input_receipt':observed_receipts.get('multiscale_mesh_input'),
                                   'multiscale_mesh_screen_receipt':observed_receipts.get('multiscale_mesh_screen'),
                                   'multiscale_mesh_development_dia_eval_receipt':observed_receipts.get('multiscale_mesh_development_dia_eval'),
                                   'multiscale_mesh_source_ready_exports':mesh_ready_exports,
                                   'observed_noncompleted_stages':{name:result.get('status') for name,result in observed_stages.items() if result.get('status')!='completed'},
                                   'new_stage_export_policy':'config/status/summary JSON only; no8 NPY CSR cache, raw labels or weights copied; failed acquisition receipts exported without completed-stage credit',
                                   'summaries':list(summaries),'raw_data_checkpoints_predictions':'retained locally; not included in Git payload',
                                   'updated_at_utc':utc_now()})
    if geo and load and comparisons and not (preserve_existing_plots and (dest/'diagnosis.png').is_file()):
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
    print(json.dumps({'completed':len(summaries),'planned':len(jobs),**counts,'research_status':'ongoing','report':str(dest/'REPORT.md')}))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Export small GeoBC receipts and reports')
    parser.add_argument('--preserve-existing-plots',action='store_true',help='Reuse unchanged historical diagnosis figure')
    main(preserve_existing_plots=parser.parse_args().preserve_existing_plots)
