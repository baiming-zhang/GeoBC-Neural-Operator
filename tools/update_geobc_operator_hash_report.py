"""Reconcile actual scalar results and publication allowlist, without GPU/network."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def write(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    actual=read('results/geobc_operator_hash_v5/manifest.json');assert actual['status']=='completed'
    native=read('runs/geobc_exploration/macro_operator_hash_native_interface/summary.json')
    outer=read('runs/geobc_exploration/macro_operator_hash_native_interface/external_terminal.json')
    assert native['status']=='completed' and native['native_TCNN_interface_verified'] and outer['within_external_budget']
    folder=ROOT/'results/geobc_trained_operator_hash_queries';folder.mkdir(exist_ok=True)
    write('results/geobc_trained_operator_hash_queries/manifest.json',dict(status='completed',actual=native,outer=outer,
          root_exact_exits_verified=True,physical_accuracy_claim=False,independent_offnode_FEM_accuracy_measured=False))
    (folder/'BRIEF.md').write_text('# 训练后真实CUDA连续查询\n\n'
        '使用全局Hash4800步真实权重、固定首TRAIN CAD／ver的输入，读取目标0、训练更新0。整轮4.83秒；'
        '257节点的原查询／重算stencil／分块／重排最大encoded误差均0。'
        '64内部连续点在显式固定native25扩展下，坐标、局部特征、几何context与算子latent梯度有限非零。\n\n'
        '坐标扰动从1e-3减至1e-5，最大encoded输出变化0.3701→0.05273→0.005859。'
        '这些是官方TCNN实际数值接口和固定上下文检查，未构造真实离节点BC场，也未测独立离节点FEM精度；不代表物理精度或C1光滑。\n',encoding='utf-8')
    summary=['# GeoBC：一分钟结果','',
        '**当前充分训练的主候选仍是FNO100／12000：新CAD位移／应力误差23.17%／50.85%。全局Hash已成为值得继续比较的候选，最终选型未定。**','',
        '同100训练CAD、4800步／16遍、20开发CAD×三载荷全原节点：','',
        '| 方法 | 位移误差 | 应力误差 |','|---|---:|---:|']
    for p in actual['points']:summary.append(f"| {p['label']} | {p['U_percent']:.2f}% | {p['S_percent']:.2f}% |")
    summary+=['','全局Hash相对FNO：位移差1.27个百分点，应力好2.48、热点好6.74个百分点；20/20 CAD应力／热点更好，位移仅5/20更好。总容量12.25M vs4.53M且混合精度不同，不是纯容量匹配对照。',
        '', '全局latent置零，误差升至83.73%／100.00%；几何descriptor置零35.43%／64.29%。这是未重训的分布外依赖诊断，不能把零输入版本当求解器或因果消融。[结果与成本](geobc_operator_hash_v5/BRIEF.md)。',
        '', '训练后真实CUDA连续查询已通过，目标读取和训练更新均0，整轮4.83秒。[接口范围](geobc_trained_operator_hash_queries/BRIEF.md)。仅给有竞争力的全局Hash追加7200步到12000／40遍，保留model／Adam／四类RNG／曝光cursor，不改配方；追加训练上限10分钟，整体20分钟。',
        '', '历史同40CAD／12000／约4.53M公平对照：FNO26.50%／55.55%，Transolver++52.35%／83.99%。TS系列以++为主；当前配方FNO领先，不宣布架构上限。[TS比较](geobc_short_screen/BRIEF.md)。',
        '', 'Hash<1%属于已知单／双样本应力拟合，不能直接比较新CAD联合位移／应力。全部保留原版GRID对齐CSV、signed U(mm)/S(MPa)、原节点；oldTRAIN40-only scaler、knownDEV非FINAL，FINAL数值标签未读。单seed、可用性条件池，尚未证明独立离节点FEM精度。']
    (ROOT/'results/SOLVER_BRIEF.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    text=(ROOT/'README.md').read_text(encoding='utf-8');a=text.index('下一宏观比较使用同100CAD');b=text.index('\n\n此前原生++',a)
    text=text[:a]+('同100CAD／4800更新宏观比较已完成：FNO28.59%／56.49%，几何条件Hash41.27%／61.61%，全局算子Hash29.86%／54.01%。'
        '全局Hash20/20开发CAD应力与热点更低，但位移较差、参数及混合精度不同；未全面胜出。全局latent置零导致83.73%／100.00%，仅为推理依赖诊断。'
        '240完整有限预测先提交／重验再knownGT60共享一次，开发整轮240.66秒，CPU导出18.22秒，实际图已验看。[三行比较](results/geobc_operator_hash_v5/BRIEF.md)。\n\n'
        '真实TCNN全节点梯度／FNO初始化字节检查通过；训练后真实CUDA连续查询整轮4.83秒通过节点／chunk／perm一致性和输入梯度，GT／更新0。'
        '原全局阶段元数据getter技术失败126秒与修复间隔469秒均保留，原20分钟绝对budget未重置；恢复训练287.95秒、原完整campaign1011.26秒。'
        '只将有竞争力的全局Hash从自身4800恢复model／Adam／四RNG／cursor，追加7200到与FNO相同12000／40遍：训练最多10分钟、整体20分钟、30.5GB。'
        '不延长几何Hash、不调微小参数；新端点尚未完成，不声称最终选型。[CUDA接口](results/geobc_trained_operator_hash_queries/BRIEF.md)。')+text[b:]
    text=text.replace('最近已验收提交 `5a8d1d7` 包含三候选10遍与双候选40遍真实结果、代码及100CAD初始启动状态。本次100CAD完成结果和新宏观比较正在整理为下一快照；',
        '最近已验收提交 `5c89902` 包含FNO100CAD实际完成结果。全局Hash宏观对照、必要依赖诊断、CUDA连续接口和续训启动状态正在整理为下一快照；')
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    exploration=read('notes/geobc_exploration_plan.json');exploration['operator_hash_v5']=read('notes/geobc_operator_hash_v5_actual_result_decision.json')
    exploration['operator_hash_extension_v7']=dict(plan='notes/geobc_operator_hash_extension_v7_plan.json',status='released_actual_started_endpoint_pending',main_until_completed='FNO10012000',final_solver_selected=False)
    write('notes/geobc_exploration_plan.json',exploration)
    pub=read('notes/github_publish_plan.json');prior='notes/github_publish_prior_5c89_canonical_receipt.json'
    if not (ROOT/prior).exists():write(prior,read('notes/github_publish_receipt.json'))
    for file in sorted((ROOT/'notes').glob('*.json')):
        if any(m in file.name for m in ('operator_hash','trained_operator','github_gitdata_5c89','github_immutable_push_e40')):
            item=file.relative_to(ROOT).as_posix()
            if item not in pub['allowlist']:pub['allowlist'].append(item)
    if prior not in pub['allowlist']:pub['allowlist'].append(prior)
    pub['commit_message']='Compare whole-BC Hash against FNO on unseen CAD and validate native continuous queries'
    write('notes/github_publish_plan.json',pub)
    print(dict(status='local_actual_reports_updated',raw_reads=0,GPU_calls=0,network_calls=0))


if __name__=='__main__':main()
