"""Local scalar report reconciliation; no raw arrays, GPU or network."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    result = read('results/geobc_fno_coverage_v4/manifest.json')
    assert result['status'] == 'completed' and len(result['points']) == 3
    best = result['points'][2]
    assert best['updates'] == 12000 and best['TRAIN_CAD'] == 100
    short = (f"**最新学习候选优先FNO：100个训练CAD×三载荷、12000更新，在knownDEV20 CAD×三载荷上位移／signed应力误差"
             f"{best['U_percent']:.2f}%／{best['S_percent']:.2f}%。最终求解器尚未选定。**")
    brief = ['# GeoBC：一分钟结果', '', short, '',
        '| FNO训练范围 | 更新／遍数 | 位移误差 | 应力误差 |', '|---|---:|---:|---:|']
    for p in result['points']:
        brief.append(f"| {p['TRAIN_CAD']} CAD | {p['updates']}／{p['passes']} | {p['U_percent']:.2f}% | {p['S_percent']:.2f}% |")
    brief += ['', '扩大CAD覆盖并给予相同40遍曝光后，误差降低4.76／5.21个百分点；位移19/20、应力20/20 CAD改善。相同4800更新时没有改善，不能把更多数据与更多计算的作用混为一谈。整轮训练16.61分钟，开发读出98.23秒。[三行覆盖报告](geobc_fno_coverage_v4/BRIEF.md)。',
        '', '**与Transolver的公平历史对照仍支持FNO。** 同40CAD、R64/native25、约4.53M参数、12000更新：FNO26.50%／55.55%，++52.35%／83.99%。TS系列当前以++为主，不据此宣布其架构上限。[同配置两行对照](geobc_short_screen/BRIEF.md)。',
        '', '**Hash小于1%来自已知少样本场拟合。** 单／双例约0.4%／0.8%；同40CAD、4800更新的几何条件Hash，新CAD位移／应力49.19%／64.96%。不能将单例记忆与跨CAD联合预测直接比较。[Hash与FNO](geobc_two_warm_screens_v3/BRIEF.md)。',
        '', '正在比较同100CAD、4800更新的几何条件Hash与全局几何／BC算子控制的连续Hash；每臂训练最多10分钟，整体最多20分钟。CPU接口检查通过不等于真实TCNN精度；实际指标完成后才判断收益。',
        '', '全部指标为原版CSV按GRID对齐的全原节点、物理单位病例relative-L2均值；保留signed U(mm)/S(MPa)。只有oldTRAIN40拟合scaler，FINAL数值标签未读；knownDEV属于可用官方TRAIN池、单seed，非最终测试。原节点精度和连续查询一致性不能替代独立离节点FEM精度。',
        '', '[原30项复现](../ALL_METHODS_RESULTS.md) · [当前状态](../README.md) · [研究历史](RESEARCH_HISTORY.md)']
    (ROOT / 'results/SOLVER_BRIEF.md').write_text('\n'.join(brief) + '\n', encoding='utf-8')
    comparison = (ROOT / 'results/TRANSOLVER_COMPARISON.md').read_text(encoding='utf-8')
    paragraph = ('\n最新数据覆盖阶段：FNO100CAD／12000更新达到23.17%／50.85%，整轮16.61分钟；'
                 '同40CAD／12000旧参考26.50%／55.55%，但driver与数值轨迹不同，仅作参考。'
                 '同40遍对照和同4800更新对照见[三行报告](geobc_fno_coverage_v4/BRIEF.md)。'
                 '新100CAD结果不与旧40CAD的++冒称完全同配置；目前继续短预算宏观连续场比较。\n')
    if '最新数据覆盖阶段：' not in comparison:
        comparison += paragraph
    comparison = comparison.replace('新100CAD覆盖训练已于北京时间09:46启动，结果待完成。',
                                    '新100CAD覆盖训练和完整DEV均已完成，最新数据覆盖阶段见下文。')
    (ROOT / 'results/TRANSOLVER_COMPARISON.md').write_text(comparison, encoding='utf-8')
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    start = text.index('## 目前结论')
    end = text.index('## 数据与资源', start)
    body = ('## 目前结论\n\n' + short + '\n\n'
        '同40CAD／12000更新、R64/native25、约4.53M参数的历史公平对照：FNO26.50%／55.55%，Transolver++52.35%／83.99%。'
        '目前TS系列以++为主，FNO优先；不宣布架构上限。[同配置比较](results/geobc_short_screen/BRIEF.md)。\n\n'
        'Hash低于1%来自单／双已知病例的应力拟合，不能直接比较新CAD的联合位移／应力。'
        '几何条件Hash在同40CAD／4800更新的新60病例上49.19%／64.96%，FNO27.93%／56.06%；'
        '总训练阶段3.43／4.93分钟。[40遍确认](results/geobc_two_warm_screens_v3/BRIEF.md)。\n\n'
        '## 最新进展\n\n'
        'FNO扩大TRAIN40→100：同4800更新没有改善（28.59%／56.49%）；给予相同40遍曝光、12000更新后改善至23.17%／50.85%，'
        '位移19/20、应力20/20 CAD更低。整轮训练996.53秒（16.61分钟），开发读出98.23秒，CPU导出4.38秒，图已验看；'
        '120份完整有限预测先验证再共享knownGT60一次。未读FINAL／原CSV、新拟合scaler0。'
        '同更新数与同曝光数分别解释，不能声称纯数据单因素。[三行覆盖报告](results/geobc_fno_coverage_v4/BRIEF.md)。\n\n'
        '下一宏观比较使用同100CAD／4800更新：几何条件Hash对照已启动，全局几何／边界条件算子控制的连续Hash排队。'
        '新模型复用FNO完整R64几何／BC／材料上下文，TCNN解码器沿用已有配置；每臂训练最多10分钟、整体20分钟、30.5GB。'
        '初始FNO骨干字节、真实最大全节点梯度及资源在预算内验证。CPU接口检查0.97秒已通过，不能据此宣称GPU精度或新算法。'
        '不重复训练既有FNO100基线，不进行Hash微小救参。[接口范围](results/geobc_operator_hash_interface/BRIEF.md)。\n\n'
        '此前原生++10遍／1200更新阶段26.32分钟，开发69.68%／94.05%，当前配方贵且欠拟合，暂降低投入优先级。'
        '原双24000计划已主动终止，真实非零退出、日志和成本保留；技术交接空argv快照失败已用实际完整进程身份核对，'
        '未重跑预测／GT、不冒称完成。[三候选初筛](results/geobc_three_short_screens/BRIEF.md)。\n\n')
    text = text[:start] + body + text[end:]
    text = text.replace('100CAD实际载入与最大全节点梯度正在本次预算内验证。',
                        '本次100CAD缓存18.39GB、最大全节点梯度已通过，设备使用峰值21.83GB。')
    pub_start = text.index('[私人仓库 GeoBC-Neural-Operator]')
    pub_end = text.index('\n\n更早逐阶段', pub_start)
    text = text[:pub_start] + ('[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；'
        '最近已验收提交 `5a8d1d7` 包含三候选10遍与双候选40遍真实结果、代码及100CAD初始启动状态。'
        '本次100CAD完成结果和新宏观比较正在整理为下一快照；仅在远端精确commit、owner/repo ID及private状态核验后声明发布。'
        '[发布计划](notes/github_publish_plan.json)。') + text[pub_end:]
    (ROOT / 'README.md').write_text(text, encoding='utf-8')
    exploration = read('notes/geobc_exploration_plan.json')
    exploration['coverage100_v4'].update(status='completed_training_and_DEV_export',
        result_manifest='results/geobc_fno_coverage_v4/manifest.json',
        current_main_FNO_U_S_percent=[best['U_percent'], best['S_percent']],
        plot_visually_reviewed=True, not_a_completed_result=False)
    exploration['operator_hash_v5'] = read('notes/geobc_operator_hash_v5_macro_decision.json')
    exploration['operator_hash_v5']['status'] = 'HASH_PN_actual_started_global_hash_queued'
    write('notes/geobc_exploration_plan.json', exploration)
    decision = dict(status='actual_results_reviewed', manifest='results/geobc_fno_coverage_v4/manifest.json',
                    actual_training_whole_seconds=996.5279881954193,
                    actual_DEV_seconds=result['cost']['DEV_readout_UTC_seconds'],
                    actual_export_seconds=result['actual_CPU_export_seconds'], plot_visually_reviewed=True,
                    points=result['points'], current_main_solver='FNO100_step12000',
                    interpretation='MoreCAD coverage helps with sufficient exposure; fixed4800 alone did not help. SingleknownDEVseed, not final or causal dataonlyproof.',
                    next_macro='Same100CAD4800 shapeHash vs wholeBCoperatorHash with existingFNO anchors', final_solver_selected=False)
    write('notes/geobc_fno_coverage100_v4_actual_result_decision.json', decision)
    publication = read('notes/github_publish_plan.json')
    prior = 'notes/github_publish_prior_5a8d_canonical_receipt.json'
    if not (ROOT / prior).exists():
        write(prior, read('notes/github_publish_receipt.json'))
    markers = ('fno_coverage100_', 'fno_coverage_readout_', 'fno_coverage_v4_', 'operator_hash_', 'github_gitdata_5a8d')
    for file in sorted((ROOT / 'notes').glob('*.json')):
        if any(m in file.name for m in markers):
            path = file.relative_to(ROOT).as_posix()
            if path not in publication['allowlist']:
                publication['allowlist'].append(path)
    if prior not in publication['allowlist']:
        publication['allowlist'].append(prior)
    publication['commit_message'] = 'Report FNO CAD coverage results and start whole-case BC continuous Hash comparison'
    write('notes/github_publish_plan.json', publication)
    print(dict(status='local_reports_current', current_FNO_U_S=[best['U_percent'], best['S_percent']], network_calls=0))


if __name__ == '__main__':
    main()
