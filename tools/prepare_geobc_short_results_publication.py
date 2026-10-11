"""Local documentation and allowlist only; no credentials or network."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    prior = 'notes/github_publish_prior_7d8d_canonical_receipt.json'
    if not (ROOT / prior).exists():
        write(prior, read('notes/github_publish_receipt.json'))
    plan = read('notes/github_publish_plan.json')
    markers = ('macro_screen_', 'macro_warm_', 'two_warm_', 'three_short_',
               'short_screen_handoff', 'warm40_actual', 'fno_coverage100_',
               'fno_coverage_readout_', 'fno_coverage_v4_', 'operator_hash_')
    for file in sorted((ROOT / 'notes').glob('geobc_*.json')):
        if any(marker in file.name for marker in markers):
            relative = file.relative_to(ROOT).as_posix()
            if relative not in plan['allowlist']:
                plan['allowlist'].append(relative)
    if prior not in plan['allowlist']:
        plan['allowlist'].append(prior)
    plan['commit_message'] = 'Report actual short solver screens and warm40 comparison; start FNO CAD coverage check'
    write('notes/github_publish_plan.json', plan)
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    text = text.replace('新科学计划和源码尚需冻结放行，当前新100训练0。',
                        '独立v4科学计划和源码已冻结并通过真实上游终态放行，100CAD任务已于北京时间09:46启动，exec99080；4800及12000检查点，训练最多30分钟、整体40分钟，实际结果待完成。')
    text = text.replace('也未启动新100CAD训练。', '100CAD实际载入与最大全节点梯度正在本次预算内验证。')
    text = text.replace('当前运行实验保留自身较小的原24GiB上限。', '历史实验保留其原24GiB上限；当前v4采用用户30.5GB预算。')
    start = text.index('[私人仓库 GeoBC-Neural-Operator]')
    end = text.index('\n\n更早逐阶段', start)
    paragraph = ('[私人仓库 GeoBC-Neural-Operator](https://github.com/baiming-zhang/GeoBC-Neural-Operator)；'
                 '最近已验收提交 `7d8d556` 包含同12000更新完整开发60例比较、原24k主动终止和三候选10遍短筛代码／计划。'
                 '新10遍全部终态、warm40确认及100CAD新阶段现已在本地整理，待本次具体快照发布；不将本地文件冒充远端。'
                 '上一发布50.63秒，精确commit和private状态均已核验。[发布计划](notes/github_publish_plan.json)。')
    (ROOT / 'README.md').write_text(text[:start] + paragraph + text[end:], encoding='utf-8')
    exploration = read('notes/geobc_exploration_plan.json')
    exploration['coverage100_v4'] = dict(status='actual_launched',
        plan='notes/geobc_fno_coverage100_v4_plan.json',
        actual_launch='notes/geobc_fno_coverage100_v4_actual_launch.json',
        exec_session=99080, scientific_digest=read('notes/geobc_fno_coverage100_v4_plan.json')['scientific_digest'],
        training_budget_seconds=1800, whole_budget_seconds=2400, not_a_completed_result=True)
    write('notes/geobc_exploration_plan.json', exploration)
    print(json.dumps(dict(publication_allowlist=len(plan['allowlist']), readme_current=True,
                          prior_success_preserved=True)))


if __name__ == '__main__':
    main()
