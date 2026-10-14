"""Reuse saved scalar results only; historical configurations are not matched."""
import argparse,csv,json
from pathlib import Path


def run(root):
    out=Path('results/sample_overfitting');out.mkdir(parents=True,exist_ok=True)
    names=['mlp_single_exact','ngp4_506_533_ver_u','ngp4_506_533_ver_stress','hash_single_exact','tcnn_hash_single_exact','transolver_single','transolverpp_single_asinh','ts3_single','fno64_joint','fno64_stress','fno192_stress','fno192_hash_distill','hash_five_aligned','tcnn_hash_five_aligned','transolver_five','transolverpp_five','ts3_five']
    rows=[];records={}
    for name in names:
        path=root/'runs/slides_reproduction'/name/'summary.json'
        if not path.exists():continue
        record=json.loads(path.read_text());m=record['best_metrics'];records[name]=m
        if 'nodes' in m and isinstance(m['nodes'],dict):
            pooled=m['nodes'] if 'relative_l2' not in m['nodes'] else {'stress':m['nodes']}
            per={'506_533/ver':pooled}
        else:per=m['per_case'];pooled=m['pooled']
        u=[v['displacement']['relative_l2']*100 for v in per.values() if 'displacement' in v]
        s=[v['stress']['relative_l2']*100 for v in per.values() if 'stress' in v]
        def field(kind):return pooled[kind]['relative_l2']*100 if kind in pooled else None
        rows.append({'experiment':name,'samples':len(per),'updates':record['step'],'selected_step':record['best_step'],
                     'U_pooled_percent':field('displacement'),'S_pooled_percent':field('stress'),
                     'U_mean_percent':sum(u)/len(u) if u else None,'S_mean_percent':sum(s)/len(s) if s else None,
                     'U_worst_percent':max(u) if u else None,'S_worst_percent':max(s) if s else None,
                     'both_fields_all_cases_below_1_percent':bool(len(u)==len(per) and len(s)==len(per) and max(u+s)<1),
                     'recorded_seconds':record['elapsed_seconds']})
    with (out/'historical.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'historical_metrics.json').write_text(json.dumps({'experiments':rows,'selected_checkpoint_metrics':records,'new_GT_reads':0,'new_training_updates':0},indent=2)+'\n')
    table=['|历史实验|样本数|U均值 / 最差 %|S均值 / 最差 %|实际训练更新|保存耗时|','|---|---:|---:|---:|---:|---:|']
    def fmt(mean,worst):return '—' if mean is None else f'{mean:.4f} / {worst:.4f}'
    for r in rows:table.append(f"|{r['experiment']}|{r['samples']}|{fmt(r['U_mean_percent'],r['U_worst_percent'])}|{fmt(r['S_mean_percent'],r['S_worst_percent'])}|{r['updates']}|{r['recorded_seconds']:.2f}s|")
    text='''# 单样本与五样本过拟合

当前范围仅训练样本拟合，目标是**同一个 checkpoint 下，每个样本的原始全节点位移 U3 与原始有符号应力 S 相对 L2 都严格低于1%**。五样本以最差样本、最差字段验收，不用 pooled 或平均误差替代。

单样本：506_533/ver。五样本：506_533/ver、506_25/tor、10_30/ver、10_421/hor、10_460/dia。“样本”指 CAD×载荷，不是5个CAD的全部载荷。

## 已有结果

以下是已保存的历史 checkpoint 指标，未重训或重新读取标签。容量、优化器、训练预算、输出变换和 checkpoint 选择规则不同，因此不是统一配方排名。缺失 U/S 写 —，不是0或达标。NGP 的 U-only 与 S-only 是不同模型，不能组合成同一联合模型达标。FNO 只列 FEM 原节点误差；网格/教师拟合误差另属不同指标。

'''+ '\n'.join(table)+'''

历史中尚无表内方法证明单样本或五样本联合 U/S 均达到1%。单样本 Hash 的 S-only 已低于1%；五样本 TCNN Hash 的 S 均值约1.45%、最差2.11%。Transolver 系列历史预算内远未拟合好。MLP、NGP、FNO 的五样本联合结果缺失，保留为空，不用跨CAD结果填充。

## 新的统一目标试验

[配置](../../configs/sample_overfitting.json)固定8种实现：MLP、4-level NGP、12-level Torch Hash、官方 TCNN Hash、带原节点查询头的 FNO、原版 Transolver、Transolver++、Transolver-3。每种分别 fresh 单样本和五样本；seed2026，不旧 checkpoint 暖启、不残差教师、不加入物理输入或样本ID。每次更新完整一个原节点样本，五样本循环；所有方法联合输出 U3/S，训练目标为两个字段物理相对 L2平方的等权均值。仅这些拟合样本确定输出尺度，和原跨CAD实验的 TRAIN40 scaler 区分。学习率从.001开始，训练集最差字段指标停滞时统一减半到1e-5。

FNO 新使用 R64 输入表示、native XYZ/BC 查询头，并在原始节点上监督；区别于旧网格输出再插值的 FNO。Transformer 全部原节点参与每层全局上下文，不以独立小块更换模型。TS++保留官方随机 Gumbel，单次训练集读出达标不表示重复读出都满足1%。普通模型 FP32、TF32加速；TCNN按官方 FP16/FP32 master。参数量和成本分别记录，不声称等容量排名。

每臂固定600秒（其中末60秒预留读出/保存）、最多20000次更新；全阶段10800秒上限含导入、缓存、训练、读出、保存和退出。每100步完整读出 U/S、保留同一个最差字段最优 checkpoint；两字段所有样本低于1%提前结束，否则诚实报告预算内未达标。固定预算是上限，不是完成时间估计。真实模型异常保留已完成步骤与 Adam/RNG/顺序/游标，未自动重放。数值目标与实现完成分别判断。

新运行结果保留在本地 runs/sample_overfitting；完成后在此目录添加标注配置和逐样本误差的新结果，历史表保持独立。这是训练集记忆能力诊断，不是泛化、独立测试或工程精度证明。
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);run(p.parse_args().project_root)
