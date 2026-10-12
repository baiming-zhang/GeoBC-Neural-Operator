"""One bounded CPU diagnosis of immutable knownDEV stress predictions."""
import time
ENTRY_EPOCH, ENTRY_MONOTONIC = time.time(), time.monotonic()
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


def child(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.signed_stress_diagnostic_v84 import approved, decompose, summarize, check_identity
    p, review, up = approved(plan_path); science = p['scientific']; spec = science['design']
    outer = json.loads(os.environ['GEOBC_SIGNED_STRESS_V84_GUARD']); output = d.path(p['output'])
    assert outer['scientific_digest'] == p['scientific_digest']
    assert outer['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert d.process_snapshot(outer['guard_process']['pid']) == outer['guard_process']
    assert not (output/'execution.json').exists() and not (output/'summary.json').exists()
    execution = dict(run_process=d.process_snapshot(os.getpid()), guard=outer, boot_id=outer['boot_id'], scientific_digest=p['scientific_digest'])
    assert execution['run_process']['ppid'] == outer['guard_process']['pid']
    d.write(output/'execution.json', execution)
    a = dict(status='running', active=True, closed=False, phase='prediction_identity_verification', execution=execution,
        owned_GT_array_read_attempts=0, owned_GT_array_reads=0, GT_body_hash_reads=0, prediction_body_hash_reads=0,
        prediction_array_reads=0, raw_CSV_H5_reads=0, PT_reads=0, FINAL_reads=0, scaler_fits=0, GPU_calls=0,
        inference_calls=0, training_updates=0, oracle_field_files_saved=0, automatic_replay=False, per_case={},
        peak_RSS_bytes=0, protocol_qualified=False, final_solver_selected=False)
    lock = None
    def guard():
        if min(outer['deadline_epoch']-time.time(), outer['deadline_monotonic']-time.monotonic()) <= 0:
            raise TimeoutError('Frozen earlier UTC/monotonic120-second CPU deadline')
    def persist():
        a.update(elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'status.json', a)
    signal.signal(signal.SIGTERM, lambda *args: (_ for _ in ()).throw(TimeoutError('External guard terminated child')))
    try:
        import fcntl
        lock = d.path(science['GPU_lock']).open('a+'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import resource
        names, keys = spec['candidate_names'], spec['case_keys']; records = {}
        # The upstream true terminal barrier already proved finite arrays. Verify
        # immutable bodies for all240 before this derivative numerical-GT pass.
        for name in names:
            a['per_case'][name] = {}
            for key in keys:
                guard(); record = d.read(check_identity(up['summary']['prediction_inventory'][name+':'+key]))
                assert record['candidate'] == name and record['key'] == key and record['signed_stress_unchanged']
                assert record['step'] == 6000 and record['status'] == 'completed'
                row = up['summary']['candidates'][name]
                assert record['package'] == row['package'] and record['model_seed'] == row['model_seed'] and record['checkpoint'] == row['checkpoint']
                check_identity(record['physical']); a['prediction_body_hash_reads'] += 1
                records[name+':'+key] = record
        assert len(records) == a['prediction_body_hash_reads'] == 240 and a['owned_GT_array_reads'] == 0
        a['all_prediction_identities_verified'] = True; a['phase'] = 'shared_cached_GT_once'; persist()
        for key in keys:
            guard(); gt = up['truth']['records'][key]['physical']; a['owned_GT_array_read_attempts'] += 1
            check_identity(gt); a['GT_body_hash_reads'] += 1
            truth = np.load(d.path(gt['path']), mmap_mode='r', allow_pickle=False); a['owned_GT_array_reads'] += 1
            assert truth.ndim == 2 and truth.shape[1] == 4 and str(truth.dtype) == 'float32' and np.isfinite(truth).all()
            for name in names:
                guard(); rec = records[name+':'+key]; d.stat_matches(rec['physical'])
                pred = np.load(d.path(rec['physical']['path']), mmap_mode='r', allow_pickle=False); a['prediction_array_reads'] += 1
                assert pred.shape == truth.shape and str(pred.dtype) == 'float32' and np.isfinite(pred).all()
                z = decompose(np, truth[:, 3], pred[:, 3]); old = up['summary']['candidates'][name]['full60']['per_case'][key]
                assert math.isclose(z['actual_signed_S_relative_L2'], old['stress_relative_l2'], rel_tol=1e-10, abs_tol=1e-10)
                assert math.isclose(z['actual_hotspot_relative_L2'], old['stress_hotspot_relative_l2'], rel_tol=1e-10, abs_tol=1e-10)
                a['per_case'][name][key] = z; del pred
                a['peak_RSS_bytes'] = max(a['peak_RSS_bytes'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            del truth; persist()
        assert a['owned_GT_array_read_attempts'] == a['owned_GT_array_reads'] == a['GT_body_hash_reads'] == 60
        assert a['prediction_array_reads'] == a['prediction_body_hash_reads'] == 240
        a['arms'] = {name: summarize(a['per_case'][name], 60) for name in names}
        assert all(v['complete_primary_metrics'] for v in a['arms'].values())
        a['per_load20'] = {name: {load: summarize({k: a['per_case'][name][k] for k in keys if k.endswith('/'+load)}, 20)
            for load in ('ver', 'hor', 'tor')} for name in names}
        packages = ('FNO9_plain_query_MLP', 'FNO9_TCNN_query'); package_means = {}
        for package in packages:
            matched = [x['name'] for x in up['plan']['scientific']['candidates'] if x['package'] == package]
            assert len(matched) == 2
            fields = a['arms'][matched[0]]['equal_case_means']
            package_means[package] = {field: sum(a['arms'][n]['equal_case_means'][field] for n in matched)/2
                if all(a['arms'][n]['equal_case_means'][field] is not None for n in matched) else None for field in fields}
        a['package_two_new_seed_means'] = package_means
        a['paired_CAD20'] = {}
        for seed in (2027, 2028):
            plain, hashed = 'FNO20_seed'+str(seed), 'Hash20_seed'+str(seed)
            a['paired_CAD20'][str(seed)] = {cad: {
                field: sum(a['per_case'][hashed][cad+'/'+load][field]-a['per_case'][plain][cad+'/'+load][field] for load in ('ver', 'hor', 'tor'))/3
                for field in ('actual_signed_S_relative_L2', 'sign_oracle_relative_L2', 'magnitude_oracle_relative_L2')}
                for cad in sorted({k.split('/')[0] for k in keys})}
        a['phase'] = 'CPU_report'; persist(); guard()
        out = d.path('results/geobc_signed_stress_bottleneck_v84'); assert not (out/'manifest.json').exists()
        out.mkdir(parents=True, exist_ok=True)
        labels = {packages[0]: 'FNO', packages[1]: 'FNO + 条件Hash'}
        lines = ['# 应力误差：符号与空间幅值', '', '**两个新增配对seed／固定20个已迭代开发CAD×三载荷。下面两种oracle使用真值，仅诊断，不能部署或当实际solver改善。**', '',
            '| 两新增seed平均 | 实际signed应力误差 | 真值符号oracle误差 | 真值幅值oracle误差 | 精确符号惩罚占SSE |',
            '|---|---:|---:|---:|---:|']
        def percent(value): return 'undefined' if value is None else f'{100*value:.2f}%'
        for package in packages:
            m = package_means[package]
            fields = ('actual_signed_S_relative_L2', 'sign_oracle_relative_L2', 'magnitude_oracle_relative_L2', 'exact_sign_penalty_share_of_signed_SSE')
            lines.append('| '+labels[package]+' | '+' | '.join(percent(m[f]) for f in fields)+' |')
        lines += ['', '精确分解：signed SSE = 幅值差平方和 + 异号非零节点的4×|预测×真值|之和。它把幅值误差与额外符号惩罚分开；异号节点上的全部SSE还含幅值误差，二者不能混称。真值为0时符号oracle会消除预测假阳性，零值贡献单独列在JSON。', '',
            '所有原signed标签不改；oracle为abs(预测)×sign(真值)与abs(真值)×sign(预测)，0按0。热点沿用真值绝对应力最高1%节点。零附近按0.01×真值RMS预声明，未定义范数／相关保持null，不挑子集均值。', '',
            '读取已有physicalGT60一次、240份既有physical预测一次；字节SHA验证另记60／240，原CSV／H5／PT／FINAL／scaler／GPU／重推理／训练0。固定案例、每载荷、CAD配对、相关／范数／正负能量与全部成本见JSON／CSV。两seed共20CAD，不是独立最终盲测；2026没有重新读数组，不能冒充三seed诊断。']
        (out/'BRIEF.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file); fields = list(next(iter(a['per_case'][names[0]].values())))
            writer.writerow(['arm', 'case']+fields)
            for name in names:
                for key in keys: writer.writerow([name, key]+[a['per_case'][name][key][f] for f in fields])
        os.environ['MPLCONFIGDIR'] = str(output/'.mplconfig')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 3.5)); fields = ('actual_signed_S_relative_L2', 'sign_oracle_relative_L2', 'magnitude_oracle_relative_L2')
        for offset, package, color in zip((-.18, .18), packages, ('#356fbd', '#cd7435')):
            ax.bar([i+offset for i in range(3)], [100*package_means[package][f] for f in fields], .36,
                color=color, label='FNO' if package == packages[0] else 'FNO + TCNN')
        ax.set_xticks(range(3), ('Actual signed S', 'True-sign oracle', 'True-magnitude oracle'))
        ax.set_ylabel('Case-mean relative L2 (%)'); ax.set_title('KnownDEV / two paired seeds; truth-conditioned oracles are not deployable')
        ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)
        guard(); a.update(status='completed', phase='terminal', protocol_qualified=True, cached_prediction_reference_matched=True,
            exact_SSE_identity_all240=True, oracle_is_deployable=False, result_output=d.relative(out))
    except BaseException as error:
        a.update(status='failed', phase='terminal_failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        a.update(active=False, closed=True, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC)
        d.write(output/'summary.json', a); d.write(output/'status.json', a)
        if a['status'] == 'completed':
            d.write(out/'metrics.json', a)
            d.write(out/'manifest.json', dict(status='completed', SCI=p['scientific_digest'], source_pins=p['source_pins'],
                upstream_binding=up['binding'], actual_CPU_epoch_seconds=a['elapsed_epoch_seconds'],
                actual_CPU_monotonic_seconds=a['elapsed_monotonic_seconds'], external_exit_acceptance_pending=True,
                files=['BRIEF.md', 'per_case.csv', 'metrics.json', 'comparison.png'], GT_numeric_reads=60, prediction_array_reads=240, GT_body_hash_reads=60,
                prediction_body_hash_reads=240, raw_PT_FINAL_GPU_inference_training=0, oracle_deployable=False))
        if lock is not None: lock.close()


def run(plan_path):
    from geobc import available_cad_pilot_data as d
    from geobc.signed_stress_diagnostic_v84 import approved
    p, r, up = approved(plan_path); output = d.path(p['output']); output.mkdir(parents=True, exist_ok=False)
    budget = p['scientific']['design']['CPU_budget_seconds']; assert budget == 120
    guard = dict(guard_process=d.process_snapshot(os.getpid()), scientific_digest=p['scientific_digest'],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), entry_epoch=ENTRY_EPOCH, entry_monotonic=ENTRY_MONOTONIC,
        deadline_epoch=ENTRY_EPOCH+budget, deadline_monotonic=ENTRY_MONOTONIC+budget)
    receipt = dict(**guard, status='starting', active=True, closed=False, attempts=1, automatic_replay=False,
        actual_child_exited=False, source_pins=p['source_pins'], plan_path=plan_path)
    launch = output/'external_launch.json'; d.write(launch, receipt)
    env = os.environ.copy(); env['GEOBC_SIGNED_STRESS_V84_GUARD'] = json.dumps(guard)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): env[name] = '4'
    process = None; timeout = False; code = None; error_kind = None
    def stop():
        if process is None or process.poll() is not None: return
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=5)
    try:
        with (output/'process.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'tools.run_geobc_signed_stress_v84', '--child', '--plan', plan_path],
                cwd=d.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = None
            for _ in range(50):
                record = d.process_snapshot(process.pid)
                if record or process.poll() is not None: break
                time.sleep(.01)
            assert record and record['ppid'] == guard['guard_process']['pid']
            receipt.update(status='running', child_process=record); d.write(launch, receipt)
            while process.poll() is None:
                remaining = min(guard['deadline_epoch']-time.time(), guard['deadline_monotonic']-time.monotonic())
                if remaining <= 0: timeout = True; stop(); break
                try: process.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired: pass
            code = process.returncode
    except BaseException as error:
        error_kind = type(error).__name__; raise
    finally:
        stop()
        if process is not None: code = process.returncode
        receipt.update(status='terminal', active=False, closed=True, exit_code=code, external_timeout=timeout,
            guard_error_type=error_kind, elapsed_epoch_seconds=time.time()-ENTRY_EPOCH, elapsed_monotonic_seconds=time.monotonic()-ENTRY_MONOTONIC,
            actual_child_exited=('child_process' in receipt and d.exact_process_exited(receipt['child_process'])))
        receipt['within_external_budget'] = code == 0 and not timeout and error_kind is None and receipt['actual_child_exited'] and max(receipt['elapsed_epoch_seconds'], receipt['elapsed_monotonic_seconds']) <= budget
        d.write(output/'external_terminal.json', receipt); d.write(launch, receipt)
    print({k: receipt[k] for k in ('status', 'exit_code', 'within_external_budget')})
    return 0 if receipt['within_external_budget'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group()
    group.add_argument('--start', action='store_true'); group.add_argument('--child', action='store_true'); parser.add_argument('--plan')
    args = parser.parse_args()
    if args.start or args.child:
        if not args.plan: parser.error('Actual execution requires a separately released --plan')
        if args.child: child(args.plan)
        else: raise SystemExit(run(args.plan))
    else:
        print('{"status":"source_only","GT_PT_arrays_GPU":0}')
