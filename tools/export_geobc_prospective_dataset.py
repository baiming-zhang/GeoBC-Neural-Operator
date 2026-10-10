"""CPU scalar acceptance/report for assembled420 records and actual TRAIN loader probe."""
import argparse
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding


def export():
    began=time.monotonic();assembly=d.read('notes/geobc_prospective_dataset_plan.json')
    probe=d.read('notes/geobc_prospective_loader_recovery_plan.json');original=d.read('notes/geobc_prospective_loader_probe_plan.json')
    assert probe['scientific_digest']==original['scientific_digest']==d.digest(probe['scientific']) and not probe['budget_reset']
    for plan in (assembly,probe):assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    evidence={};ends={}
    for name,plan in [('assembly',assembly),('probe_recovery',probe)]:
        folder=d.path(plan['output']);binding={k:d.identity(folder/f,hashed=True) for k,f in [('summary','summary.json'),('external','external_terminal.json')]}
        final=terminal_binding(binding);external=d.read(binding['external']['path'])
        assert final['execution']['scientific_digest']==external['scientific_digest']==plan['scientific_digest']
        limit=180 if name=='assembly' else 300
        assert max(external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds'])<=limit
        ends[name]=(final,external);evidence.update({name+'_'+key:value for key,value in binding.items()})
    assembled,assembly_ext=ends['assembly'];tested,recovery_ext=ends['probe_recovery']
    assert assembled['total_case_records']==420 and assembled['TRAIN_target_records']==300 and assembled['protected_input_only_records']==120
    assert assembled['array_GT_model_GPU_reads']==assembled['scaler_fits']==0
    assert tested['qualified'] and not tested['torch_CUDA_initialized']
    assert tested['actual_TRAIN_array_reads']==6 and tested['actual_input_array_reads']==21
    assert tested['DEV_FINAL_target_array_reads']==tested['source_CSV_H5_reads']==tested['scaler_fits']==tested['model_GPU_work']==0
    assert all(tested['checks'][name] for name in ('all120_protected_requests_rejected_before_array_access','all180_newTRAIN_rejected_under40scope','all420_input_records_resolve'))
    failed=d.read(probe['recovery_original_failure']['path']);failed_ext=d.read(probe['recovery_original_external']['path'])
    assert d.identity(probe['recovery_original_failure']['path'],hashed=True)==probe['recovery_original_failure']
    assert d.identity(probe['recovery_original_external']['path'],hashed=True)==probe['recovery_original_external']
    assert failed['status']=='failed' and failed['actual_TRAIN_array_reads']==0 and failed['actual_input_array_reads']==6
    assert failed_ext['closed'] and not failed_ext['active'] and all(d.exact_process_exited(failed_ext[k]) for k in ('guard_process','child_process'))
    assert failed_ext['deadline_epoch']==recovery_ext['deadline_epoch'] and failed_ext['deadline_monotonic']==recovery_ext['deadline_monotonic']
    evidence.update(original_failed_summary=probe['recovery_original_failure'],original_failed_external=probe['recovery_original_external'],
        immutable_input_manifest=assembled['manifest_identity'])
    output=d.path('results/geobc_prospective_dataset_interface');assert not (output/'manifest.json').exists(),'Prior export preserved'
    output.mkdir(parents=True,exist_ok=True)
    for name,pin in evidence.items():d.write(output/'source_evidence'/(name+'.json'),d.read(pin['path']))
    manifest=d.read(assembled['manifest_identity']['path']);assert d.digest(manifest['records'])==manifest['records_digest']
    d.write(output/'dataset_index.json',manifest)
    result=dict(status='completed',qualified=True,TRAIN_CADs=100,knownDEV_CADs=20,reservedFINAL_CADs=20,
        case_records=420,TRAIN_target_records=300,protected_input_only_records=120,scaler_fits=0,
        assembly_native_epoch_seconds=assembled['elapsed_epoch_seconds'],assembly_native_monotonic_seconds=assembled['elapsed_monotonic_seconds'],
        assembly_outer_epoch_seconds=assembly_ext['elapsed_epoch_seconds'],
        probe_original_budget_seconds=300,probe_total_original_epoch_span=recovery_ext['elapsed_epoch_seconds'],
        probe_total_original_monotonic_span=recovery_ext['elapsed_monotonic_seconds'],
        first_failed_attempt_epoch_seconds=failed_ext['elapsed_epoch_seconds'],recovery_attempt_epoch_seconds=recovery_ext['attempt_epoch_seconds'],
        probe_numerical_TRAIN_array_reads_total=6,probe_input_array_reads_including_failed_attempt=27,
        DEV_FINAL_target_array_reads=0,source_CSV_H5_reads=0,GPU_work=0,all300_numerical_bodies_checked=False,
        old_new_TRAIN_case_checks=tested['case_checks'],access_checks=tested['checks'],budget_reset=False,
        corrected_loader_module='geobc.prospective_dataset_v2',new_training_started=False,FINAL_target_readout_released=False,
        source_evidence=evidence,CPU_export_array_GT_PT_model_GPU_reads=0,CPU_export_seconds=time.monotonic()-began)
    text='# 公共数据接口：100／20／20 CAD\n\n'
    text+=f"420个病例索引已整合：TRAIN300携带已有训练目标，knownDEV60／reservedFINAL60仅携带输入。元数据整合UTC{assembled['elapsed_epoch_seconds']:.2f}秒／mono{assembled['elapsed_monotonic_seconds']:.2f}秒，未读取数组体或拟合scaler。\n\n"
    text+='实际CPU读取一个旧TRAIN和一个新TRAIN病例：旧例九项输入／目标与原加载器完全一致，两例缓存encoded目标与原physical4重新编码完全一致。120个DEV／FINAL目标请求及180个不属于TRAIN40的目标请求均在数组／库访问前拒绝；全部420个输入记录可解析。\n\n'
    text+=f"首次检查因旧索引int32与新索引int64兼容问题退出，只读取6个输入数组、训练目标0。修复后的v2加载器保留原dtype／数值，恢复仍使用原300秒双轴截止；含失败和修正的实际UTC跨度{recovery_ext['elapsed_epoch_seconds']:.2f}秒，预算未重置。历史失败与源码保留。\n\n"
    text+='后续实验使用`geobc.prospective_dataset_v2.load_case`及显式40／100TRAIN范围，不能使用旧200别名。沿用TRAIN40-only scaler；FINAL标签仍未读取，新100CAD训练未启动。此检查覆盖两个实际TRAIN病例的兼容性和所有保护请求，未重新验证全部300个目标数组体，也没有运行模型或证明求解器精度。\n'
    (output/'BRIEF.md').write_text(text,encoding='utf-8');d.write(output/'metrics.json',result)
    d.write(output/'manifest.json',dict(status='completed',actual_export=True,qualified=True,scientific_digest=assembly['scientific_digest'],
        probe_scientific_digest=probe['scientific_digest'],exporter_source_sha256=d.sha('tools/export_geobc_prospective_dataset.py'),
        artifacts={name:d.identity(output/name,hashed=True) for name in ('BRIEF.md','metrics.json','dataset_index.json')},
        CPU_export_seconds=result['CPU_export_seconds'],exporter_array_GT_PT_model_GPU_reads=0))
    print({'status':'completed','CPU_export_seconds':result['CPU_export_seconds'],'probe_budget_reset':False,'FINAL_target_reads':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:export()
