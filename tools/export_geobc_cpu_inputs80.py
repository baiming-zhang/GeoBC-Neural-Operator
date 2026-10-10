"""Accept complete fixed80 CPU input preparation from metadata, without opening arrays."""
import argparse
import ast
import time
from geobc import available_cad_pilot_data as d
from tools.train_geobc_single_case_diagnostic import terminal_binding

OWN='tools/export_geobc_cpu_inputs80.py'
REVIEW='notes/geobc_cpu_inputs80_exporter_root_review.json'
OUTPUT='results/geobc_prospective_input_preparation'


def freeze():
    assert not d.path(REVIEW).exists()
    ast.parse(d.path(OWN).read_text(encoding='utf-8'))
    plan=d.read('notes/geobc_cpu_inputs80_plan.json')
    d.write(REVIEW,dict(passed=False,whole_SOURCE_reviewed=False,actual_export_released=False,
        exporter_source_sha256=d.sha(OWN),scientific_digest=plan['scientific_digest'],actual_export=0))
    print({'status':'source_frozen_actual_export0','sha256':d.sha(OWN)})


def export():
    began=time.monotonic();review=d.read(REVIEW);plan=d.read('notes/geobc_cpu_inputs80_plan.json')
    assert review['passed'] and review['whole_SOURCE_reviewed'] and review['actual_export_released']
    assert review['exporter_source_sha256']==d.sha(OWN)
    assert review['scientific_digest']==plan['scientific_digest']==d.digest(plan['scientific'])
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    probe=terminal_binding(plan['scientific']['probe_binding'])
    assert probe['qualified'] and not probe['torch_CUDA_initialized']
    folder=d.path(plan['output']);binding={k:d.identity(folder/f,hashed=True) for k,f in
        [('summary','summary.json'),('external','external_terminal.json')]}
    summary=terminal_binding(binding);external=d.read(binding['external']['path']);manifest=d.read(folder/'manifest.json')
    contract=plan['scientific']['helper_contract']
    assert summary['execution']['scientific_digest']==external['scientific_digest']==manifest['scientific_digest']==plan['scientific_digest']
    assert summary['all_fixed80'] and summary['prepared_CADs']==80 and summary['prepared_cases']==240
    assert summary['numerical_GT_reads']==summary['scaler_fits']==summary['GPU_work']==0 and not summary['torch_CUDA_initialized']
    assert all(summary['read_receipt'][name]==0 for name in d.ZERO_COUNTERS)
    assert max(external['elapsed_epoch_seconds'],external['elapsed_monotonic_seconds'])<=7200
    assert manifest['status']=='completed' and manifest['feasibility_pass'] and manifest['all_fixed80']
    assert manifest['contract']==contract and manifest['scaler']==contract['scaler'] and not manifest['train_keys']
    assert set(manifest['records'])==set(contract['case_keys']) and len(manifest['records'])==240
    assert set(manifest['CADs'])==set(contract['selected_CADs']) and len(manifest['CADs'])==80
    roles=contract['scientific_CAD_roles'];train=manifest['TRAIN_input_case_keys'];final=manifest['FINAL_inference_input_case_keys']
    assert len(train)==180 and len(final)==60 and not set(train)&set(final) and set(train+final)==set(contract['case_keys'])
    for cad,item in manifest['CADs'].items():
        assert item['status']=='completed' and item['feasibility_pass']
        if roles[cad]=='TRAIN_input_only':assert item['matches_owned_TRAIN_alignment']
        else:assert roles[cad]=='FINAL_inference_input_only'
    for key,record in manifest['records'].items():
        d._record_is_target_free(record,contract)
        assert record['scientific_role']==roles[key.split('/')[0]]
    output=d.path(OUTPUT);assert not (output/'manifest.json').exists(),'Prior export preserved'
    output.mkdir(parents=True,exist_ok=True)
    evidence={**binding,'manifest':d.identity(folder/'manifest.json',hashed=True),
        'read_receipt':d.identity(folder/'read_receipt.json',hashed=True),
        'plan':d.identity('notes/geobc_cpu_inputs80_plan.json',hashed=True),
        **{'CPU_probe_'+key:pin for key,pin in plan['scientific']['probe_binding'].items()}}
    for name,pin in evidence.items():d.write(output/'source_evidence'/(name+'.json'),d.read(pin['path']))
    d.write(output/'source_cache_index.json',dict(schema='prospective80_CPU_target_free_inputs_v1',
        scientific_digest=plan['scientific_digest'],records=manifest['records'],scaler=manifest['scaler'],
        TRAIN_input_case_keys=train,FINAL_inference_input_case_keys=final,train_keys={},numerical_GT_members=0,
        inference_input_preparation_does_not_release_FINAL_targets=True))
    brief='# 前瞻划分：输入准备已完成\n\n'
    brief+=f"固定80个CAD／240载荷病例全部完成，CPU实际UTC{summary['elapsed_epoch_seconds']:.2f}秒／mono{summary['elapsed_monotonic_seconds']:.2f}秒；外部完整阶段UTC{external['elapsed_epoch_seconds']:.2f}秒。\n\n"
    brief+='新增TRAIN60的180例输入与已准备训练目标的节点排列SHA一致；reserved FINAL20的60例仅准备几何、材料、载荷与约束。坐标／25特征／PointNet输入云／R64网格和查询权重均复用原准备数学的CPU后端。\n\n'
    brief+='本阶段没有数值GT、模型、GPU或scaler拟合；FINAL标签仍未读取，新100CAD训练尚未启动。一个已知CAD的CPU／GPU核对已通过，不声称所有新CAD的CPU／GPU数组逐位一致。完整输入索引与阶段收据已保存，报告不重复打开数组。\n'
    (output/'BRIEF.md').write_text(brief,encoding='utf-8')
    result=dict(status='completed',all_fixed80=True,prepared_CADs=80,prepared_cases=240,
        TRAIN_input_cases=180,FINAL_input_cases=60,actual_numerical_GT_reads=0,scaler_fits=0,GPU_work=0,
        native_epoch_seconds=summary['elapsed_epoch_seconds'],native_monotonic_seconds=summary['elapsed_monotonic_seconds'],
        outer_epoch_seconds=external['elapsed_epoch_seconds'],outer_monotonic_seconds=external['elapsed_monotonic_seconds'],
        CPU_probe_qualified=True,CPU_probe_native_epoch_seconds=probe['elapsed_epoch_seconds'],
        CPU_probe_max_grid_absolute_difference=max(v['max_absolute_difference'] for k,v in probe['comparisons'].items() if k.endswith('grid_path')),
        source_evidence=evidence,CPU_export_seconds=time.monotonic()-began,exporter_array_GT_PT_model_GPU_reads=0,
        new_TRAIN100_training_started=False,FINAL_labels_released=False)
    d.write(output/'metrics.json',result)
    d.write(output/'manifest.json',dict(status='completed',actual_export=True,scientific_digest=plan['scientific_digest'],
        exporter_source_sha256=d.sha(OWN),artifacts={name:d.identity(output/name,hashed=True) for name in
        ('BRIEF.md','metrics.json','source_cache_index.json')},CPU_seconds=result['CPU_export_seconds'],exporter_array_GT_PT_model_GPU_reads=0))
    print({'status':'completed','native_epoch_seconds':result['native_epoch_seconds'],'CPU_export_seconds':result['CPU_export_seconds']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--export',action='store_true')
    args=parser.parse_args();assert not(args.freeze and args.export)
    if args.freeze:freeze()
    if args.export:export()
