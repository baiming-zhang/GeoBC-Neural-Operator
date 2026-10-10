"""Accept genuine CPU preparation terminal receipts; metadata/stat only."""
import argparse
import math
import os
import time
from geobc import available_cad_pilot_data as d


def run():
    started=time.monotonic();assert os.name=='posix'
    plan=d.read('notes/geobc_prospective_train_targets_plan.json')
    folder=d.path(plan['output']);result=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json')
    assert result['scientific_digest']==external['scientific_digest']==plan['scientific_digest']==d.digest(plan['scientific'])
    assert all(d.sha(name)==sha for name,sha in plan['source_pins'].items())
    assert result['status']==external['status']=='completed' and external['exit_code']==0 and not external['timeout']
    assert external['closed'] and not external['active'] and external['exact_child_exited']
    assert result['clock']['closed'] and not result['clock']['active']
    for record in (external['child_process'],external['guard_process']):assert d.exact_process_exited(record)
    for value in (result['clock'],external):
        assert all(math.isfinite(value[name]) and 0<=value[name]<=1800 for name in
                   ('elapsed_epoch_seconds','elapsed_monotonic_seconds'))
    assert result['CUDA_initialized'] is False and result['GPU_work']==0 and result['new_model_training']==0
    selected=plan['scientific']['selected_TRAIN60_CADs'];keys=d.keys(selected)
    assert set(result['CADs'])==set(selected) and set(result['records'])==set(keys) and len(keys)==180
    receipt=result['read_receipt']
    assert receipt['H5_vertices_reads']==receipt['CSV_coordinate_and_known3_parse_attempts']==receipt['CSV_coordinate_and_known3_parses']==60
    assert receipt['TRAIN_U_case_reads']==receipt['TRAIN_S_case_reads']==180 and receipt['raw_byte_hashes']==120
    assert all(receipt[name]==0 for name in ('DEV_target_parses','FINAL_target_parses','H5_nodal_variable_reads',
                   'source_strain_reads','dia_target_reads','scaler_fits'))
    checks={};nodes=0
    for key in keys:
        row=result['records'][key]
        assert row['scientific_group']=='TRAIN' and row['status']=='completed' and row['num_nodes']>0
        assert row['label_alignment']=='official CSV GRID coordinates -> H5 vertices'
        assert all(math.isfinite(row[name]) and row[name]>=0 for name in ('displacement_squared_norm','stress_squared_norm'))
        nodes+=row['num_nodes']
        for item in list(row['file_identities'].values())+[row['permutation_identity']]:checks[item['path']]=item
    for item in checks.values():d.stat_matches(item)
    assert len(checks)==420
    max_distance=max(row['CSV_H5_max_distance_mm'] for row in result['CADs'].values())
    assert max_distance<=.0002
    output=d.path('results/geobc_prospective_train_preparation');assert not output.exists();output.mkdir()
    evidence={name:d.identity(folder/file,True) for name,file in
              [('summary','summary.json'),('external','external_terminal.json'),('clock','clock.json'),('execution','execution.json')]}
    summary={'status':'completed_target_only','scientific_digest':plan['scientific_digest'],
        'CADs':60,'cases':180,'original_node_rows_across_cases':nodes,'owned_array_stat_checks':len(checks),
        'max_CSV_H5_coordinate_distance_mm':max_distance,'epoch_seconds':external['elapsed_epoch_seconds'],
        'monotonic_seconds':external['elapsed_monotonic_seconds'],'read_receipt':receipt,'terminal_evidence':evidence,
        'actual_CPU_acceptance_seconds':time.monotonic()-started,'new_input_grids':0,'new_model_training':0,
        'FINAL_target_reads':0,'scaler_fits':0,'actual_new_GPU_work':0,
        'limitations':['Metadata/stat acceptance, not an independent reread of target array bodies',
                       'New80 geometry/feature/grid preparation and new100CAD solver experiments remain incomplete',
                       'CPU IO overlapped the existing GPU job; runtime is not an isolated performance comparison']}
    d.write(output/'manifest.json',summary)
    d.write(output/'source_cache_index.json',{'schema':'TRAIN_target_only_index_v1','scientific_digest':plan['scientific_digest'],
                                            'records':result['records'],'scaler':result['scaler']})
    text=f"# 新划分：TRAIN目标准备\n\n新增60 CAD×三载荷、180例原节点目标已准备。CPU整体耗时{summary['epoch_seconds']:.2f}秒；没有新增GPU任务。\n\nCSV GRID坐标对齐H5 vertices，最大距离{max_distance:.8g}毫米。保留原signed U(mm)/S(MPa)，继承旧TRAIN40-only scaler，不重新拟合。\n\nDEV／FINAL标签读取均为0。新几何／25特征／R64输入尚未准备，新模型训练为0；不能把目标准备完成当作求解器结果或完整数据输入完成。CPU I/O与当前GPU训练重叠，耗时并非隔离性能测试。\n"
    (output/'BRIEF.md').write_text(text,encoding='utf-8')
    review=d.read(plan['root_review']);review.update(actual_preparation_executions=1,CPU_preparation_released=False,
        actual_completed_terminal_evidence=evidence,actual_target_only_result=d.identity(output/'manifest.json',True))
    d.write(plan['root_review'],review)
    print({name:summary[name] for name in ('status','CADs','cases','epoch_seconds','monotonic_seconds','FINAL_target_reads','new_input_grids')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--accept',action='store_true')
    if parser.parse_args().accept:run()
