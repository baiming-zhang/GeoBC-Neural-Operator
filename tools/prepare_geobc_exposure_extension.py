"""CPU scalar/metadata freeze for continuing both matched solvers; no training/array IO."""
import argparse
import ast
import time
from geobc import available_cad_pilot_data as d
from geobc import coverage_comparison_data as data
from tools.train_geobc_single_case_diagnostic import terminal_binding
from tools.train_geobc_exposure_extension import schedule


def prepare():
    parent = d.read('notes/geobc_common_grid_train_plan.json')
    binding = {k:d.identity(d.path(parent['output'])/f,hashed=True) for k,f in
               [('summary','summary.json'),('external','external_terminal.json')]}
    baseline = terminal_binding(binding); assert baseline['both4800']
    assert all(d.sha(p)==h for p,h in parent['source_pins'].items())
    loss = d.read('results/geobc_physical_loss_pair/metrics.json')
    assert loss['scientific_qualification'] and loss['final4800_absolute_gain']['S'] < 0
    assert loss['final4800_absolute_gain']['hot'] < 0
    manifest = data.read_bootstrap_TRAIN40('notes/geobc_coverage_comparison_plan.json')
    keys=manifest['train_keys']['40']; order=schedule(keys)
    assert d.digest(order[:4800]) == parent['scientific']['case_order_sha256']
    pins = dict(parent['source_pins'])
    guard_path='tools/run_geobc_exposure_guard.py'
    guard=d.path('tools/run_geobc_common_training_guard.py').read_text(encoding='utf-8')
    guard=guard.replace('42000','100000').replace('geobc_common_grid_train_plan','geobc_exposure_extension_plan')
    guard=guard.replace('actual_training_released','actual_execution_released').replace('GEOBC_COMMON_TRAIN_GUARD','GEOBC_EXPOSURE_GUARD')
    guard=guard.replace('tools.train_geobc_common_grid','tools.train_geobc_exposure_extension')
    guard=guard.replace("    output = d.path(plan['output'])", "    from tools.train_geobc_single_case_diagnostic import terminal_binding\n    parent = terminal_binding(review['parent_binding'])\n    assert parent['both4800'] and parent['execution']['scientific_digest'] == plan['scientific']['parent_digest']\n    output = d.path(plan['output'])")
    d.path(guard_path).write_text(guard,encoding='utf-8')
    for p in ['tools/train_geobc_exposure_extension.py',guard_path,'tools/prepare_geobc_exposure_extension.py',
              'tools/train_geobc_single_case_diagnostic.py']:
        ast.parse(d.path(p).read_text(encoding='utf-8')); pins[p]=d.sha(p)
    names=['xyz','features','cloud','mask','grid','query_indices','query_weights','target']; files={}
    for key in keys:
        for name in names:
            item=manifest['records'][key]['file_identities'][name+'_path'];data.stat_matches(item);files[item['path']]=item
    science=dict(question='Does increasing shared case exposure40 to100/200 improve fitting and development performance for both matched solvers?',
        parent_digest=parent['scientific_digest'], original_TRAIN_plan='notes/geobc_coverage_comparison_plan.json',
        TRAIN_keys=keys, scaler=manifest['scaler'],arms=['FNO9','PP256'],parent_order_digest=d.digest(order[:4800]),
        full_order_digest=d.digest(order), GPU_lock=parent['scientific']['GPU_lock'],
        checkpoints={arm:baseline['arms'][arm]['checkpoint_identity'] for arm in ['FNO9','PP256']},
        fixed_endpoints=[4800,12000,24000],exposures=[40,100,200],additional_updates_per_arm=19200,
        loss='Original encoded4 full-node MSE unchanged',optimizer='Restore Adam .001/fused/noWD and all RNG/cursor from each own step4800 checkpoint',
        cache='All120 TRAIN cases immutable GPU tensors, file-path dedup; actual first-case equality with original loader and wholeN gradient/resource gate',
        cache_metadata=dict(unique_files=len(files),file_GiB=sum(i['bytes'] for i in files.values())/2**30,arrays_read=0),
        cache_setup_budget_seconds=900,training_budget_per_arm_seconds=45000,readout_per_endpoint_seconds=1200,
        whole_budget_seconds=100000,memory_cap_GiB=24,
        training_clock_includes_midpoint_readout_and_checkpoint_IO=True,
        readout='TRAIN120 at12000/24000; separate fixed development60 only after all training ends',
        selection='No earlystop or DEV adjustment. Future all2armsx2endpointsx60 prediction/failure barrier before owned GT60 once',
        cost='Report baseline costs, new additional costs and shared setup separately; cache acceleration is not inference gain',
        controls='Same original120/scaler/architecture/capacity/precision/order/loss/optimizer trajectory; cache changes IO, not input values',
        limits='Single seed/known availability-conditional DEV; no FINAL/source-target/scaler fitting')
    filename='notes/geobc_exposure_extension_plan.json'; assert not d.path(filename).exists()
    plan=dict(scientific=science,scientific_digest=d.digest(science),source_pins=pins,
              output='runs/geobc_exploration/matched_exposure24000_v1',root_review='notes/geobc_exposure_extension_root_review.json')
    review=dict(passed=False,whole_SOURCE_reviewed=False,actual_execution_released=False,
        scientific_digest=plan['scientific_digest'],source_pins=pins,parent_binding=binding,
        objective_negative_evidence=d.identity('results/geobc_physical_loss_pair/metrics.json',hashed=True),
        preparation_epoch=time.time(),preparation_GT_PT_array_GPU_reads=0)
    d.write(filename,plan);d.write(plan['root_review'],review)
    print({'status':'source_prepared_not_released','science':plan['scientific_digest'],'cache_GiB':science['cache_metadata']['file_GiB']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:prepare()
