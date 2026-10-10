"""Archive the pre-input failure and pin the minimal caller adapter fix."""
import argparse
import ast
import time
from geobc import available_cad_pilot_data as d


def run():
    plan_path='notes/geobc_common_grid_readout_plan.json'
    plan=d.read(plan_path);folder=d.path(plan['output'])
    summary=d.read(folder/'summary.json');external=d.read(folder/'external_terminal.json')
    original_launch=d.read(folder/'external_launch.json')
    assert summary['status']=='failed' and summary['closed'] and not summary['active']
    assert summary['error_type']=='KeyError' and summary['error']=="'40'"
    assert summary['committed_finite_predictions']==summary['owned_GT_array_reads']==summary['raw_GT_reads']==0
    assert all(d.exact_process_exited(external[k]) for k in ('child_process','guard_process'))
    assert original_launch['deadline_epoch']>time.time()
    assert {p.name for p in folder.iterdir()}=={'execution.json','external_launch.json','external_terminal.json','process.log','status.json','summary.json'}
    archive=folder/'recovery'/('attempt_'+str(time.time_ns()));archive.mkdir(parents=True)
    assert archive.resolve().is_relative_to(folder.resolve())
    prior_identities={}
    d.write(archive/'plan_before_fix.json',plan)
    for name in ('execution.json','external_launch.json','external_terminal.json','process.log','status.json','summary.json'):
        source=folder/name;destination=archive/name
        assert source.resolve().is_relative_to(folder.resolve()) and destination.resolve().is_relative_to(archive.resolve())
        before=d.identity(source,True);source.rename(destination)
        after=d.identity(destination,True);assert before['sha256']==after['sha256'] and before['bytes']==after['bytes']
        prior_identities[name]=after
    old_sha=plan['source_pins']['tools/evaluate_geobc_common_grid.py']
    new_sha=d.sha('tools/evaluate_geobc_common_grid.py')
    ast.parse(d.path('tools/evaluate_geobc_common_grid.py').read_text())
    ast.parse(d.path('tools/resume_geobc_common_readout_guard.py').read_text())
    assert new_sha!=old_sha
    plan['source_pins']['tools/evaluate_geobc_common_grid.py']=new_sha
    plan['source_pins']['tools/resume_geobc_common_readout_guard.py']=d.sha('tools/resume_geobc_common_readout_guard.py')
    d.write(plan_path,plan)
    review=d.read(plan['root_review']);review['source_pins']=plan['source_pins']
    review['review_basis'].append('Root actual KeyError40 pre-input failure reviewed; truthful empty TRAIN scope caller adapter only. Separate reviewed recovery guard retains original UTC deadline and real monotonic attempt segments across WSL boots.')
    d.write(plan['root_review'],review)
    # Future single-case source dependencies must match the real fixed caller;
    # scientific case selection, loss, endpoints and budgets are unchanged.
    child_path='notes/geobc_single_case_fit_plan.json'
    child=d.read(child_path);assert not (d.path(child['output'])/'execution.json').exists()
    child['source_pins']['tools/evaluate_geobc_common_grid.py']=new_sha
    d.write(child_path,child)
    child_review=d.read(child['root_review']);child_review['source_pins']=child['source_pins']
    child_review['review_basis'].append('Upstream evaluator caller empty-TRAIN adapter corrected; same parent science, single-case source/math/selection/endpoints/budget unchanged.')
    d.write(child['root_review'],child_review)
    receipt={'root_recovery_released':True,'scientific_digest':plan['scientific_digest'],'output':plan['output'],
        'archived_external_launch':prior_identities['external_launch.json'],'archived_identities':prior_identities,
        'prior_predictions':0,'prior_GT_reads':0,'original_absolute_UTC_deadline':original_launch['deadline_epoch'],
        'old_evaluator_sha256':old_sha,'new_evaluator_sha256':new_sha,
        'original_failed_attempt_external_seconds':external['elapsed_epoch_seconds'],
        'repair_scope':'Evaluation-only cache has no TRAIN records; supply empty train40 to legacy loader, no model/data/math/seed change',
        'cost_policy':'Original1800UTC cutoff retained; original attempt/gap/continuation all explicit, no monotonic origin transplantation across boots',
        'actual_recovery_executions':0}
    d.write('notes/geobc_common_readout_recovery.json',receipt)
    print({'status':'explicit_pre_input_recovery_prepared','original_deadline_epoch':original_launch['deadline_epoch'],
           'prior_predictions':0,'prior_GT_reads':0,'budget_reset':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    if parser.parse_args().prepare:run()
