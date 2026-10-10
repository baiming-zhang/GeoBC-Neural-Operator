"""Root CPU interpretation of complete probe; preserve original failed status."""
from geobc import available_cad_pilot_data as d


def run():
    folder='runs/geobc_exploration/common_grid_fullcase_probe_v1'
    summary=d.read(folder+'/summary.json'); ext=d.read(folder+'/external_terminal.json')
    plan=d.read('notes/geobc_common_grid_probe_plan.json')
    assert summary['status']=='failed' and summary['error_type']=='AssertionError'
    assert ext['exit_code']==1 and not ext['external_timeout'] and ext['elapsed_epoch_seconds']<900
    assert ext['closed'] and not ext['active']
    assert all(d.exact_process_exited(ext[k]) for k in ('guard_process','child_process'))
    assert summary['optimizer_updates']==summary['raw_GT_reads']==summary['DEV_reads']==summary['scaler_fits']==0
    assert all(d.sha(p)==h for p,h in plan['source_pins'].items())
    fno,pp=(summary['arms'][a] for a in ['FNO9','PP256'])
    assert fno['gradient_pass'] and fno['resource_pass'] and pp['resource_pass']
    assert pp['gradients']['operator']['missing']==['backbone.embedding.weight','backbone.embedding.bias']
    assert all(g['finite'] and g['L2']>0 for g in pp['gradients'].values())
    assert not pp['gradients']['encoder']['missing'] and not pp['gradients']['head']['missing']
    assert all(v>0 for v in pp['leaf_gradients'].values())
    assert fno['shared_init_sha256']==pp['shared_init_sha256']
    # Exact official source/adaptor branches inspected by root: condition=None
    # bypasses the optional 3->256 embedding in both checkpointed/noncheckpoint paths.
    official='third_party/Transolver_plus/models/Transolver_plus.py'
    assert d.sha(official)=='7104ad5b90294d1d97b1e7e86b3eeb2ccbe11eec93d2e7bf566bcb3a8ab6e858'
    value={'passed':True,'actual_resource_and_active_gradient_qualified':True,
           'reviewer':'root','original_status_preserved':'failed/AssertionError',
           'cause':'Probe required gradients on optional condition embedding unused with condition=None; both actual whole-case backwards completed',
           'expected_inactive_parameters':['operator.backbone.embedding.weight','operator.backbone.embedding.bias'],
           'expected_inactive_count':1024,'scientific_digest':plan['scientific_digest'],
           'source_pins':plan['source_pins'],'official_source_sha256':d.sha(official),
           'summary_identity':d.identity(folder+'/summary.json',hashed=True),
           'external_identity':d.identity(folder+'/external_terminal.json',hashed=True),
           'exact_processes_exited':True,'additional_GPU_execution':0,'automatic_replay':False,
           'epoch_seconds':ext['elapsed_epoch_seconds'],'measurements':summary['arms'],
           'capacity_nominal_gap':(4526858-4523268)/4523268,
           'capacity_active_gap':(4526858-(4523268-1024))/(4523268-1024)}
    d.write('notes/geobc_common_probe_interpretation_root_review.json',value)
    print({'status':'resource_and_active_gradients_qualified','actual_replay':0,
           'FNO_seconds':fno['full_case_forward_backward_seconds'],
           'PP_seconds':pp['full_case_forward_backward_seconds']})


if __name__=='__main__':run()
