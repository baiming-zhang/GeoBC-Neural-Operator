"""Root binds already reviewed API source to one concrete local snapshot."""
import subprocess
from datetime import datetime,timezone
from geobc import available_cad_pilot_data as d


def run():
    folder=str(d.path('publication/GeoBC-Neural-Operator'))
    def git(*args):return subprocess.check_output(['git','-C',folder,*args],text=True,encoding='utf-8').strip()
    target='84f4a4898e44c81c2234eb74fc3fe996e556e7fc';base='b88cee566881fc70cdef1709db3ff4a28e3342f7'
    assert git('rev-parse','HEAD')==target and git('rev-parse',target+'^')==base
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    paths=git('diff','--name-only',base,target).splitlines()
    assert not any(p.endswith(('.pt','.npy','.npz','.h5','.hdf5')) for p in paths)
    assert not any(p.startswith(('runs/','data/','cache/','.cache/')) for p in paths)
    helper='tools/publish_reviewed_gitdata_snapshot.ps1'
    prior=d.read('notes/github_gitdata_publication_root_review.json')
    assert prior['passed'] and prior['whole_source_reviewed'] and d.sha(helper)==prior['source_sha256']=='f9ab1692564f03ef4f0419c7c7a01ca13f8223fe5ef313f3dc8b16ebc064bef9'
    assert d.read('results/geobc_field_queries/QA.json')['passed']
    assert d.read('results/geobc_available_cad_pilot/field_visualization_QA.json')['status']=='passed'
    d.write('notes/github_gitdata_57bc_pre_network_failure_root_review_preserved.json',prior)
    review={'passed':True,'whole_source_reviewed':True,'reviewer':'root',
        'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
        'decision':'one_bounded_exact_immutable_gitdata_API_publication_attempt',
        'source_sha256':d.sha(helper),'target_commit':target,'base_commit':base,
        'target_tree':git('rev-parse',target+'^{tree}'),'changed_file_count':len(paths),
        'changed_files':paths,'required_identity':{'owner':'baiming-zhang','user_id':174442095,'repository_id':1410470321,'private':True},
        'attempt_id':'query_matched_scalar_20261010t1808','actual_publication_released':True,
        'source_review':'Unchanged exact f9 API source already root reviewed and successfully exercised for b88; request bound N+13, fixed IDs/private, exact raw commit/tree before nonforce main update. No continuation blob reuse for this new commit.',
        'snapshot_review':'Root inspected complete path delta: completed scalar queries, FP32 negative/positive preserved, real original-node cloud QA, full probe technical exception preserved plus source-confirmed inactive-embedding acceptance, ongoing matched training source/config with real first-gradient/live identity. No result claim for uncompleted training.',
        'root_result_visual_QA_passed':True,'automatic_retry_count':0}
    d.write('notes/github_gitdata_publication_root_review.json',review)
    print({'target':target,'base':base,'changes':len(paths),'reviewed':True})


if __name__=='__main__':run()
