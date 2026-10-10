"""Copy only final scalar JSON evidence into reports; no training directory publication."""
import shutil
from geobc import available_cad_pilot_data as d


def run():
    folders=['continuous_query_qualification_v1','FP32_query_ablation_v1','common_grid_fullcase_probe_v1']
    index={}
    for folder in folders:
        original='runs/geobc_exploration/'+folder
        report='results/geobc_field_queries/source_evidence/'+folder
        d.path(report).mkdir(parents=True,exist_ok=True)
        for name in ['summary.json','external_terminal.json','execution.json']:
            source=original+'/'+name;target=report+'/'+name
            assert name!='summary.json' or d.read(source)['closed']
            shutil.copyfile(d.path(source),d.path(target))
            assert d.sha(source)==d.sha(target)
            index[target]={'original':source,'sha256':d.sha(target)}
    d.write('results/geobc_field_queries/source_evidence/index.json',index)
    plan=d.read('notes/github_publish_plan.json')
    plan['allowlist']=[name for name in plan['allowlist'] if name not in ['runs/geobc_exploration/'+x for x in folders]]
    d.write('notes/github_publish_plan.json',plan)
    print({'status':'final_scalar_evidence_organized','files':len(index),'model_or_raw_arrays':0})


if __name__=='__main__':run()
