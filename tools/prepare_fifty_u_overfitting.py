"""Freeze a nested 50-CAD fitting group using existing TRAIN metadata only."""
import argparse,json
from pathlib import Path

ALIGNMENT='official CSV GRID coordinates -> H5 vertices'


def selection(anchors,pool,train40,train60):
    """Keep the five anchors, then archive cases on TRAIN40, then TRAIN60 IDs.

    No label values or model errors enter selection. One load per distinct CAD.
    """
    keys=list(anchors);seen={k.split('/')[0] for k in keys}
    for key in pool:
        cad=key.split('/')[0]
        if cad in train40 and cad not in seen:keys.append(key);seen.add(cad)
    loads=('ver','hor','tor')
    for i,cad in enumerate(sorted(train60)):
        if len(keys)==50:break
        if cad not in seen:keys.append(cad+'/'+loads[i%3]);seen.add(cad)
    if len(keys)!=50 or len(seen)!=50 or keys[:5]!=list(anchors):raise ValueError('Need exactly50 distinct CADs nested around the five anchors')
    return keys


def run(workspace,repo):
    def read(path):return json.loads((workspace/path).read_text(encoding='utf-8'))
    config=json.loads((repo/'configs/fullcycle_u_overfitting.json').read_text())
    anchors=config['five'];source=read('notes/overfit_sample_sources.json')
    pool=read('data/prepared50_multifield/manifest.json')
    split=read('notes/geobc_split.json');prospective=read('notes/geobc_prospective_cad_split_v1.json')['scientific']
    inputs=read('data/geobc_prospective_CPU_inputs_v1/manifest.json')['records']
    targets=read('data/geobc_prospective_TRAIN60_targets_v1/manifest.json')['records']
    train40=split['splits']['train']['cad_ids'];train60=prospective['additional_TRAIN60_CADs']
    keys=selection(anchors,pool['selection'],train40,train60)
    forbidden=set(split['groups']['validation']+split['groups']['test']+prospective['DEV_CADs']+prospective['FINAL_CADs']+prospective['original_sealed_DEV20_preserved'])
    if {k.split('/')[0] for k in keys}&forbidden:raise ValueError('Held-out CAD in fitting selection')
    pool_records={r['sample_key']:r for r in pool['cases']}
    sources={};records=[]
    for key in keys:
        cad=key.split('/')[0]
        if key in source['cases']:
            sources[key]=source['cases'][key];origin='original_five_fitting_TRAIN'
            m=read('data/prepared_benchmark5/manifest.json');r=next(r for r in m['cases'] if r['sample_key']==key);nodes=r['nodes']
        elif cad in train40:
            r=pool_records[key]
            if r['label_alignment']!=ALIGNMENT:raise ValueError('Unaligned existing case')
            sources[key]=str(workspace/'data/prepared50_multifield'/Path(r['npz'].replace('\\','/')).name)
            origin='TRAIN40';nodes=r['nodes']
        else:
            a,b=inputs[key],targets[key]
            if a['label_alignment']!=ALIGNMENT or b['label_alignment']!=ALIGNMENT or b['scientific_group']!='TRAIN' or b['status']!='completed' or a['num_nodes']!=b['num_nodes']:raise ValueError('Input/U alignment or TRAIN role mismatch')
            sources[key]=dict(sample_key=key,scientific_group='TRAIN',label_alignment=ALIGNMENT,
                xyz_path=str(workspace/a['xyz_path']),features_path=str(workspace/a['features_path']),physical_path=str(workspace/b['physical_path']))
            origin='additional_TRAIN60';nodes=a['num_nodes']
        records.append(dict(sample_key=key,cad_id=cad,nodes=nodes,training_source=origin))
    for value in sources.values():
        for file in ([value] if isinstance(value,str) else [value[k] for k in ('xyz_path','features_path','physical_path')]):
            if not Path(file).is_file():raise FileNotFoundError(file)
    for field in ('selected_arms','require_selected_arms','rerun_source'):config.pop(field,None)
    config.update(groups=['fifty'],fifty=keys,arm_seconds=10800,whole_seconds=28800,
        report_directory='results/fifty_u_overfitting',save_best_checkpoint=False,save_best_predictions=False,
        selection='Nested50 distinct TRAIN CADs, one CAD/load each: original5, remaining TRAIN40 cases in prepared archive order, then sorted additionalTRAIN60 IDs with ver/hor/tor cycling. Frozen before labels are read; no performance selection.',
        protocol='Same architectures, seed2026, U3 loss/input schema, Adam and OneCycle peak.001/5000, clip1 and eval50 as completed single/five. Exactly5000 complete original-node case updates, each of50 cases100 visits. No early stopping/warmstart. Output RMS fit only these50. Cache each case once across all8 methods. Save final predictions and full last.pt, skip unused best binaries.',
        goal='Every fitted case U-vector relative L2 below1%; report actual5000 regardless of goal.')
    manifest=dict(selection_policy=config['selection'],cases=records,case_count=50,distinct_CADs=50,nodes_total=sum(r['nodes'] for r in records),largest_case_nodes=max(r['nodes'] for r in records),numerical_array_reads=0,metadata_only=True)
    destinations=[(repo/'configs/fifty_u_overfitting.json',config),(repo/'configs/fifty_u_samples.json',manifest),(workspace/'notes/fifty_u_sources.json',dict(cases=sources,transformer_manifest=source['transformer_manifest']))]
    for path,_ in destinations:
        if path.exists():raise FileExistsError(path)
    for path,value in destinations:path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path('.'));a=p.parse_args();run(a.workspace,a.repo)
