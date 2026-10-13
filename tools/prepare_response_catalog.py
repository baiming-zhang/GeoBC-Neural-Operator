"""Compose a private response catalog from completed target-free input indexes."""
import argparse
import json
from pathlib import Path
from geobc.physics_inputs import local_path


def prepare(args):
    predictors=json.loads(Path(args.predictors).read_text())
    geometry=json.loads(Path(args.geometry_index).read_text())
    physical=json.loads(Path(args.physical_index).read_text())
    if not geometry['complete_sources'] or physical['complete_CADs']!=len(geometry['CADs']) or physical['complete_loads']!=3*len(geometry['CADs']) or geometry['scientific_group']!=physical['scientific_group']:
        raise ValueError('Complete matching prepared-input catalogs required')
    catalog=dict(seed=2026,input_mode='FIELDWISE',scientific_group=geometry['scientific_group'],checkpoint=predictors['checkpoints']['fieldwise_seed2026'],scaler=predictors['scaler'],cases={})
    for cad in geometry['CADs']:
        item=geometry['sources'][cad];fact=json.loads((local_path(item['path']).parent/'source.json').read_text())
        records={cad+'/'+load:predictors['records'][cad+'/'+load] for load in ['ver','hor','tor']}
        if any(row['scientific_group']!=catalog['scientific_group'] for row in records.values()):raise ValueError('Predictor role differs')
        catalog['cases'][cad]=dict(nodes=item['expected_geometry']['nodes'],cells=item['expected_geometry']['cells'],geometry=item['path'],physical_inputs=physical['features'][cad],reference_load6={load:fact['canonical_LOADs'][load]['reference_load6'] for load in ['ver','hor','tor']},predictors=records)
    with Path(args.output).open('x') as output:output.write(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(dict(registered_CADs=len(catalog['cases']),seed=2026,scientific_body_reads=0,GT_reads=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['predictors','geometry-index','physical-index','output']:p.add_argument('--'+name,required=True)
    prepare(p.parse_args())
