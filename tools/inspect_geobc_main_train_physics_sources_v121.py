"""Metadata/stat inventory for the fixed main TRAIN100 physical-input question.

Does not open raw files, archives or numerical arrays; not a physics gate.
"""
import argparse,time
from geobc import available_cad_pilot_data as d
def inspect():
    began=time.perf_counter();dataset_path='data/geobc_prospective_dataset_v1/manifest.json';split_path='notes/geobc_prospective_cad_split_v1.json'
    dataset=d.read(dataset_path);split=d.read(split_path);ss=split['scientific'];cads=ss['TRAIN_CADs'];old40=set(ss['old_TRAIN40_CADs'])
    assert len(cads)==len(set(cads))==100 and len(old40)==40 and old40<=set(cads)
    assert not set(cads)&(set(ss['DEV_CADs'])|set(ss['FINAL_CADs']))and ss['loads']==['ver','hor','tor']
    keys={cad+'/'+load for cad in cads for load in ('ver','hor','tor')};assert keys==set(dataset['train_keys']['100'])and len(keys)==300
    assert all(dataset['records'][key]['scientific_group']=='TRAIN'and dataset['records'][key]['feasibility_pass']for key in keys)
    new_index=d.read(dataset['source_bindings']['new_input_index']['path'])['records'];rows={}
    for cad in cads:
        root='data/raw'if cad in old40 else'data/raw_geobc_coverage';assets={kind:root+'/'+kind+'/'+cad+suffix for kind,suffix in (('FieldMesh','.h5'),('SolverDeck','.fem'),('Field','.csv'))}
        identities={kind:d.identity(path,hashed=False)if d.path(path).is_file()else None for kind,path in assets.items()}
        if cad in old40:
            grid=d.relative(d.path('data/prepared_geobc_multiload/'+cad+'_ver.npz'))
            descriptor=dict(kind='legacy canonical archive GRID member only;no defaulttargetloader',path=grid,exists=d.path(grid).is_file(),member_body_not_opened=True,
                required_actual_member='grid_node_ids',actual_member_name_and_mapping_not_yet_verified=True)
        else:
            perm=new_index[cad+'/ver']['permutation'];d.stat_matches(perm)
            descriptor=dict(kind='existing official CSV/H5 row permutation;CSV numeric GRID+XYZ4 only if mapping needed',permutation_identity=perm,
                source_csv_identity=new_index[cad+'/ver']['deferred_source'],actual_numeric_columns_before_prediction='GRID+XYZ4 only,never U/S',member_body_not_opened=True)
        rows[cad]=dict(scientific_group='TRAIN',nodes=dataset['records'][cad+'/ver']['num_nodes'],old_TRAIN40=cad in old40,
            raw_source_stat_identities=identities,source_asset_metadata_present=all(v is not None for v in identities.values()),canonical_GRID_input_descriptor=descriptor,
            actual_source_geometry_material_constraints_qualification=False,actual_K_PCG_feature_bodies_read=0)
    out='notes/geobc_main_train_physics_input_v121_source_inventory.json';assert not d.path(out).exists()
    result=dict(status='actual_metadata_inventory_completed',dataset_identity=d.identity(dataset_path,hashed=True),split_identity=d.identity(split_path,hashed=True),
        fixed_TRAIN_CAD_denominator=100,fixed_canonical_case_denominator=300,source_assets_present_CADs=sum(v['source_asset_metadata_present']for v in rows.values()),
        actual_metadata_seconds=time.perf_counter()-began,rows=rows,scaler=dataset['scaler'],scaler_policy=dataset['scaler_policy'],
        raw_body_hashes_performed=0,raw_H5_deck_CSV_numeric_reads=0,array_archive_bodies_read=0,GT_reads=0,DEV_FINAL_numeric_reads=0,GPU_K_PCG_model_reads=0,
        actual_physics_input_qualified=False,main300_kinematic_inputs_ready=False,neural_training_released=False,
        legacy_positive_VER_reuse_requires_exact_original_source_and_row_binding=True,prior_scientific_input_negatives_require_no_automatic_replay=True,
        geometry_family_separation_proven=False,no_CAD_substitution_or_subset_model_selection=True)
    d.write(out,result);print({k:v for k,v in result.items()if k not in ('rows','scaler')})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inspect',action='store_true')
    if p.parse_args().inspect:inspect()
    else:print('{"status":"metadata_SOURCE_only","scientific_bodies_GPU_GT":0}')
