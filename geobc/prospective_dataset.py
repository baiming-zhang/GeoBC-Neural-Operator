"""Explicit100-CAD protocol: shared inputs, TRAIN-only target access, no200 alias.

This loader does not create targets, fit a scaler, construct a model or authorize
FINAL evaluation. Future experiment guards bind the immutable manifest identity.
All validation/FINAL numerical target requests fail before array/library access.
"""
from . import available_cad_pilot_data as d

SCHEMA='GeoBC_prospective100_dataset_v1'
BASE_INPUTS=('xyz','features','cloud','mask')
GRID_INPUTS=('grid','query_indices','query_weights')


def resolve_case(manifest, key, with_targets=False, train_CAD_count=100, for_fno=False):
    """Resolve an allowed read from metadata; reject protected targets before I/O."""
    if train_CAD_count not in (40,100):raise ValueError('This protocol has40/100TRAIN scopes only')
    if manifest.get('schema')!=SCHEMA or manifest.get('status')!='completed' or not manifest.get('feasibility_pass'):
        raise ValueError('A complete prospective100 manifest is required')
    if key not in manifest['records']:raise KeyError(key)
    role=manifest['case_roles'][key]
    if with_targets and (role!='TRAIN' or key not in manifest['train_keys'][str(train_CAD_count)]):
        raise PermissionError('Numerical targets are permitted only in the selected TRAIN scope: '+key)
    record=manifest['records'][key]
    if record['scientific_group']!=role or record['cad_id']!=key.split('/')[0] or record['key']!=key:
        raise ValueError('Case identity/scientific role differs')
    if record['status']!='completed' or not record['feasibility_pass'] or record['num_nodes']<=0:
        raise ValueError('Incomplete input record')
    if role in ('DEV','FINAL') and any(name in record or name in record['file_identities'] for name in ('target_path','physical_path')):
        raise ValueError('Protected input records must never carry numerical target members')
    members=list(BASE_INPUTS)+(list(GRID_INPUTS) if for_fno else [])
    if with_targets:members+=['target','physical']
    identities={name:record['file_identities'][name+'_path'] for name in members}
    if any(record[name+'_path']!=item['path'] for name,item in identities.items()):
        raise ValueError('Member path differs from immutable identity')
    return record,identities


def load_case(manifest,key,with_targets=False,train_CAD_count=100,device='cuda',for_fno=False,on_read=None):
    record,identities=resolve_case(manifest,key,with_targets,train_CAD_count,for_fno)
    # The protected-target rejection above intentionally precedes NumPy/Torch.
    import numpy as np
    import torch
    n=record['num_nodes']
    shapes={'xyz':(n,3),'features':(n,25),'cloud':(1024,3),'mask':(1024,),
            'grid':(29,64,64,64),'query_indices':(n,8),'query_weights':(n,8),'target':(n,4),'physical':(n,4)}
    receipt=dict(input_array_reads=0,cached_TRAIN_target_array_reads=0,DEV_FINAL_target_array_reads=0,
                 source_CSV_H5_reads=0,scaler_fits=0,attempted_members=[],completed_members=[])
    result={'key':key,'scientific_group':record['scientific_group'],'read_receipt':receipt}
    for name,item in identities.items():
        is_target=name in ('target','physical');d.stat_matches(item)
        receipt['attempted_members'].append(name)
        if on_read:on_read('attempt',key,name,item)
        value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False)
        receipt['cached_TRAIN_target_array_reads' if is_target else 'input_array_reads']+=1
        if on_read:on_read('opened',key,name,item)
        expected_dtype='bool' if name=='mask' else 'int64' if name=='query_indices' else 'float32'
        if value.shape!=shapes[name] or str(value.dtype)!=expected_dtype:
            raise ValueError('Original cached member shape/dtype differs: '+name)
        if value.dtype.kind=='f' and not np.isfinite(value).all():raise FloatingPointError('Nonfinite input/target: '+name)
        if name=='query_indices' and (value.min()<0 or value.max()>=64**3):raise ValueError('Query index outside R64')
        if name=='physical':result[name]=value
        else:result[name]=torch.as_tensor(np.array(value,copy=True),device=device)
        receipt['completed_members'].append(name)
        if on_read:on_read('completed',key,name,item)
    result['cloud']=result['cloud'].reshape(1,1024,3);result['mask']=result['mask'].reshape(1,1024)
    if for_fno:result['grid']=result['grid'].reshape(1,29,64,64,64)
    return result
