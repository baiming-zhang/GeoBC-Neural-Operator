"""Preserve oldint32 and newint64 query indices; all other access/math unchanged."""
from . import available_cad_pilot_data as d
from .prospective_dataset import SCHEMA,resolve_case

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
        valid_dtype=str(value.dtype) in ('int32','int64') if name=='query_indices' else str(value.dtype)==expected_dtype
        if value.shape!=shapes[name] or not valid_dtype:
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
