"""Purpose-specific TRAIN cache and target-free development predictors."""
import numpy as np
import torch
from .physics_inputs import local_path

INPUTS=['xyz','features','cloud','mask','grid','query_indices','query_weights']


class LearningCache:
    def __init__(self,catalog,guard):
        self.catalog=catalog; self.guard=guard; self.shared={}; self.cases={}; self.targets={}
        self.input_reads=0; self.training_target_reads=0; self.feature_reads=0

    def array(self,path,shape,allowed,device='cuda'):
        resolved=str(local_path(path))
        if resolved not in self.shared:
            self.guard(); value=np.load(resolved,allow_pickle=False)
            if value.shape!=tuple(shape) or str(value.dtype) not in allowed or not np.isfinite(value).all():
                raise ValueError('Learning array schema differs: '+resolved)
            tensor=torch.as_tensor(value,device=device)
            if tensor.dtype==torch.int32:
                tensor=tensor.to(torch.int64)
            self.shared[resolved]=tensor; self.input_reads+=1
            self.guard()
        return self.shared[resolved]

    def load_predictors(self,records,features,role,device='cuda'):
        cad_features={}
        for key,record in records.items():
            self.guard(); n=record['num_nodes']; cad,load=key.split('/')
            if record['scientific_group'] != role:
                raise ValueError('Dataset scientific role differs')
            shapes=dict(xyz=(n,3),features=(n,25),cloud=(1024,3),mask=(1024,),grid=(29,64,64,64),query_indices=(n,8),query_weights=(n,8))
            case={}
            for name in INPUTS:
                allowed=['int32','int64'] if name=='query_indices' else ['bool'] if name=='mask' else ['float32']
                case[name]=self.array(record[name+'_path'],shapes[name],allowed,device)
            if cad not in cad_features:
                self.guard(); item=features[cad]
                value=np.load(local_path(item['path']),allow_pickle=False)
                if value.dtype!=np.float64 or value.shape!=(3,n,12) or not np.isfinite(value).all():
                    raise ValueError('Physical covariate dimensions differ')
                cad_features[cad]=torch.as_tensor(value,device=device,dtype=torch.float32); self.feature_reads+=1
                self.guard()
            case['kinematic_inputs']=cad_features[cad][['ver','hor','tor'].index(load)]
            case['cloud']=case['cloud'][None]; case['mask']=case['mask'][None]; case['grid']=case['grid'][None]
            case.update(key=key,scientific_group=role)
            self.cases[key]=case
        return self

    def load_training_targets(self):
        for key,record in self.catalog['TRAIN'].items():
            self.guard(); value=np.load(local_path(record['physical_path']),allow_pickle=False)
            if value.dtype!=np.float32 or value.shape!=(record['num_nodes'],4) or not np.isfinite(value).all():
                raise ValueError('Original physical TRAIN targets differ')
            self.targets[key]=torch.as_tensor(value,device='cuda'); self.training_target_reads+=1
            self.guard()
        if len(self.targets)!=300:
            raise ValueError('Full TRAIN target set required')
