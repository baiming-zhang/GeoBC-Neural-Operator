"""Nine fixed TRAIN fields; original inputs deduplicated, physical labels once."""
from .train_resident_cache import TRAINResidentCache
from .prospective_dataset import resolve_case
from . import available_cad_pilot_data as d


class NineCaseCache(TRAINResidentCache):
    def __init__(self, manifest, keys, **kwargs):
        super().__init__(manifest,100,'FNO9',**kwargs)
        assert len(keys)==len(set(keys))==9 and set(keys)<=set(self.keys)
        self.keys=list(keys)
        self.case_specs={k:{n:p for n,p in self.case_specs[k].items() if n!='target'} for k in keys}
        retained={p for fields in self.case_specs.values() for p in fields.values()}
        self.specs={p:s for p,s in self.specs.items() if p in retained}
        self.disk_bytes_upper=sum(s['identity']['bytes'] for s in self.specs.values())
        self.physical_specs={}
        for k in keys:
            record, identities=resolve_case(manifest,k,True,100,True)
            self.physical_specs[k]=dict(identity=identities['physical'],shape=(record['num_nodes'],4))
        self.physical_targets={};self.case_norms={}
        self.receipt.update(physical_TRAIN_target_read_attempts=0,physical_TRAIN_target_reads=0,
                            Gaussian_basis_unique_CADs=0,Gaussian_basis_bytes=0)

    def load(self):
        super().load();self.completed=False
        import numpy as np
        import torch
        from .physical_field_loss import norms_from_complete_physical_target
        from .local_query_geometry_v76 import gaussian_weights
        for key,spec in self.physical_specs.items():
            self.resource_check(torch);d.stat_matches(spec['identity'])
            self.receipt['physical_TRAIN_target_read_attempts']+=1
            if self.on_read:self.on_read('attempt','physical',spec['identity'])
            array=np.load(d.path(spec['identity']['path']),mmap_mode='r',allow_pickle=False)
            self.receipt['physical_TRAIN_target_reads']+=1
            if array.shape!=spec['shape'] or str(array.dtype)!='float32' or not np.isfinite(array).all():
                raise ValueError('Finite full original-node signed physical field required')
            target=torch.as_tensor(np.array(array,copy=True),device=self.device)
            self.physical_targets[key]=target
            self.case_norms[key]=norms_from_complete_physical_target(target)
            self.receipt['actual_tensor_bytes']+=target.numel()*target.element_size()
            if self.on_read:self.on_read('completed','physical',spec['identity'])
            del array
        bases={}
        with torch.no_grad():
            for key,case in self.cases.items():
                fields=self.case_specs[key]
                identity=tuple(fields[n] for n in ('xyz','cloud','mask'))
                if identity not in bases:
                    count=len(case['xyz'])
                    basis=case['xyz'].new_empty((count,1024))
                    for start in range(0,count,32768):
                        self.resource_check(torch);stop=min(start+32768,count)
                        basis[start:stop]=gaussian_weights(case['xyz'][start:stop],case['cloud'],case['mask'])
                    if not torch.isfinite(basis).all() or (basis.sum(-1)-1).abs().max()>3e-6:
                        raise ValueError('Finite normalized target-free Gaussian basis required')
                    bases[identity]=basis
                    self.receipt['Gaussian_basis_unique_CADs']+=1
                    self.receipt['Gaussian_basis_bytes']+=basis.numel()*basis.element_size()
                case['geometry_basis']=bases[identity]
        assert self.receipt['physical_TRAIN_target_reads']==9 and self.receipt['encoded_TRAIN_target_reads']==0
        assert self.receipt['Gaussian_basis_unique_CADs']==3
        self.resource_check(torch);self.completed=True
        return self
