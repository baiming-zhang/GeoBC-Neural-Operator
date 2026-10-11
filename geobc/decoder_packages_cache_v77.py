"""Same fixed9 TRAIN physical cache; no unused Gaussian or encoded targets."""
from .local_query_geometry_cache_v76 import NineCaseCache as MetadataCache
from .train_resident_cache import TRAINResidentCache
from . import available_cad_pilot_data as d


class NineCaseCache(MetadataCache):
    def load(self):
        TRAINResidentCache.load(self);self.completed=False
        import numpy as np
        import torch
        from .physical_field_loss import norms_from_complete_physical_target
        for key,spec in self.physical_specs.items():
            self.resource_check(torch);d.stat_matches(spec['identity'])
            self.receipt['physical_TRAIN_target_read_attempts']+=1
            if self.on_read:self.on_read('attempt','physical',spec['identity'])
            array=np.load(d.path(spec['identity']['path']),mmap_mode='r',allow_pickle=False)
            self.receipt['physical_TRAIN_target_reads']+=1
            if array.shape!=spec['shape'] or str(array.dtype)!='float32' or not np.isfinite(array).all():
                raise ValueError('Complete signed physical TRAIN field required')
            target=torch.as_tensor(np.array(array,copy=True),device=self.device)
            self.physical_targets[key]=target
            self.case_norms[key]=norms_from_complete_physical_target(target)
            self.receipt['actual_tensor_bytes']+=target.numel()*target.element_size()
            if self.on_read:self.on_read('completed','physical',spec['identity'])
            del array
        assert self.receipt['physical_TRAIN_target_reads']==9 and self.receipt['encoded_TRAIN_target_reads']==0
        assert self.receipt['Gaussian_basis_bytes']==0
        self.resource_check(torch);self.completed=True
        return self
