"""Reuse the original deduplicated physical cache without unused encoded labels."""
from .train_physical_cache_v12 import PhysicalTRAINCache


class PhysicalTargetOnlyCache(PhysicalTRAINCache):
    def __init__(self,manifest,train_CAD_count,family,**kwargs):
        # The target-free grid/native inputs are identical to the FNO storage
        # layout used by the original FNO-conditioned Hash experiment.
        if family!='FNO_GLOBAL_HASH' or train_CAD_count!=100:
            raise ValueError('Only this fixed globalHash100 continuation')
        super().__init__(manifest,train_CAD_count,'FNO9',**kwargs)
        removed={fields.pop('target')for fields in self.case_specs.values()}
        assert all(self.specs[path]['name']=='target'for path in removed)
        for path in removed:del self.specs[path]
        self.disk_bytes_upper=sum(spec['identity']['bytes']for spec in self.specs.values())
        self.receipt['unused_encoded_target_files_removed_before_load']=len(removed)
        assert not any(spec['name']=='target'for spec in self.specs.values())

    def load(self):
        super().load()
        assert self.receipt['encoded_TRAIN_target_read_attempts']==self.receipt['encoded_TRAIN_target_reads']==0
        assert all('target'not in case for case in self.cases.values())
        return self
