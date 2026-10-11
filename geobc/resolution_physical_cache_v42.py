"""Native-input physicalTRAIN cache; isolated from frozen R64-only cache.

Metadata construction does not read numerical bodies. One actual load reads
complete old TRAIN100 arrays under a released public GPU lock and budget.
The PP256_NATIVE base-family token is a member selector, not an operator.
"""
from .train_resident_cache import TRAINResidentCache
from .prospective_dataset import resolve_case
from . import available_cad_pilot_data as d


class NativePhysicalTRAINCache(TRAINResidentCache):
    def __init__(self, manifest, train_CAD_count=100, **kwargs):
        if train_CAD_count != 100:
            raise ValueError('Fixed original TRAIN100 only')
        # PP256_NATIVE selects only common native input members in the existing
        # loader. It neither builds a PP model nor names the candidate solver.
        super().__init__(manifest, train_CAD_count, 'PP256_NATIVE', **kwargs)
        self.physical_specs = {}
        self.physical_targets = {}; self.case_norms = {}
        self.receipt.update(physical_TRAIN_target_read_attempts=0,
                            physical_TRAIN_loaded_cases=0, physical_TRAIN_tensor_bytes=0)
        for key in self.keys:
            record, identities = resolve_case(manifest, key, True, 100, False)
            self.physical_specs[key] = dict(identity=identities['physical'], shape=(record['num_nodes'], 4))
        if self.disk_bytes_upper + sum(s['identity']['bytes'] for s in self.physical_specs.values()) > self.memory_limit_bytes:
            raise MemoryError('Both caches exceed fixed memory budget before any bodies are opened')

    def load(self):
        super().load()
        self.completed = False
        import numpy as np
        import torch
        from .physical_field_loss import norms_from_complete_physical_target
        for key, spec in self.physical_specs.items():
            self.resource_check(torch); item = spec['identity']; d.stat_matches(item)
            self.receipt['physical_TRAIN_target_read_attempts'] += 1
            if self.on_read:
                self.on_read('attempt', 'physical', item)
            value = np.load(d.path(item['path']), mmap_mode='r', allow_pickle=False)
            self.receipt['physical_TRAIN_target_reads'] += 1
            if value.shape != spec['shape'] or str(value.dtype) != 'float32' or not np.isfinite(value).all():
                raise ValueError('Complete original aligned physical TRAIN target required')
            target = torch.as_tensor(np.array(value, copy=True), device=self.device)
            self.physical_targets[key] = target
            self.case_norms[key] = norms_from_complete_physical_target(target)
            size = target.numel()*target.element_size()
            self.receipt['actual_tensor_bytes'] += size
            self.receipt['physical_TRAIN_tensor_bytes'] += size
            self.receipt['physical_TRAIN_loaded_cases'] += 1
            if self.on_read:
                self.on_read('completed', 'physical', item)
            del value
            self.resource_check(torch)
        assert set(self.physical_targets) == set(self.case_norms) == set(self.keys)
        assert self.receipt['physical_TRAIN_loaded_cases'] == 300
        self.completed = True
        return self

    def metadata_profile(self):
        value = super().metadata_profile()
        value.pop('physical_DEV_FINAL_read')
        value.update(cache_role='FNO_resolution_native_inputs_and_physical_TRAIN', loader_member_policy='native-only original inputs, no oldR64 grid or stencil', physical_TRAIN_read=self.receipt['physical_TRAIN_target_reads']>0,
                     DEV_FINAL_read=False, physical_TRAIN_disk_bytes_upper=sum(s['identity']['bytes'] for s in self.physical_specs.values()))
        return value

