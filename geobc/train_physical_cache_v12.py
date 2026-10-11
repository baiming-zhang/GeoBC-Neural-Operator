"""One-shot TRAIN physical cache and adapter; existing loss mathematics reused."""
from .train_resident_cache import TRAINResidentCache
from .prospective_dataset import resolve_case
from . import available_cad_pilot_data as d


class PhysicalTRAINCache(TRAINResidentCache):
    def __init__(self, manifest, train_CAD_count, family, **kwargs):
        if family != 'FNO9' or train_CAD_count != 100:
            raise ValueError('This release is only the fixed FNO100 experiment')
        super().__init__(manifest, train_CAD_count, family, **kwargs)
        self.physical_specs = {}
        self.physical_targets = {}; self.case_norms = {}
        self.receipt.update(physical_TRAIN_target_read_attempts=0,
                            physical_TRAIN_loaded_cases=0, physical_TRAIN_tensor_bytes=0)
        for key in self.keys:
            record, identities = resolve_case(manifest, key, True, 100, True)
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
        value.update(physical_TRAIN_read=self.receipt['physical_TRAIN_target_reads']>0,
                     DEV_FINAL_read=False, physical_TRAIN_disk_bytes_upper=sum(s['identity']['bytes'] for s in self.physical_specs.values()))
        return value


def full_physical_backward(model, case, cache, scale, allowed_keys, guard):
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Fixed TRAIN scope only')
    if model.macro_family != 'FNO9' or not cache.completed:
        raise RuntimeError('Actual FNO and complete fixed physical cache required')
    from .physical_field_loss import full_common_case_backward
    key = case['key']; count = len(case['xyz'])
    loss, leaf = full_common_case_backward(model, case, cache.physical_targets[key], scale, cache.case_norms[key], guard)
    return loss, dict(geometry_context_L2=leaf['context'], operator_latent_L2=leaf['latent'],
                     original_nodes_used=count, node_sampling=False)
