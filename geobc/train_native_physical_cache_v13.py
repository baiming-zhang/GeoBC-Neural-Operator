"""One-shot TRAIN physical cache and adapter; existing loss mathematics reused."""
from .train_resident_cache import TRAINResidentCache
from .prospective_dataset import resolve_case
from . import available_cad_pilot_data as d


class NativePhysicalTRAINCache(TRAINResidentCache):
    def __init__(self, manifest, train_CAD_count, family, **kwargs):
        if family != 'PP256_NATIVE' or train_CAD_count != 100:
            raise ValueError('This release is only the fixed nativePP100 experiment')
        super().__init__(manifest, train_CAD_count, family, **kwargs)
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
        value.update(physical_TRAIN_read=self.receipt['physical_TRAIN_target_reads']>0,
                     DEV_FINAL_read=False, physical_TRAIN_disk_bytes_upper=sum(s['identity']['bytes'] for s in self.physical_specs.values()))
        return value



def full_physical_backward(model, case, cache, scale, allowed_keys, guard):
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Fixed TRAIN scope only')
    if model.macro_family != 'PP256_NATIVE' or not cache.completed:
        raise RuntimeError('NativePP and complete physical cache required')
    import torch
    from .macro_training import encode_case, query_case
    from .physical_field_loss import physical_relative_squared_loss
    key=case['key']; count=len(case['xyz']); norms=cache.case_norms[key]
    assert norms.node_count==count and cache.physical_targets[key].shape==(count,4)
    guard(); context,latent=encode_case(model,case)
    assert context.shape==(1,32) and context.requires_grad and latent.shape==(count,16) and latent.requires_grad
    c=context.detach().requires_grad_(True); l=latent.detach().requires_grad_(True)
    total=0.; consumed=0
    for start in range(0,count,32768):
        guard(); stop=min(start+32768,count)
        prediction=query_case(model,case,start,stop,c,l)
        assert prediction.shape==(stop-start,4)
        loss=physical_relative_squared_loss(prediction,cache.physical_targets[key][start:stop],scale,norms)
        loss.backward(); total+=float(loss.detach()); consumed+=stop-start
    assert consumed==count and c.grad is not None and l.grad is not None
    assert torch.isfinite(c.grad).all() and torch.isfinite(l.grad).all()
    torch.autograd.backward((context,latent),(c.grad,l.grad))
    return total,dict(geometry_context_L2=float(c.grad.norm()),operator_latent_L2=float(l.grad.norm()),
                      original_nodes_used=count,node_sampling=False)
