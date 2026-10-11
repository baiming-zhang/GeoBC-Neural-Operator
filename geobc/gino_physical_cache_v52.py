"""One-shot native TRAIN100 inputs/physical labels for the GINO candidate.

Metadata construction opens no arrays. Unlike the FNO adapter, this cache
needs neither pre-splatted grids nor encoded target copies. All100 CAD and
three loads must be loaded before training. Radius CSR remains a separately
verified geometry-only cache; no solver-family aliases or old source edits.
"""
from .prospective_dataset import resolve_case
from . import available_cad_pilot_data as d


class NativeGINOPhysicalCache:
    def __init__(self, manifest, *, device='cuda', memory_limit_bytes=30500000000,
                 guard=lambda: None, on_read=None):
        if memory_limit_bytes != 30500000000:
            raise ValueError('The fixed SI memory budget is30.5GB')
        self.keys = list(manifest['train_keys']['100'])
        self.device, self.memory_limit_bytes = device, memory_limit_bytes
        self.guard, self.on_read = guard, on_read
        self.attempted = self.completed = False
        self.specs, self.case_specs, self.arrays, self.cases = {}, {}, {}, {}
        self.physical_targets, self.case_norms = {}, {}
        cad_loads = {}
        self.receipt = dict(input_array_reads=0, physical_TRAIN_target_reads=0,
            encoded_TRAIN_target_reads=0, DEV_FINAL_target_reads=0,
            source_CSV_H5_reads=0, scaler_fits=0, actual_tensor_bytes=0,
            peak_allocated_bytes=0, peak_reserved_bytes=0,
            peak_total_device_used_bytes_sampled=0, loaded_cases=0)
        for key in self.keys:
            cad, load = key.split('/')
            cad_loads.setdefault(cad, set()).add(load)
            record, ids = resolve_case(manifest, key, True, 100, False)
            n = record['num_nodes']
            shapes = dict(xyz=(n,3), features=(n,25), cloud=(1024,3),
                          mask=(1024,), physical=(n,4))
            self.case_specs[key] = {}
            for name, shape in shapes.items():
                item = ids[name]
                spec = dict(name=name, shape=shape, identity=item)
                if item['path'] in self.specs and self.specs[item['path']] != spec:
                    raise ValueError('Shared array identity/shape/member differs')
                self.specs[item['path']] = spec
                self.case_specs[key][name] = item['path']
        if len(self.keys) != 300 or len(set(self.keys)) != 300 or len(cad_loads) != 100:
            raise ValueError('Fixed100CAD/300 distinct TRAIN cases required')
        if any(loads != {'ver','hor','tor'} for loads in cad_loads.values()):
            raise ValueError('Every CAD must contain all three fixed loads')
        self.disk_bytes_upper = sum(s['identity']['bytes'] for s in self.specs.values())
        if self.disk_bytes_upper >= memory_limit_bytes:
            raise MemoryError('Native cache exceeds budget before any array body is read')

    def resource_check(self, torch):
        self.guard()
        if str(self.device).startswith('cuda'):
            values = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            free, total = torch.cuda.mem_get_info(self.device)
            values['peak_total_device_used_bytes_sampled'] = total-free
            for name, value in values.items():
                self.receipt[name] = max(self.receipt[name], value)
                if value > self.memory_limit_bytes:
                    raise MemoryError('Measured separate memory component exceeds30.5GB: '+name)

    def load(self):
        if self.attempted:
            raise RuntimeError('No automatic cache retry or partial replay')
        self.guard()
        self.attempted = True
        import numpy as np
        import torch
        from .physical_field_loss import norms_from_complete_physical_target
        for path, spec in self.specs.items():
            self.resource_check(torch)
            name, item = spec['name'], spec['identity']
            d.stat_matches(item)
            if self.on_read:
                self.on_read('attempt', name, item)
            value = np.load(d.path(path), mmap_mode='r', allow_pickle=False)
            category = 'physical_TRAIN_target_reads' if name == 'physical' else 'input_array_reads'
            self.receipt[category] += 1
            if tuple(value.shape) != spec['shape'] or str(value.dtype) != ('bool' if name == 'mask' else 'float32'):
                raise ValueError('Original native input/physical target shape or dtype differs')
            if value.dtype.kind == 'f' and not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite native array; no rescue')
            tensor = torch.as_tensor(np.array(value, copy=True), device=self.device)
            self.arrays[path] = tensor
            self.receipt['actual_tensor_bytes'] += tensor.numel()*tensor.element_size()
            del value
            if self.on_read:
                self.on_read('completed', name, item)
            self.resource_check(torch)
        for key, fields in self.case_specs.items():
            case = {name:self.arrays[path] for name,path in fields.items() if name != 'physical'}
            case.update(key=key, scientific_group='TRAIN')
            case['cloud'] = case['cloud'].reshape(1,1024,3)
            case['mask'] = case['mask'].reshape(1,1024)
            target = self.arrays[fields['physical']]
            self.physical_targets[key] = target
            self.case_norms[key] = norms_from_complete_physical_target(target)
            self.cases[key] = case
            self.receipt['loaded_cases'] += 1
        if len(self.cases) != 300 or self.receipt['physical_TRAIN_target_reads'] != 300:
            raise RuntimeError('Complete TRAIN100 cache required; do not train a subset')
        self.resource_check(torch)
        self.completed = True
        return self

    def case(self, key):
        if key not in self.case_specs:
            raise PermissionError('Key outside fixed TRAIN100')
        if not self.completed:
            raise RuntimeError('Native cache must be complete before training')
        return self.cases[key]
