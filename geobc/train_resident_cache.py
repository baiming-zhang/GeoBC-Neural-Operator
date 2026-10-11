"""Explicit TRAIN-only, path-deduplicated resident cache for future macro jobs.

Construction is metadata-only. load() is one shot and must be called under a
released experiment's lock, dual deadline and resource guard. This module
does not launch work, release a protocol, parse raw CSV/H5, fit a scaler or
read physical/DEV/FINAL labels. It preserves original int32/int64 stencils.
"""
from .prospective_dataset import resolve_case
from . import available_cad_pilot_data as d

FAMILIES = ('FNO9', 'PP256_NATIVE', 'HASH_PN')
BASE = ('xyz', 'features', 'cloud', 'mask', 'target')
GRID = ('grid', 'query_indices', 'query_weights')


class TRAINResidentCache:
    def __init__(self, manifest, train_CAD_count, family, device='cuda',
                 memory_limit_bytes=30500000000, guard=lambda: None, on_read=None):
        if family not in FAMILIES or train_CAD_count not in (40, 100):
            raise ValueError('Only declared solver families and TRAIN40/100 allowed')
        if not isinstance(memory_limit_bytes, int) or isinstance(memory_limit_bytes, bool) or memory_limit_bytes <= 0:
            raise ValueError('Positive fixed memory budget required')
        self.keys = list(manifest['train_keys'][str(train_CAD_count)])
        if len(self.keys) != train_CAD_count*3 or len(set(self.keys)) != len(self.keys):
            raise ValueError('Exactly three fixed load cases per selected TRAIN CAD required')
        groups = {}
        for key in self.keys:
            cad, load = key.split('/')
            groups.setdefault(cad, set()).add(load)
        if len(groups) != train_CAD_count or any(loads != {'ver', 'hor', 'tor'} for loads in groups.values()):
            raise ValueError('Every selected TRAIN CAD must have exactly ver/hor/tor')
        self.family, self.device = family, device
        self.guard, self.on_read = guard, on_read
        self.memory_limit_bytes = memory_limit_bytes
        self.specs, self.case_specs, self.arrays, self.cases = {}, {}, {}, {}
        self.attempted = self.completed = False
        self.receipt = dict(input_array_read_attempts=0, input_array_reads=0,
                            encoded_TRAIN_target_read_attempts=0, encoded_TRAIN_target_reads=0,
                            physical_TRAIN_target_reads=0, DEV_FINAL_target_reads=0,
                            source_CSV_H5_reads=0, scaler_fits=0, loaded_cases=0,
                            actual_tensor_bytes=0, peak_allocated_bytes=None, peak_reserved_bytes=None)
        members = BASE+(GRID if family == 'FNO9' else ())
        for key in self.keys:
            record, identities = resolve_case(manifest, key, True, train_CAD_count, family == 'FNO9')
            n = record['num_nodes']
            shapes = {'xyz': (n, 3), 'features': (n, 25), 'cloud': (1024, 3), 'mask': (1024,),
                      'target': (n, 4), 'grid': (29, 64, 64, 64),
                      'query_indices': (n, 8), 'query_weights': (n, 8)}
            self.case_specs[key] = {}
            for name in members:
                item = identities[name]
                path = item['path']
                identity = {k: item[k] for k in ('path', 'bytes', 'mtime_ns')}
                spec = dict(name=name, identity=identity, shape=shapes[name])
                if path in self.specs and self.specs[path] != spec:
                    raise ValueError('Shared path has inconsistent identity/member/shape')
                self.specs[path] = spec
                self.case_specs[key][name] = path
        self.disk_bytes_upper = sum(s['identity']['bytes'] for s in self.specs.values())
        if self.disk_bytes_upper > memory_limit_bytes:
            raise MemoryError('Resident cache alone exceeds fixed budget; no arrays opened')

    def resource_check(self, torch):
        self.guard()
        if str(self.device).startswith('cuda'):
            self.receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(self.device)
            self.receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(self.device)
            if max(self.receipt['peak_allocated_bytes'], self.receipt['peak_reserved_bytes']) > self.memory_limit_bytes:
                raise MemoryError('Actual allocated/reserved separately exceed fixed budget')

    def load(self):
        if self.attempted:
            raise RuntimeError('No automatic resident-cache retry or partial replay')
        self.guard()
        self.attempted = True
        import numpy as np
        import torch
        self.resource_check(torch)
        for path, spec in self.specs.items():
            self.resource_check(torch)
            name = spec['name']
            d.stat_matches(spec['identity'])
            category = 'encoded_TRAIN_target' if name == 'target' else 'input_array'
            self.receipt[category+'_read_attempts'] += 1
            if self.on_read:
                self.on_read('attempt', name, spec['identity'])
            value = np.load(d.path(path), mmap_mode='r', allow_pickle=False)
            self.receipt[category+'_reads'] += 1
            allowed = ('int32', 'int64') if name == 'query_indices' else ('bool',) if name == 'mask' else ('float32',)
            if value.shape != spec['shape'] or str(value.dtype) not in allowed:
                raise ValueError('Original cached array shape/dtype differs: '+name)
            if value.dtype.kind == 'f' and not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite original cached array: '+name)
            if name == 'query_indices' and (value.min() < 0 or value.max() >= 64**3):
                raise ValueError('Original stencil index outside R64')
            self.arrays[path] = torch.as_tensor(np.array(value, copy=True), device=self.device)
            self.receipt['actual_tensor_bytes'] += self.arrays[path].numel()*self.arrays[path].element_size()
            if self.on_read:
                self.on_read('completed', name, spec['identity'])
            del value
            self.resource_check(torch)
        for key, fields in self.case_specs.items():
            case = {name: self.arrays[path] for name, path in fields.items()}
            case.update(key=key, scientific_group='TRAIN')
            case['cloud'] = case['cloud'].reshape(1, 1024, 3)
            case['mask'] = case['mask'].reshape(1, 1024)
            if self.family == 'FNO9':
                case['grid'] = case['grid'].reshape(1, 29, 64, 64, 64)
            self.cases[key] = case
            self.receipt['loaded_cases'] += 1
        self.resource_check(torch)
        self.completed = True
        return self

    def case(self, key):
        if key not in self.case_specs:
            raise PermissionError('Case is not in this fixed TRAIN scope')
        if not self.completed:
            raise RuntimeError('Incomplete resident cache must not train a subset')
        return self.cases[key]

    def metadata_profile(self):
        return dict(family=self.family, cases=len(self.keys), unique_files=len(self.specs),
                    disk_bytes_with_headers_upper=self.disk_bytes_upper,
                    memory_limit_bytes=self.memory_limit_bytes, numeric_bodies_read=self.receipt['input_array_reads']+self.receipt['encoded_TRAIN_target_reads'],
                    physical_DEV_FINAL_read=False,
                    actual_GPU_cache_loaded=self.completed and str(self.device).startswith('cuda'),
                    actual_model_gradient_resource_qualified=False)


__all__ = ['TRAINResidentCache']
