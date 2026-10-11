"""Future native-TS macro storage; TRAIN-only targets, no unused encoded reads.

This is a cache implementation, not an experiment release. The old native++
layout supplies the same four target-free input members and physical labels;
no ++ model is constructed. Actual model family is recorded separately.
"""
from .train_native_physical_cache_v13 import NativePhysicalTRAINCache

FAMILY = 'TS_ORIGINAL_BF16_NATIVE_HASH'


class NativePhysicalTargetOnlyCache(NativePhysicalTRAINCache):
    def __init__(self, manifest, train_CAD_count, family, **kwargs):
        if family != FAMILY or train_CAD_count != 100:
            raise ValueError('Only the prospective native-originalTS100 storage scope')
        super().__init__(manifest, train_CAD_count, 'PP256_NATIVE', **kwargs)
        self.model_family = family
        unused = [path for path, spec in self.specs.items() if spec['name'] == 'target']
        self.excluded_encoded_TRAIN_target_files = len(unused)
        for path in unused:
            del self.specs[path]
        for fields in self.case_specs.values():
            del fields['target']
        self.disk_bytes_upper = sum(x['identity']['bytes'] for x in self.specs.values())
        self.receipt['excluded_encoded_TRAIN_target_files_before_load'] = len(unused)
        self.receipt['grid_input_array_reads'] = 0
        if self.disk_bytes_upper + sum(x['identity']['bytes'] for x in self.physical_specs.values()) > self.memory_limit_bytes:
            raise MemoryError('Complete native inputs and physical targets exceed fixed budget')

    def load(self):
        super().load()
        if (self.receipt['encoded_TRAIN_target_read_attempts'] or
                self.receipt['encoded_TRAIN_target_reads'] or
                any(set(case)-{'key','scientific_group','xyz','features','cloud','mask'}
                    for case in self.cases.values())):
            raise RuntimeError('Unexpected labels or grid members in native input cache')
        return self

    def metadata_profile(self):
        result = super().metadata_profile()
        result.update(family=self.model_family,
            storage_layout_family='PP256_NATIVE input layout only; no PP model',
            excluded_encoded_TRAIN_target_files_before_load=self.excluded_encoded_TRAIN_target_files,
            numeric_encoded_targets_requested=False, R64_input_members_requested=False,
            actual_model_gradient_resource_qualified=False)
        return result
