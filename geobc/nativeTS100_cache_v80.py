"""Existing deduplicated TRAIN cache: native inputs and physical targets only."""
from .train_resident_cache import TRAINResidentCache
from .train_physical_cache_v12 import PhysicalTRAINCache
from .prospective_dataset import resolve_case


class NativePhysicalCache(PhysicalTRAINCache):
    def __init__(self, manifest, **kwargs):
        # Storage family selects only xyz/features/cloud/mask. No model alias.
        TRAINResidentCache.__init__(self, manifest, 100, 'HASH_PN', **kwargs)
        removed = {fields.pop('target') for fields in self.case_specs.values()}
        assert all(self.specs[p]['name'] == 'target' for p in removed)
        for p in removed:
            del self.specs[p]
        assert all(z['name'] in ('xyz', 'features', 'cloud', 'mask') for z in self.specs.values())
        self.disk_bytes_upper = sum(z['identity']['bytes'] for z in self.specs.values())
        self.physical_specs = {}
        self.physical_targets = {}; self.case_norms = {}
        self.receipt.update(physical_TRAIN_target_read_attempts=0,
            physical_TRAIN_loaded_cases=0, physical_TRAIN_tensor_bytes=0,
            unused_encoded_target_files_removed_before_load=len(removed), R64_input_array_reads=0)
        for key in self.keys:
            record, identities = resolve_case(manifest, key, True, 100, False)
            self.physical_specs[key] = dict(identity=identities['physical'], shape=(record['num_nodes'], 4))
        if self.disk_bytes_upper + sum(z['identity']['bytes'] for z in self.physical_specs.values()) > self.memory_limit_bytes:
            raise MemoryError('Native input and physical target cache exceeds budget before array reads')

    def load(self):
        super().load()
        assert self.receipt['encoded_TRAIN_target_reads'] == self.receipt['encoded_TRAIN_target_read_attempts'] == 0
        assert all(not {'target', 'grid', 'query_indices', 'query_weights'} & set(c) for c in self.cases.values())
        return self
