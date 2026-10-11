"""One metadata plus synthetic CPU cache probe; no original numerical arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    if not parser.parse_args().probe:
        print('Source-only; --probe tests metadata and self-generated CPU arrays.')
        return
    root = Path(__file__).resolve().parents[1]
    output = root/'notes/geobc_train_cache_CPU_probe.json'
    if output.exists():
        raise RuntimeError('Do not repeat/overwrite an existing CPU attempt')
    entry = (time.time(), time.monotonic())
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('CPU120s budget')))
    signal.alarm(120)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    record = dict(status='running', CPU_contract_passed=False, original_numeric_array_GT_PT_reads=0,
                  original_metadata_reads=1, actual_GPU_reads_or_allocations=0,
                  actual_new_training_updates=0, actual_GPU_resource_qualified=False,
                  metadata_profiles={}, synthetic_checks={}, failure=None)
    try:
        from geobc import available_cad_pilot_data as d
        from geobc.train_resident_cache import TRAINResidentCache, FAMILIES
        from geobc.prospective_dataset import SCHEMA
        manifest = d.read('data/geobc_prospective_dataset_v1/manifest.json')
        imports_before = set(sys.modules)
        for scope in (40, 100):
            for family in FAMILIES:
                cache = TRAINResidentCache(manifest, scope, family, device='cpu')
                record['metadata_profiles'][str(scope)+'_'+family] = cache.metadata_profile()
        try:
            TRAINResidentCache(manifest, 40, 'FNO9', memory_limit_bytes=1)
            raise AssertionError('Impossible cache budget accepted')
        except MemoryError:
            pass
        assert not ({'numpy', 'torch'} & (set(sys.modules)-imports_before))
        record['metadata_and_impossible_budget_before_array_library_imports'] = True
        import numpy as np
        import torch
        torch.set_num_threads(4)
        rng = np.random.default_rng(2026)
        with tempfile.TemporaryDirectory(prefix='synthetic_cache_', dir=root/'runs/geobc_exploration') as scratch:
            scratch = Path(scratch)
            n = 23
            values = dict(xyz=rng.random((n, 3)).astype('float32'),
                          features=rng.normal(size=(n, 25)).astype('float32'),
                          cloud=rng.random((1024, 3)).astype('float32'), mask=np.ones(1024, dtype='bool'),
                          grid=np.zeros((29, 64, 64, 64), dtype='float32'),
                          query_indices32=np.zeros((n, 8), dtype='int32'),
                          query_indices64=np.zeros((n, 8), dtype='int64'),
                          query_weights=np.full((n, 8), .125, dtype='float32'))
            for load, sign in [('ver', 1.), ('hor', -1.), ('tor', .5)]:
                values['target_'+load] = (sign*rng.normal(size=(n, 4))).astype('float32')
            paths = {}
            for name, value in values.items():
                path = scratch/(name+'.npy')
                np.save(path, value, allow_pickle=False)
                paths[name] = d.identity(path)
            fixed_keys = [f'syntheticCAD{i}/{load}' for i in range(40) for load in ('ver', 'hor', 'tor')]
            synthetic = dict(schema=SCHEMA, status='completed', feasibility_pass=True,
                             train_keys={'40': fixed_keys}, records={}, case_roles={})
            for key in fixed_keys:
                cad, load = key.split('/')
                rec = dict(key=key, cad_id=cad, scientific_group='TRAIN', status='completed',
                           feasibility_pass=True, num_nodes=n, file_identities={})
                for name in ('xyz', 'features', 'cloud', 'mask', 'grid', 'query_indices', 'query_weights', 'target'):
                    chosen = 'target_'+load if name == 'target' else 'query_indices32' if name == 'query_indices' and load == 'ver' else 'query_indices64' if name == 'query_indices' else name
                    rec[name+'_path'] = paths[chosen]['path']
                    rec['file_identities'][name+'_path'] = paths[chosen]
                # Deliberately nonexistent: physical targets must never open.
                rec['physical_path'] = 'runs/forbidden_synthetic_physical_not_created.npy'
                rec['file_identities']['physical_path'] = dict(path=rec['physical_path'], bytes=0, mtime_ns=0)
                synthetic['records'][key] = rec
                synthetic['case_roles'][key] = 'TRAIN'
            for family in FAMILIES:
                calls = []
                cache = TRAINResidentCache(synthetic, 40, family, device='cpu',
                                           on_read=lambda *args: calls.append(args[:2]))
                try:
                    cache.case(fixed_keys[0])
                    raise AssertionError('Partial cache exposed a training case')
                except RuntimeError:
                    pass
                cache.load()
                assert cache.receipt['loaded_cases'] == 120
                assert cache.receipt['physical_TRAIN_target_reads'] == 0
                assert cache.receipt['encoded_TRAIN_target_reads'] == 3
                assert len(calls) == 2*len(cache.specs)
                assert cache.case('syntheticCAD0/ver')['xyz'].data_ptr() == cache.case('syntheticCAD1/hor')['xyz'].data_ptr()
                for load in ('ver', 'hor', 'tor'):
                    got = cache.case('syntheticCAD0/'+load)['target']
                    assert torch.equal(got, torch.from_numpy(values['target_'+load]))
                if family == 'FNO9':
                    assert cache.case('syntheticCAD0/ver')['query_indices'].dtype == torch.int32
                    assert cache.case('syntheticCAD0/hor')['query_indices'].dtype == torch.int64
                else:
                    assert 'grid' not in cache.case(fixed_keys[0])
                for forbidden in ('syntheticDEV/ver', 'syntheticFINAL/hor', 'newCAD/tor'):
                    try:
                        cache.case(forbidden)
                        raise AssertionError('Out-of-scope cache query permitted')
                    except PermissionError:
                        pass
                try:
                    cache.load()
                    raise AssertionError('Cache replay permitted')
                except RuntimeError:
                    pass
                assert cache.receipt['actual_tensor_bytes'] <= cache.disk_bytes_upper
                record['synthetic_checks'][family] = dict(receipt=cache.receipt, unique_files=len(cache.specs),
                    shared_geometry_aliases=True, original_signed_fixture_targets_exact=True,
                    premature_partial_cache_rejected=True, out_of_scope_rejections=3, replay_rejected=True)
                del cache
        record.update(status='completed', CPU_contract_passed=True, CUDA_initialized=torch.cuda.is_initialized())
    except Exception as error:
        record.update(status='failed', failure=type(error).__name__+': '+str(error))
        raise
    finally:
        record.update(epoch_seconds=time.time()-entry[0], monotonic_seconds=time.monotonic()-entry[1])
        record['budget_passed'] = max(record['epoch_seconds'], record['monotonic_seconds']) <= 120
        names = ['geobc/train_resident_cache.py', 'tools/probe_train_cache_cpu.py',
                 'geobc/prospective_dataset.py', 'geobc/available_cad_pilot_data.py']
        record['source_sha256'] = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in names}
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        signal.alarm(0)
    print(json.dumps({k: record[k] for k in ['status', 'CPU_contract_passed', 'epoch_seconds', 'CUDA_initialized']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
