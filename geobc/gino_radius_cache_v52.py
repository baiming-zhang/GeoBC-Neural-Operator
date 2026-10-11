"""Current TRAIN100 geometry adapter over the validated complete-CSR backend.

Old40 completed caches are read in place; new60 geometries use a separate
directory and the same exact GPU radius builder. No arrays are opened by the
constructor. The outer released driver must provide deadlines/resources/lock.
"""
from .gino_train import RadiusCache
from . import available_cad_pilot_data as d


class CurrentTRAINRadiusCache(RadiusCache):
    def __init__(self, plan, contract, torch, np, device, costs, progress,
                 geometry_inventory):
        rows = geometry_inventory['per_CAD']
        self.current_rows = {r['cad']:r for r in rows}
        if len(self.current_rows) != 100 or set(self.current_rows) != set(contract['geometries']):
            raise ValueError('Exact fixed100 TRAIN geometries required')
        if plan['radius_cache'] != 'data/geobc_gino_radius_train100_v52':
            raise ValueError('New CSR construction must use the separate current directory')
        for cad, row in self.current_rows.items():
            expected = contract['geometries'][cad]
            item = row['current_xyz_identity']
            d.stat_matches(item)
            if d.path(expected['xyz_path']).resolve() != d.path(item['path']).resolve():
                raise ValueError('Current original coordinate path differs')
            if expected['xyz_stat'] != {k:item[k] for k in ('bytes','mtime_ns')}:
                raise ValueError('Original coordinate stat differs')
        super().__init__(plan, contract, torch, np, device, costs, progress)

    def _inventory(self, cad, xyz):
        row = self.current_rows[cad]
        if not row['existing_CSR_reuse_metadata_passed']:
            return super()._inventory(cad, xyz)
        witness = row['CSR_manifest_identity']
        if d.identity(d.path(witness['path']), hashed=True) != witness:
            raise ValueError('Existing complete CSR metadata witness changed')
        saved = d.read(witness['path'])
        expected = self.contract['geometries'][cad]
        if cad == '101_479' and 'geometry' not in saved:
            source = saved['config']['contract']['source']
            if (saved['status'] != 'completed' or saved.get('feasible') is not True
                    or source['stat'] != expected['xyz_stat']
                    or d.path(source['xyz_path']).resolve() != d.path(expected['xyz_path']).resolve()
                    or saved['stats']['construction']['R'] != 64
                    or saved['stats']['construction']['radius'] != .033):
                raise ValueError('Original target-free pilot differs')
            saved = dict(status='completed', geometry=expected, radius=.033,
                resolution=64, inventory=saved['inventory'], checks=saved['checks'],
                construction=saved['stats']['construction'], origin=witness['path'],
                cache_bytes=saved['cache_bytes'],
                costs={'build_seconds':0., 'save_seconds':0.},
                cost_scope='Current reuse only; old construction cost remains in old receipt')
        if (saved['status'] != 'completed' or saved['geometry'] != expected
                or saved['radius'] != .033 or saved['resolution'] != 64
                or len(saved['inventory']) != 8):
            raise ValueError('Complete old geometry cache identity differs')
        return saved, False

    def attach(self, case):
        # Resident case dictionaries must never accumulate100 device CSRs.
        # Parent keeps one active CAD; only the returned temporary case gains
        # its complete CSR references, released by the driver after each step.
        if case.get('scientific_group') != 'TRAIN' or case['key'].split('/')[0] not in self.current_rows:
            raise PermissionError('Only fixed current TRAIN geometry is allowed')
        return super().attach(dict(case))
