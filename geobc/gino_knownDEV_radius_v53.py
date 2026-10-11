"""Explicit fixed20 knownDEV input-only CSR scope, distinct from TRAIN100."""
from .gino_train import RadiusCache
from . import available_cad_pilot_data as d


class KnownDEVRadiusCache(RadiusCache):
    def __init__(self, plan, contract, torch, np, device, costs, progress):
        if getattr(device,'type',None) != 'cuda':
            raise TypeError('Old complete-CSR backend requires torch.device(cuda)')
        if plan['radius_cache'] != 'data/geobc_gino_radius_DEV20_v53':
            raise ValueError('Separate fixed knownDEV geometry directory required')
        if len(contract['geometries']) != 20 or '101_479' in contract['geometries']:
            raise ValueError('Exact disjoint20 DEV geometries, never the TRAIN pilot')
        self.allowed_keys = set(contract['case_keys'])
        if len(self.allowed_keys) != 60:
            raise ValueError('All three loads of every fixed DEV CAD required')
        for cad,geometry in contract['geometries'].items():
            if {cad+'/'+load for load in ('ver','hor','tor')} - self.allowed_keys:
                raise ValueError('KnownDEV Cartesian case scope differs')
            d.stat_matches(dict(path=geometry['xyz_path'],**geometry['xyz_stat']))
        super().__init__(plan,contract,torch,np,device,costs,progress)

    def attach(self, case):
        if case.get('scientific_group') != 'DEV' or case.get('key') not in self.allowed_keys:
            raise PermissionError('Explicit fixed knownDEV role/key required')
        if {'target','physical'} & set(case):
            raise PermissionError('Target bodies never belong in prediction inputs')
        return super().attach(dict(case))
