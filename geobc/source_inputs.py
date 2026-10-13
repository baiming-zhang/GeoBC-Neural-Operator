"""Build canonical physics inputs from geometry and GRID/XYZ columns only."""
import csv
import time
import numpy as np
from .deck import _parse_deck,_cards,_deck_case,_build,UnsupportedInput
from .physics_inputs import local_path

COORDINATES=['nodeID','coord_x(mm)','coord_y(mm)','coord_z(mm)']


def prepare_source(record,recipe,guard,counts,seconds=300):
    import h5py
    began=time.monotonic()
    guard()
    counts['H5_attempts']+=1
    with h5py.File(local_path(record['mesh']),'r') as file:
        vertices=file['vertices'][...];cells=file['cells'][...]
    counts['H5_inputs']+=1
    if vertices.dtype!=np.float32 or vertices.shape!=(record['nodes'],3) or not np.isfinite(vertices).all():
        raise ValueError('Original geometry vertices differ')
    if cells.ndim!=2 or cells.shape[1]!=10 or cells.dtype.kind not in 'iu' or cells.min()<0 or cells.max()>=len(vertices):
        raise ValueError('Original TET10 connectivity differs')
    cells=np.asarray(cells,np.int64)
    permutation=np.load(local_path(record['permutation']),allow_pickle=False);counts['permutation_reads']+=1
    if permutation.dtype!=np.int64 or permutation.shape!=(len(vertices),) or not np.array_equal(np.sort(permutation),np.arange(len(vertices))):
        raise ValueError('Existing row permutation differs')
    path=local_path(record['coordinates'])
    with path.open(encoding='utf-8-sig',newline='') as file:
        header=next(csv.reader(file))
    columns=[header.index(name) for name in COORDINATES]
    counts['coordinate_attempts']+=1
    coordinates=np.loadtxt(path,delimiter=',',skiprows=1,usecols=columns,dtype=np.float64,ndmin=2)
    counts['GRID_XYZ_inputs']+=1
    if coordinates.shape!=(len(vertices),4) or not np.isfinite(coordinates).all():
        raise ValueError('GRID/XYZ dimensions differ')
    ids=coordinates[:,0]
    if not ((ids>0)&(ids<2**53)).all() or not np.array_equal(ids,np.rint(ids)) or len(np.unique(ids))!=len(ids):
        raise ValueError('Invalid GRID identifiers')
    aligned=coordinates[permutation]
    distance=np.linalg.norm(aligned[:,1:]-vertices,axis=1)
    if float(distance.max())>.0002:
        raise UnsupportedInput('Existing GRID/XYZ mapping fails; no new mapping fitted')
    ids=aligned[:,0].astype(np.int64)
    guard()
    deck=_parse_deck(local_path(record['deck']),recipe,began,seconds);counts['deck_parses']+=1
    header_pass=_cards(local_path(record['deck']),recipe['reader_scope'],began,seconds)
    try:controls=next(header_pass)[3]
    finally:header_pass.close()
    selected=[_deck_case(deck,controls,load) for load in ['ver','hor','tor']]
    constraints=[sorted((int(v[0]),''.join(sorted(v[1])),float(v[2])) for v in case['active_constraints']) for case in selected]
    if not constraints[0] or any(v!=constraints[0] for v in constraints) or any(v[2]!=0 for v in constraints[0]):
        raise UnsupportedInput('Canonical loads need identical zero constraints')
    expected=dict(nodes=len(vertices),cells=len(cells),characteristic_length_mm=float((vertices.max(0)-vertices.min(0)).max()))
    arrays,facts=_build(selected[0],vertices.astype(np.float64),cells,ids,dict(recipe=recipe,expected_geometry=expected),began,seconds)
    H=arrays['rbe3_Hbar'];length=float(arrays['rbe3_length_mm']);members=arrays['rbe3_member_node_ids']
    offset=arrays['vertices_mm'][members]-arrays['reference_xyz_mm']
    rbe=next(iter(deck['rbe3'].values()));forces=[];loads={}
    for load,case in zip(['ver','hor','tor'],selected):
        guard();force6=np.zeros(6,np.float64)
        for action in case['active_actions']:
            if action['node']!=rbe['reference'] or action['kind'] not in ['FORCE','MOMENT']:
                raise UnsupportedInput('Canonical load location or type differs')
            start=0 if action['kind']=='FORCE' else 3
            force6[start:start+3]+=action['vector']
        if not np.isfinite(force6).all() or np.linalg.norm(force6)==0 or any(v!=0 and d not in rbe['REFC'] for d,v in zip('123456',force6)):
            raise UnsupportedInput('Canonical force outside actual reference DOFs')
        scaled=np.r_[force6[:3],force6[3:]/length]
        transferred=(H.T@scaled).reshape(-1,3)
        nodal=np.zeros_like(arrays['force']);np.add.at(nodal,members,transferred)
        ferr=float(np.max(np.abs(transferred.sum(0)-force6[:3]))/max(1.,np.linalg.norm(scaled)))
        merr=float(np.max(np.abs(np.cross(offset,transferred).sum(0)-force6[3:]))/max(1.,length*np.linalg.norm(scaled)))
        witness=np.random.default_rng(2026).standard_normal(transferred.size)
        work=float(abs(witness@(H.T@scaled)-(H@witness)@scaled)/max(1.,np.linalg.norm(witness)*np.linalg.norm(H.T@scaled)))
        if max(ferr,merr,work)>recipe['load_transfer']['relative_guard_tolerance']:
            raise UnsupportedInput('Source force/moment/work transfer fails')
        if not np.isfinite(nodal).all() or np.linalg.norm(nodal[arrays['free_node_ids']])==0:
            raise UnsupportedInput('Canonical free load is empty or nonfinite')
        forces.append(nodal)
        loads[load]=dict(reference_load6=force6.tolist(),force_relative_error=ferr,moment_relative_error=merr,virtual_work_relative_error=work)
    if not np.array_equal(forces[0],arrays['force']):
        raise UnsupportedInput('VER source construction differs')
    arrays['canonical_forces']=np.stack(forces)
    guard()
    if time.monotonic()-began>seconds:raise TimeoutError('Source preparation allowance')
    return arrays,dict(expected_geometry=expected,source_facts=facts,canonical_LOADs=loads,GRID_XYZ_maximum_distance_mm=float(distance.max()))
