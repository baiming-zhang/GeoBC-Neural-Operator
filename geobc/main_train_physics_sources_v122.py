"""Target-free main TRAIN source qualification; reused positive caches stay input-only."""
from . import available_cad_pilot_data as d

def read_cache(np,item,guard):
    guard();expected={k:v for k,v in item.items() if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
    assert d.identity(item['path'],hashed=True)==expected
    with np.load(d.path(item['path']),allow_pickle=False) as source:
        arrays={k:source[k] for k in item['array_schema']}
    from .elastic_pilot_data import _array_sha
    for k,s in item['array_schema'].items():
        v=arrays[k];assert list(v.shape)==s['shape'] and str(v.dtype)==s['dtype'] and _array_sha(v,v.dtype)==s['sha256'] and np.isfinite(v).all()
    return arrays

def prepare(np,cad,row,science,a,guard,began):
    import h5py,time
    from . import elastic_pilot_data as old,coverage_comparison_data as coverage
    recipe=science['physics_recipe'];sources=row['raw_source_stat_identities'];reuse=science['legacy_inputs'].get(cad)
    for item in sources.values():d.stat_matches(item)
    deck_identity=d.identity(sources['SolverDeck']['path'],hashed=True);a['deck_hash_reads']+=1;guard()
    if reuse:
        arrays=read_cache(np,reuse['cache'],guard);a['legacy_cache_reads']+=1
        vertices=arrays['vertices_mm'];cells=arrays['cells10'];ids=arrays['grid_node_ids']
        facts=reuse['facts'];expected=dict(nodes=len(vertices),cells=len(cells),characteristic_length_mm=float(arrays['characteristic_length_mm']))
    else:
        h5_identity=d.identity(sources['FieldMesh']['path'],hashed=True);a['H5_hash_reads']+=1;guard()
        with h5py.File(d.path(h5_identity['path']),'r') as f:
            vertices=f['vertices'][...];cells=f['cells'][...]
        a['H5_input_reads']+=1
        assert vertices.dtype==np.float32 and vertices.shape==(row['nodes'],3) and np.isfinite(vertices).all()
        assert cells.ndim==2 and cells.shape[1]==10 and cells.dtype.kind in 'iu' and cells.min()>=0 and cells.max()<len(vertices)
        cells=np.asarray(cells,np.int64)
        descriptor=row['canonical_GRID_input_descriptor']
        if row['old_TRAIN40']:
            path=descriptor['path'];archive_identity=d.identity(path,hashed=True);a['GRID_archive_hash_reads']+=1
            with np.load(d.path(path),allow_pickle=False) as source:ids=source['grid_node_ids']
            a['GRID_archive_member_reads']+=1;mapping=dict(identity=archive_identity,member='grid_node_ids',no_target_member=True)
        else:
            perm_record=descriptor['permutation_identity'];expected_id={k:v for k,v in perm_record.items() if k in ('path','bytes','mtime_ns','sha256')};expected_id['path']=d.relative(d.path(perm_record['path']))
            assert d.identity(perm_record['path'],hashed=True)==expected_id;permutation=np.load(d.path(perm_record['path']),allow_pickle=False);a['permutation_reads']+=1
            assert permutation.dtype==np.int64 and permutation.shape==(len(vertices),) and np.array_equal(np.sort(permutation),np.arange(len(vertices)))
            csvitem=descriptor['source_csv_identity'];d.stat_matches(csvitem);csvidentity=d.identity(csvitem['path'],hashed=True);a['CSV_hash_reads']+=1
            columns=coverage._columns(csvitem['path']);a['CSV_header_reads']+=1
            assert all(k in columns for k in coverage.COORDS)
            coords=np.loadtxt(d.path(csvitem['path']),delimiter=',',skiprows=1,usecols=[columns.index(k) for k in coverage.COORDS],dtype=np.float64,ndmin=2);a['CSV_GRID_XYZ_reads']+=1
            assert coords.shape==(len(vertices),4) and np.isfinite(coords).all() and (coords[:,0]>0).all() and (coords[:,0]<2**53).all() and np.array_equal(coords[:,0],np.rint(coords[:,0]))
            aligned=coords[permutation];distance=np.linalg.norm(aligned[:,1:]-vertices,axis=1)
            if distance.max()>.0002:raise old.UnsupportedInput('Existing GRID/XYZ permutation does not align original H5 rows; no remapping')
            ids=aligned[:,0].astype(np.int64);mapping=dict(permutation_identity=expected_id,CSV_identity=csvidentity,numeric_columns=coverage.COORDS,maximum_distance_mm=float(distance.max()),no_U_S_columns=True)
        assert ids.dtype==np.int64 and ids.shape==(len(vertices),) and np.unique(ids).size==len(ids) and (ids>0).all()
        expected=dict(nodes=len(vertices),cells=len(cells),characteristic_length_mm=float((vertices.max(0)-vertices.min(0)).max()))
    guard();deck=old._parse_deck(d.path(deck_identity['path']),recipe,began,300);a['deck_parses']+=1
    generator=old._cards(d.path(deck_identity['path']),recipe['reader_scope'],began,300)
    try:controls=next(generator)[3]
    finally:generator.close()
    a['deck_header_reads']+=1
    selected=[coverage._deck_case(deck,controls,label) for label in ('ver','hor','tor')]
    constraints=[sorted((int(x[0]),''.join(sorted(x[1])),float(x[2])) for x in case['active_constraints']) for case in selected]
    if not constraints[0] or not all(x==constraints[0] for x in constraints) or any(x[2]!=0 for x in constraints[0]):raise old.UnsupportedInput('Canonical load source constraints differ or are nonzero')
    if not reuse:
        arrays,facts=old._build(selected[0],vertices.astype(np.float64),cells,ids,dict(recipe=recipe,expected_geometry=expected),began,300)
        source_info=dict(H5_identity=h5_identity,mapping=mapping,new_geometry_input_build=True)
    else:
        assert len(vertices)==row['nodes'] and facts['allN_partition_exact'] and facts['load_guard']['passed']
        assert sorted(deck['rbe2'][key]['reference'] for key in deck['rbe2'])==facts['constraints']['support_reference_GRID_IDs']
        source_info=dict(legacy_input_manifest_identity=reuse['manifest_identity'],legacy_cache_identity={k:v for k,v in reuse['cache'].items() if k!='array_schema'},new_H5_GRID_geometry_builds=0,
            inherited_geometry_qualification=True,old_solve_and_operator_replayed=False)
    rbe=next(iter(deck['rbe3'].values()));ref=rbe['reference'];H=arrays['rbe3_Hbar'];length=float(arrays['rbe3_length_mm']);members=arrays['rbe3_member_node_ids'];offset=arrays['vertices_mm'][members]-arrays['reference_xyz_mm'];forces=[];loads={}
    for label,case in zip(('ver','hor','tor'),selected):
        guard();v=np.zeros(6,np.float64)
        for action in case['active_actions']:
            if action['node']!=ref or action['kind'] not in ('FORCE','MOMENT'):raise old.UnsupportedInput('Canonical load location/type unsupported')
            j=0 if action['kind']=='FORCE' else 3;v[j:j+3]+=action['vector']
        if not np.isfinite(v).all() or np.linalg.norm(v)==0 or any(x!=0 and dof not in rbe['REFC'] for dof,x in zip('123456',v)):raise old.UnsupportedInput('Canonical force outside source RBE reference DOFs')
        scaled=np.r_[v[:3],v[3:]/length];transfer=(H.T@scaled).reshape(-1,3);force=np.zeros_like(arrays['force']);np.add.at(force,members,transfer)
        ferr=float(np.max(np.abs(transfer.sum(0)-v[:3]))/max(1.,np.linalg.norm(scaled)));merr=float(np.max(np.abs(np.cross(offset,transfer).sum(0)-v[3:]))/max(1.,length*np.linalg.norm(scaled)))
        w=np.random.default_rng(2026).standard_normal(transfer.size);work=float(abs(w@(H.T@scaled)-(H@w)@scaled)/max(1.,np.linalg.norm(w)*np.linalg.norm(H.T@scaled)))
        if max(ferr,merr,work)>recipe['load_transfer']['relative_guard_tolerance']:raise old.UnsupportedInput('Actual source force/moment/work transfer fails')
        forces.append(force);loads[label]=dict(reference_load6=v.tolist(),active_actions=case['active_actions'],selected_controls=case['selected_controls'],force_relative_error=ferr,moment_relative_error=merr,virtual_work_relative_error=work)
    if not np.array_equal(forces[0],arrays['force']):raise old.UnsupportedInput('Current source VER differs from accepted original cache/input build')
    assert all(np.isfinite(f).all() and np.linalg.norm(f[arrays['free_node_ids']])>0 for f in forces)
    assert max(time.monotonic()-began,0)<=300
    return arrays,np.stack(forces),dict(expected_geometry=expected,source_info=source_info,deck_identity=deck_identity,full_card_inventory=deck['cards'],source_facts=facts,canonical_LOADs=loads,zero_same_constraints=True,scientific_group='TRAIN',raw_targets_read=0,actual_K_GPU_qualification=False)
