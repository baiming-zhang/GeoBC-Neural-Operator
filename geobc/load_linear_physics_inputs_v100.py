"""Target-free selected CAD geometry and six real reduced loading vectors."""
from . import available_cad_pilot_data as d


def prepare_inputs(np,science,a,guard,setup_began):
    import time
    import h5py
    from . import elastic_pilot_data as old,coverage_comparison_data as coverage
    cad=science['selection']['CAD'];sources=science['source_input_identities'];recipe=science['old_accepted_physics_recipe'];expected=science['expected_geometry']
    for item in sources.values():
        guard();assert d.identity(item['path'],hashed=True)==item;a['source_body_hash_reads']+=1
    item=sources['FieldMesh'];a['H5_input_attempts']+=1
    with h5py.File(d.path(item['path']),'r')as f:
        vertices=f['vertices'][...];cells=f['cells'][...]
    a['H5_input_reads']+=1;a['H5_members_read']=['vertices','cells']
    assert vertices.shape==(expected['nodes'],3)and vertices.dtype==np.float32 and np.isfinite(vertices).all()
    assert cells.shape==(expected['cells'],10)and cells.dtype.kind in 'iu'and cells.min()>=0 and cells.max()<len(vertices)
    cells=np.asarray(cells,np.int64);gid=science['GRID_map_identity'];guard()
    actual=d.identity(gid['path'],hashed=True);assert actual=={k:v for k,v in gid.items()if k in ('path','bytes','mtime_ns','sha256')}
    ids=np.load(d.path(gid['path']),allow_pickle=False);a['GRID_map_reads']+=1
    assert ids.shape==(len(vertices),)and ids.dtype==np.int64 and np.unique(ids).size==len(ids)
    length=float((vertices.max(0)-vertices.min(0)).max());assert length==expected['characteristic_length_mm']
    deck=old._parse_deck(d.path(sources['SolverDeck']['path']),recipe,setup_began,300);a['deck_parses']+=1
    gen=old._cards(d.path(sources['SolverDeck']['path']),recipe['reader_scope'],setup_began,300)
    try:controls=next(gen)[3]
    finally:gen.close()
    a['deck_header_reads']+=1
    selected=[coverage._deck_case(deck,controls,l)for l in ('ver','hor','tor')]
    cons=[sorted((int(x[0]),''.join(sorted(x[1])),float(x[2]))for x in c['active_constraints'])for c in selected]
    if not cons[0]or not all(c==cons[0]for c in cons)or any(x[2]!=0. for x in cons[0]):raise old.UnsupportedInput('Canonical zero supports differ')
    record=dict(recipe=recipe,expected_geometry=expected)
    # Existing complete geometry/material/constraint/load mathematics built once.
    arrays,facts=old._build(selected[0],vertices.astype(np.float64),cells,ids,record,setup_began,300)
    rbe=next(iter(deck['rbe3'].values()));ref=rbe['reference'];H=arrays['rbe3_Hbar'];fit=float(arrays['rbe3_length_mm']);members=arrays['rbe3_member_node_ids'];offset=arrays['vertices_mm'][members]-arrays['reference_xyz_mm']
    forces=[];load_records={}
    for label,case in zip(('ver','hor','tor'),selected):
        guard();v=np.zeros(6,np.float64)
        for action in case['active_actions']:
            if action['node']!=ref or action['kind']not in ('FORCE','MOMENT'):raise old.UnsupportedInput('Unsupported loaded location/type')
            j=0 if action['kind']=='FORCE'else 3;v[j:j+3]+=action['vector']
        if not np.isfinite(v).all()or np.linalg.norm(v)==0 or any(x!=0 and dof not in rbe['REFC']for dof,x in zip('123456',v)):raise old.UnsupportedInput('RBE3 action outside accepted REFC')
        scaled=np.concatenate((v[:3],v[3:]/fit));trans=(H.T@scaled).reshape(-1,3)
        force=np.zeros_like(arrays['force']);np.add.at(force,members,trans)
        ferr=float(np.max(np.abs(trans.sum(0)-v[:3]))/max(1.,np.linalg.norm(scaled)))
        merr=float(np.max(np.abs(np.cross(offset,trans).sum(0)-v[3:]))/max(1.,fit*np.linalg.norm(scaled)))
        witness=np.random.default_rng(2026).standard_normal(trans.size)
        work=float(abs(witness@(H.T@scaled)-(H@witness)@scaled)/max(1.,np.linalg.norm(witness)*np.linalg.norm(H.T@scaled)))
        if max(ferr,merr,work)>recipe['load_transfer']['relative_guard_tolerance']:raise old.UnsupportedInput('Canonical force/moment/work transfer fails')
        forces.append(force);load_records[label]=dict(active_actions=case['active_actions'],selected_controls=case['selected_controls'],reference_load6=v.tolist(),force_relative_error=ferr,moment_relative_error=merr,virtual_work_relative_error=work)
    assert np.array_equal(forces[0],arrays['force'])
    for coeff in science['selection']['mixed_coefficients']:forces.append(sum(coeff[j]*forces[j]for j in range(3)))
    assert len(forces)==6 and all(np.isfinite(x).all()and np.linalg.norm(x[arrays['free_node_ids']])>0 for x in forces)
    a['physics_input_facts']=dict(original=facts,all_three_source_LOAD_expansion=load_records,full_card_inventory=deck['cards'],zero_same_constraints=True,GRID_order='Inherited CSV-aligned geometry IDs;actual full deck coordinates/CTETRA coverage checked',independent_approx_P2_not_original_OptiStruct=True)
    guard();return arrays,np.stack(forces),record
