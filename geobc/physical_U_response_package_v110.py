"""One strict manifest-to-response loader; original scientific arrays stay local.

The caller must supply the reviewed payload digest, not a self-claimed mutable
descriptor. No defaults import target loaders or open labels/checkpoints.
"""

def validate_metadata(d,package,expected_digest):
    payload=package['payload']
    if package['payload_digest']!=expected_digest or d.digest(payload)!=expected_digest:
        raise ValueError('Unreviewed or changed response package metadata')
    assert package['schema']=='GeoBC_physical_U_response_package_v1'
    assert payload['kind']=='independent_approximate_P2_physical_U_force6'
    assert payload['basis_order']==['ver','hor','tor','Fy','Mx','My']
    assert payload['scope']['REFC']=='123456' and payload['scope']['material_constraints_fixed']
    assert payload['scope']['physical_U_units']=='mm' and payload['scope']['force_units']=='N' and payload['scope']['moment_units']=='Nmm'
    assert payload['new_OptiStruct_GT_accuracy']is False and payload['signedS_provided']is False
    d.stat_matches(payload['geometry_identity'])
    n=payload['geometry_schema']['nodes'];assert n>0 and payload['geometry_schema']['cells']>0
    for name in payload['basis_order']:
        expected=payload['basis_records'][name]
        assert d.identity(expected['manifest_identity']['path'],hashed=True)==expected['manifest_identity']
        rec=d.read(expected['manifest_identity']['path'])
        assert rec['key']==name and rec['initial_guess']=='independent_zero' and rec['numerical_state']=='converged' and rec['true_relative_residual']<=1e-8
        assert rec['physical_U']==expected['physical_U'] and rec['all_original_nodes']==n
        assert rec['physical_U']['shape']==[n,3] and rec['physical_U']['dtype'] in ('torch.float64','float64')
        d.stat_matches(rec['physical_U'])
    return payload

def load_response(d,np,torch,package,expected_digest,guard):
    payload=validate_metadata(d,package,expected_digest);guard()
    from .physical_U_fullrank6_v108 import FullRank6PhysicalUQuery
    from .physical_U_global_locator_v106 import PhysicalULocator
    from .located_force6_U_response_v109 import LocatedForce6UResponse
    counts=dict(geometry_body_reads=0,basis_body_reads=0,GT_reads=0,raw_H5_deck_GRID_reads=0,K_PCG_replays=0,neural_PT_reads=0)
    geometry=payload['geometry_identity'];assert d.identity(geometry['path'],hashed=True)==geometry
    with np.load(d.path(geometry['path']),allow_pickle=False)as archive:
        xyz=archive['vertices_mm'];cells=archive['cells10']
    counts['geometry_body_reads']+=1
    n,nc=payload['geometry_schema']['nodes'],payload['geometry_schema']['cells']
    assert xyz.shape==(n,3) and xyz.dtype==np.float64 and np.isfinite(xyz).all()
    assert cells.shape==(nc,10) and cells.dtype==np.int64
    assert (cells>=0).all() and (cells<n).all()
    basis=[]
    for name in payload['basis_order']:
        guard();item=payload['basis_records'][name]['physical_U']
        expected={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
        assert d.identity(item['path'],hashed=True)==expected
        value=np.load(d.path(item['path']),allow_pickle=False);counts['basis_body_reads']+=1
        dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
        assert str(value.dtype)==dtype=='float64' and list(value.shape)==item['shape']==[n,3] and np.isfinite(value).all()
        basis.append(value)
    vertices=torch.as_tensor(xyz,device='cuda');conn=torch.as_tensor(cells,device='cuda')
    fields=torch.as_tensor(np.stack(basis),device='cuda');B=torch.tensor(payload['reference_load6_columns'],dtype=torch.float64,device='cuda').T.contiguous()
    api=FullRank6PhysicalUQuery(torch,vertices,conn,fields,B,guard)
    locator=PhysicalULocator(torch,vertices,conn,guard)
    response=LocatedForce6UResponse(api,locator,payload['scope'])
    response.package_metadata=dict(package_digest=expected_digest,CAD=payload['CAD'],body_read_counts=counts,
        geometry_schema=payload['geometry_schema'],source_provenance=payload['source_provenance'],inherited_cost=payload['inherited_cost'],
        scientific_arrays_published=False,new_OptiStruct_GT_accuracy=False,signedS_provided=False)
    guard();return response
