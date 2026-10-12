"""Strict source-cache input reads and real P2 derivative qualification, no targets."""
from . import available_cad_pilot_data as d

def load_source(np,t,case,a,guard):
    item=case['input_identity'];guard();assert d.identity(item['path'],hashed=True)==item
    manifest=d.read(case['manifest_identity']['path']);assert manifest['status']=='completed' and manifest['scientific_group']=='TRAIN'
    inputs={}
    with np.load(d.path(item['path']),allow_pickle=False) as source:
        assert set(source.files)==set(manifest['array_schema'])
        for name,schema in manifest['array_schema'].items():
            guard();value=source[name];assert list(value.shape)==schema['shape'] and str(value.dtype)==schema['dtype'] and np.isfinite(value).all()
            inputs[name]=t.as_tensor(value,device='cuda')
    a['source_cache_reads']+=1;forces=inputs.pop('canonical_forces')
    assert forces.dtype==t.float64 and forces.shape==(3,len(inputs['vertices_mm']),3)
    return inputs,forces

def load_array(np,t,item,a,guard):
    guard();expected={k:v for k,v in item.items() if k in ('path','bytes','mtime_ns','sha256')};expected['path']=d.relative(d.path(item['path']))
    actual=d.identity(item['path'],hashed=True)
    assert all(actual[k]==v for k,v in expected.items())
    value=np.load(d.path(item['path']),allow_pickle=False);dtype={'torch.float64':'float64','torch.float32':'float32','torch.int64':'int64'}.get(item['dtype'],item['dtype'])
    assert list(value.shape)==item['shape'] and str(value.dtype)==dtype and np.isfinite(value).all()
    a['owned_array_reads']+=1
    a.setdefault('actual_legacy_array_hashes',{})[actual['path']]=dict(actual_identity=actual,historic_SHA_available='sha256' in expected)
    return t.as_tensor(value,device='cuda')

def load_sparse(np,t,row,a,guard):
    rp=load_array(np,t,row['rowptr'],a,guard);col=load_array(np,t,row['columns'],a,guard);val=load_array(np,t,row['values'],a,guard);nr,nc=row['shape']
    assert rp.dtype==col.dtype==t.int64 and val.dtype==t.float64 and rp.shape==(nr+1,) and col.shape==val.shape and int(rp[0])==0 and int(rp[-1])==len(col)
    assert bool((rp[1:]>=rp[:-1]).all() and (col>=0).all() and (col<nc).all())
    return t.sparse_csr_tensor(rp,col,val,size=(nr,nc),dtype=t.float64,device='cuda')

def qualify_derivative(np,t,inputs,U,guard):
    from .physical_U_query_v104 import shape
    from .continuous_query_geometry import shape_gradient
    guard();count=len(inputs['cells10']);rows=t.as_tensor(np.linspace(0,count-1,32,dtype=np.int64),device='cuda')
    assert len(t.unique(rows))==32
    bary=np.asarray([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.],[.5,.5,0,0],[0,.5,.5,0],[.5,0,.5,0],[.5,0,0,.5],[0,.5,0,.5],[0,0,.5,.5]],np.float64)
    ids=inputs['cells10'][rows];x=inputs['vertices_mm'][ids];u=U[:,ids]
    _,Dr,_=shape(t,t.as_tensor(bary[:,:3],device='cuda'));J=t.einsum('cia,eib->ceab',x,Dr);g=t.einsum('eib,ceba->ceia',Dr,t.linalg.inv(J));actual=t.einsum('lcia,ceib->lceab',u,g)
    # Independent accepted NumPy derivative has r=(b1,b2,b3); only physicalXYZ compared.
    D=shape_gradient(bary);xn=x.cpu().numpy();un=u.cpu().numpy();Jn=np.einsum('cia,eib->ceab',xn,D);gn=np.einsum('eib,ceba->ceia',D,np.linalg.inv(Jn));reference=np.einsum('lcia,ceib->lceab',un,gn)
    ref=t.as_tensor(reference,device='cuda');norm=float(t.linalg.vector_norm(ref));error=float(t.linalg.vector_norm(actual-ref));maximum=float((actual-ref).abs().max());relative=error/norm if norm>0 else None
    passed=relative<=1e-10 if relative is not None else maximum<=1e-12
    if not passed:raise ValueError('Actual P2 physical derivative convention qualification fails')
    guard();return dict(passed=True,actual_original_nodes=len(inputs['vertices_mm']),actual_cells=count,actual_geometry_only_cell_rows=32,actual_nodal_derivative_positions=320,actual_canonical_loads=3,relative=relative,maximum_absolute=maximum,independent_reference='Accepted NumPy b123 physicalXYZ vsTorch b012; actual physics fields, no toy/GT',full_node_recovery_still_checked_by_producer=True)
