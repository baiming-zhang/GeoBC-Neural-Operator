"""One numerical CSV parse per FINAL CAD after all120 predictions commit."""
import csv


def read_one_CAD(np,d,science,cad,counts,output,result,guard,persist):
    assert result['prediction_barrier_complete'] and result['committed_finite_predictions']==120
    assert cad in science['CADs'] and cad not in result['FINAL_CADs_attempted']
    source=science['FINAL_sources'][cad];field=source['CSV'];permutation=source['permutation'];n=source['num_nodes']
    guard();d.stat_matches(field);d.stat_matches(permutation)
    # Full source-byte verification is bounded and precedes interpretation.
    import hashlib
    digest=hashlib.sha256()
    with d.path(field['path']).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):guard();digest.update(block)
    assert digest.hexdigest()==field['sha256'];d.stat_matches(field)
    assert d.identity(permutation['path'],hashed=True)['sha256']==permutation['sha256']
    indices=np.load(d.path(permutation['path']),mmap_mode='r',allow_pickle=False)
    assert indices.shape==(n,) and str(indices.dtype)=='int64' and indices.min()==0 and indices.max()==n-1 and len(np.unique(indices))==n
    with d.path(field['path']).open(encoding='utf-8-sig',newline='') as stream:columns=next(csv.reader(stream))
    names=d.COORDS+d.TARGET_COLUMNS
    assert len(columns)==len(set(columns)) and all(name in columns for name in names)
    # Persist the attempt before opening any numerical source field; no replay.
    result['FINAL_CADs_attempted'].append(cad);result['FINAL_numeric_CSV_parse_attempts']+=1;persist()
    result['raw_GT_reads']+=1
    values=np.loadtxt(d.path(field['path']),delimiter=',',skiprows=1,usecols=[columns.index(name) for name in names],dtype=np.float64,ndmin=2)
    result['FINAL_numeric_CSV_parses']+=1;result['FINAL_numeric_reads']+=3;guard()
    assert values.shape==(n,16) and np.isfinite(values).all()
    ids=values[:,0];assert (ids>0).all() and (ids<2**53).all() and np.array_equal(ids,np.rint(ids)) and len(np.unique(ids))==n
    # The pinned mapping was made from these exact CSV bytes and original H5
    # vertices in the target-free input stage. Do not remap stresses or signs.
    aligned=np.ascontiguousarray(values[indices,4:],dtype=np.float32)
    assert aligned.shape==(n,12) and np.isfinite(aligned).all()
    del values,indices
    result['FINAL_CADs_parsed'].append(cad)
    return {cad+'/'+load:np.ascontiguousarray(aligned[:,4*i:4*i+4]) for i,load in enumerate(d.LOADS)}
