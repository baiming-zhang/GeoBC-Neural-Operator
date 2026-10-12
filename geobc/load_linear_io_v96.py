"""Explicit target-free scientific I/O, invoked only by the released runner."""
from . import available_cad_pilot_data as d


def load_inputs(np,torch,manifest,keys,receipt,guard):
    from .prospective_dataset import resolve_case
    arrays={};specs={};cases={}
    for key in keys:
        record,items=resolve_case(manifest,key,False,100,True)
        assert set(items)=={'xyz','features','cloud','mask','grid','query_indices','query_weights'}
        n=record['num_nodes'];shapes={'xyz':(n,3),'features':(n,25),'cloud':(1024,3),'mask':(1024,),
            'grid':(29,64,64,64),'query_indices':(n,8),'query_weights':(n,8)}
        case=dict(key=key,scientific_group='TRAIN')
        for name,item in items.items():
            guard();file=item['path']
            if file not in arrays:
                d.stat_matches(item);receipt['input_array_attempts']+=1
                value=np.load(d.path(file),allow_pickle=False);receipt['input_array_reads']+=1
                assert value.shape==shapes[name]and str(value.dtype)in (('bool',)if name=='mask'else('int32','int64')if name=='query_indices'else('float32',))
                if value.dtype.kind=='f':assert np.isfinite(value).all()
                if name=='query_indices':assert value.min()>=0 and value.max()<64**3
                arrays[file]=torch.as_tensor(value,device='cuda');specs[file]=(name,item)
            else:assert specs[file]==(name,item)
            case[name]=arrays[file]
        case['cloud']=case['cloud'][None];case['mask']=case['mask'][None];case['grid']=case['grid'][None];cases[key]=case
    assert receipt['input_array_reads']==receipt['input_array_attempts']==len(arrays)
    return cases,arrays


def load_model(torch,c,science,receipt,guard):
    guard();cp=c['checkpoint'];assert d.identity(cp['path'],hashed=True)==cp;receipt['checkpoint_hash_reads']+=1
    saved=torch.load(d.path(cp['path']),map_location='cpu',weights_only=False);receipt['checkpoint_loads']+=1
    tp=d.read(c['training_plan_identity']['path']);family=c['family']
    assert saved['family']==family and saved['step']==saved['cursor']==6000 and saved['scientific_digest']==c['training_SCI']
    assert saved['case_exposure_counts']=={k:20 for k in tp['scientific']['TRAIN_keys']}and saved['full_order_digest']==c['full_order_digest']
    assert saved['scaler']==science['scaler']and saved['dataset_identity']==science['dataset_identity']and set(saved['rng'])=={'python','numpy','torch','cuda'}
    assert saved['optimizer']['state']and all(int(x['step'])==6000 for x in saved['optimizer']['state'].values())
    boot=saved['native_origin_clocks']['boot_id']if family=='FNO9'else saved['execution']['boot_id'];assert boot==c['training_boot_id']
    if family=='FNO9':from .macro_training import build_solver
    elif family=='FNO_GLOBAL_HASH':from .operator_hash_metadata_v6 import build_solver
    else:from .transolver_bf16_hash_v65 import build_solver
    model=build_solver(family,seed=2026);model.load_state_dict(saved['model'],strict=True)
    state=model.state_dict();assert set(state)==set(saved['model'])and all(state[k].dtype==saved['model'][k].dtype and state[k].shape==saved['model'][k].shape and torch.equal(state[k].cpu(),saved['model'][k])for k in state)
    del saved,state;guard();return model.eval()


def read_owned_array(np,item,receipt,guard):
    guard();receipt['own_array_hash_attempts']+=1
    assert d.identity(item['path'],hashed=True)=={k:v for k,v in item.items()if k in ('path','bytes','mtime_ns','sha256')}
    value=np.load(d.path(item['path']),mmap_mode='r',allow_pickle=False);receipt['own_array_reads']+=1
    assert list(value.shape)==item['shape']and str(value.dtype)==item['dtype']and np.isfinite(value).all()
    return value
