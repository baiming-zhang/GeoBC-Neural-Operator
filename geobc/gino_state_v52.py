"""Hash and load GINO tensor state plus original constructor metadata.

Upstream BaseModel.state_dict injects a global '_metadata' dictionary. Keep
and verify it instead of treating every state entry as a tensor. No state,
checkpoint, model, initialization or optimizer is changed by hashing.
"""
import functools
import hashlib
import json
import math
import types


def canonical_metadata(value):
    import torch
    if value is None or isinstance(value,(str,bool,int)):
        return [type(value).__name__,value]
    if isinstance(value,float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite constructor metadata')
        return ['float',value.hex()]
    if isinstance(value,(torch.dtype,torch.device)):
        return [type(value).__name__,str(value)]
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value):
            raise ValueError('Metadata dictionary requires string keys')
        return ['dict',[[k,canonical_metadata(v)] for k,v in sorted(value.items())]]
    if isinstance(value,(tuple,list)):
        return [type(value).__name__,[canonical_metadata(v) for v in value]]
    if isinstance(value,functools.partial):
        return ['partial',canonical_metadata(value.func),canonical_metadata(value.args),canonical_metadata(value.keywords or {})]
    if isinstance(value,(types.FunctionType,types.BuiltinFunctionType,type)):
        module=getattr(value,'__module__',None)
        name=getattr(value,'__qualname__',None)
        if not module or not name:
            raise ValueError('Constructor callable needs a stable fully qualified identity')
        return ['callable',module,name]
    raise TypeError('Unsupported constructor metadata: '+type(value).__name__)


def state_digest(state):
    import torch
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(json.dumps(name).encode()+b'\0')
        if isinstance(value,torch.Tensor):
            h.update(json.dumps(['tensor',str(value.dtype),list(value.shape)]).encode()+b'\0')
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            if name != '_metadata':
                raise TypeError('Unexpected nontensor state entry: '+name)
            h.update(json.dumps(canonical_metadata(value),separators=(',',':'),ensure_ascii=True).encode())
        h.update(b'\0')
    return h.hexdigest()


def load_verified_state(model,state):
    """Verify pinned constructor metadata before the parent's strict load.

    The top adapter is an nn.Module; upstream's unprefixed metadata would be
    an unexpected key in its strict load. Verify metadata separately and load
    all tensor keys strictly, without mutating the saved checkpoint mapping.
    """
    import torch
    current=model.state_dict()
    if set(current)!=set(state) or '_metadata' not in state:
        raise ValueError('Exact original GINO state schema required')
    if canonical_metadata(current['_metadata']) != canonical_metadata(state['_metadata']):
        raise ValueError('Pinned GINO constructor metadata differs')
    tensors={}
    for name,value in state.items():
        if name=='_metadata':
            continue
        expected=current[name]
        if not isinstance(value,torch.Tensor) or not isinstance(expected,torch.Tensor):
            raise TypeError('All remaining GINO state entries must be tensors')
        if value.shape!=expected.shape or value.dtype!=expected.dtype:
            raise ValueError('Original GINO state shape/dtype differs: '+name)
        tensors[name]=value
    return model.load_state_dict(tensors,strict=True)
