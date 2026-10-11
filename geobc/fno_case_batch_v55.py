"""Whole-native-case mean gradients, with one GPU case graph at a time.

This is only the math adapter. It cannot load inputs, update an optimizer,
choose cases, or launch a campaign. A separate frozen driver is required.
"""
from .train_physical_cache_v12 import full_physical_backward


def four_case_backward(model,cases,cache,scale,allowed_keys,guard=lambda:None,on_case=None):
    import torch
    if len(cases)!=4 or len({c.get('key') for c in cases})!=4:
        raise ValueError('Exactly four distinct ordered cases required')
    if any(c.get('scientific_group')!='TRAIN' or c.get('key')not in allowed_keys for c in cases):
        raise PermissionError('Only frozen original TRAIN100 cases may update')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Caller must clear all model gradients before a batch')
    losses=[];leaves=[]
    for case in cases:
        guard()
        if on_case:on_case('attempt',case['key'])
        loss,leaf=full_physical_backward(model,case,cache,scale,allowed_keys,guard)
        if on_case:on_case('completed',case['key'])
        losses.append(loss);leaves.append(dict(key=case['key'],**leaf))
    # Each case already has its own complete physical U/S norm. Average
    # the four full-case gradients, not nodes or separately normalized chunks.
    guard()
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite accumulated case gradient')
            parameter.grad.div_(4.)
    return sum(losses)/4.,dict(case_keys=[c['key']for c in cases],case_loss=losses,
        case_leaf_gradients=leaves,case_weights=[.25]*4,
        unique_CAD_count=len({c['key'].split('/')[0]for c in cases}),
        original_nodes_used=sum(len(c['xyz'])for c in cases),node_sampling=False,
        simultaneous_case_graphs=1,optimizer_updates=0)
