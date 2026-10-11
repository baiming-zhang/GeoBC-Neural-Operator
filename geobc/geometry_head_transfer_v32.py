"""Strict learned TRAIN geometry/head transfer; operator remains fresh.

No dataset/checkpoint body loading, optimizer or scaler ownership. A future
campaign must bind the actual TRAIN-only teacher and count inherited exposure.
"""
import hashlib
import torch


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def transfer_geometry_head(model, teacher_state):
    operator_before = state_digest(model.operator.state_dict())
    original = state_digest(model.state_dict())
    expected_names = {group+'.'+name for group in ('encoder','head') for name in getattr(model,group).state_dict()}
    supplied_names = {k for k in teacher_state if k.startswith(('encoder.','head.'))}
    if expected_names != supplied_names:
        raise ValueError('Teacher encoder/head keys must exactly match the receiver')
    transferred = {}
    for group in ('encoder','head'):
        target = getattr(model,group).state_dict()
        values = {name:teacher_state[group+'.'+name] for name in target}
        if not all(value.shape==target[name].shape and value.dtype==target[name].dtype
                   and torch.isfinite(value).all() for name,value in values.items()):
            raise ValueError('Teacher weight shape/dtype/finiteness differs')
        getattr(model,group).load_state_dict(values,strict=True)
        actual = getattr(model,group).state_dict()
        if not all(torch.equal(value.detach().cpu(),values[name].detach().cpu()) for name,value in actual.items()):
            raise RuntimeError('Transferred weights do not equal the teacher')
        transferred[group] = state_digest(actual)
    operator_after = state_digest(model.operator.state_dict())
    if operator_before != operator_after:
        raise RuntimeError('Geometry/head transfer must leave fresh operator unchanged')
    return dict(fresh_model_before=original, transferred_model_after=state_digest(model.state_dict()),
        operator_before=operator_before,operator_after=operator_after,operator_unchanged=True,
        encoder_head_equal_teacher=True,transferred_group_digests=transferred,
        teacher_state_digest=state_digest(teacher_state),teacher_training_or_exposure_validated_here=False)
