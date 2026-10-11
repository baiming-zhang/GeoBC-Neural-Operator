"""GINO adapter for the existing whole-case signed physical objective.

No dataset, optimizer, checkpoint, model construction or runtime release here.
The previous actual matched-frequency GINO is preserved. This adapter changes
the objective from encoded MSE to the established equal-weight physical U/S
relative squared error; every native node contributes exactly once.
"""
from dataclasses import replace
import math


def build_candidate(device):
    """Reuse the actually trained upstream modes24/24/22 model unchanged."""
    from .gino_train import _build_model
    import torch
    model = _build_model(torch, device)
    model.source_config = dict(model.source_config,
        objective_adapter='gino_physical_training_v52',
        resource_scope='Future TRAIN100 physical5; actual fullN qualification not yet run',
        architecture_changed_from_old_matched_GINO=False)
    return model


def full_physical_backward(model, case, physical_target, scale, norms,
                           allowed_keys, guard=lambda: None):
    """One live encode, all complete query chunks, one encoder backward.

    GINO's geometry context is already inside its latent graph. A single
    detached latent leaf therefore retains the exact chain rule to PointNet,
    input GNO and FNO, while query gradients reach output GNO/projection live.
    CPU gradient parity is an algorithm check, not a real-model resource gate.
    """
    import torch
    from .physical_field_loss import physical_relative_squared_loss
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_keys:
        raise PermissionError('Only the separately fixed TRAIN keys are allowed')
    count = len(case['xyz'])
    if physical_target.shape != (count, 4) or count != norms.node_count:
        raise ValueError('Target/norm must cover every original node')
    chunk = model.query_chunk
    if type(chunk) is not int or chunk < 1:
        raise ValueError('Positive fixed query chunk required')
    guard()
    original = model.encode_case(case)
    leaf = original.latent.detach().requires_grad_(True)
    query_state = replace(original, latent=leaf)
    consumed, total = 0, 0.
    for begin in range(0, count, chunk):
        guard()
        end = min(begin + chunk, count)
        rows = torch.arange(begin, end, dtype=torch.int64, device=leaf.device)
        encoded = model.query(query_state, rows)
        if encoded.shape != (end-begin, 4):
            raise ValueError('GINO must return encoded signed4 for every query')
        loss = physical_relative_squared_loss(encoded, physical_target[begin:end], scale, norms)
        loss.backward()
        total += float(loss.detach())
        consumed += end-begin
        del loss, encoded, rows
    if consumed != count or not math.isfinite(total) or leaf.grad is None:
        raise RuntimeError('Incomplete full-node objective or latent gradient')
    if not bool(torch.isfinite(leaf.grad).all()):
        raise FloatingPointError('Nonfinite latent gradient; no clipping/rescue')
    leaf_norm = float(leaf.grad.norm())
    guard()
    original.latent.backward(leaf.grad)
    return total, dict(original_nodes_used=count, node_sampling=False,
        input_encode_calls=1, input_graph_backward_calls=1,
        output_chunks=(count+chunk-1)//chunk, operator_latent_L2=leaf_norm,
        geometry_context='Live inside original latent graph; not detached separately')
