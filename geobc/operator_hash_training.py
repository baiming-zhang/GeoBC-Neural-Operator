"""Primitives for a future whole-case operator/continuous decoder comparison.

Independent of frozen active experiments. No launch, optimizer, data I/O,
target selection or approval is performed here. The caller owns all gates.
"""
from .macro_training import balanced_schedule, encode_case, query_case

FAMILIES = ('FNO9', 'HASH_PN', 'FNO_GLOBAL_HASH')


def build_solver(family, device='cuda', seed=2026):
    if family not in FAMILIES:
        raise ValueError('Unknown whole-context decoder candidate')
    if family != 'FNO_GLOBAL_HASH':
        from .macro_training import build_solver as original
        return original(family, device=device, seed=seed)
    from .operator_conditioned_hash import OperatorConditionedHashSolver
    model = OperatorConditionedHashSolver(seed=seed, device=device)
    model.macro_family = family
    return model


def full_case_backward(model, case, allowed_TRAIN_keys, guard=lambda: None,
                       chunk_size=32768):
    """All original nodes, equal-case MSE, one full context backpropagation.

    This preserves the existing macro training chain rule independently of
    its frozen family enum. It never creates separate chunk operator contexts.
    """
    if case.get('scientific_group') != 'TRAIN' or case.get('key') not in allowed_TRAIN_keys:
        raise PermissionError('Only a frozen permitted TRAIN case may update the model')
    if model.macro_family not in FAMILIES or chunk_size < 1:
        raise ValueError('Fixed family and positive decoder chunk size required')
    import torch
    count = len(case['xyz'])
    if count == 0 or case['target'].shape != (count, 4):
        raise ValueError('All original-node signed encoded4 targets required')
    guard()
    context, latent = encode_case(model, case)
    if context.shape != (1, 32) or not context.requires_grad:
        raise RuntimeError('Trainable attached full-case geometry context required')
    c = context.detach().requires_grad_(True)
    l = None if latent is None else latent.detach().requires_grad_(True)
    total = 0.
    for start in range(0, count, chunk_size):
        guard()
        stop = min(start+chunk_size, count)
        prediction = query_case(model, case, start, stop, c, l)
        if prediction.shape != (stop-start, 4):
            raise ValueError('Prediction must retain signed four-channel layout')
        loss = (prediction-case['target'][start:stop]).square().mean()*((stop-start)/count)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite complete-case objective')
        loss.backward()
        total += float(loss.detach())
    if c.grad is None or not torch.isfinite(c.grad).all():
        raise RuntimeError('Geometry context has missing/nonfinite gradient')
    if latent is None:
        torch.autograd.backward(context, c.grad)
    else:
        if l.grad is None or not torch.isfinite(l.grad).all():
            raise RuntimeError('Whole-case operator latent has missing/nonfinite gradient')
        torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, dict(geometry_context_L2=float(c.grad.norm()),
                      operator_latent_L2=None if l is None else float(l.grad.norm()),
                      original_nodes_used=count, node_sampling=False)


def profile(model):
    if model.macro_family != 'FNO_GLOBAL_HASH':
        from .macro_training import profile as original
        return original(model)
    return dict(family=model.macro_family,
                parameter_count=sum(p.numel() for p in model.parameters()),
                global_geometry_context_dim=32, whole_case_operator_latent_channels=16,
                local_BC_material_channels=25, requires_R64_members=True,
                output='Encoded signed uXYZ and source S; protocol owns scaler',
                geometry_sampling='Same target-independent1024 cloud/mask',
                precision='FP32 encoder/operator; native TCNN FP16 decoder with FP32 masters',
                architecture=model.source_config, capacity_matched_across_families=False,
                continuous_query_policy='Fixed whole-case context and explicit native25 extension',
                actual_CPU_fixture_only=False,
                independent_offnode_FEM_accuracy_measured=False)
