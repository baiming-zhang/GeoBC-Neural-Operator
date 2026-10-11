"""Shared full-case training primitives for broad GeoBC solver comparisons.

No I/O, dataset selection, optimizer, launch or evaluation is performed here.
A future guarded experiment supplies frozen TRAIN keys and the protected
prospective loader. Existing FNO, native ++ and conditioned TCNN architectures
are reused; their precision, parameter count and representation differ and
must be reported rather than described as a capacity-matched ablation.
"""
import random

FAMILIES = ('FNO9', 'PP256_NATIVE', 'HASH_PN')


def balanced_schedule(keys, passes, seed=2026):
    """Each fixed case occurs once per pass; separate RNG from model training."""
    keys = list(keys)
    if not keys or len(set(keys)) != len(keys) or passes < 1 or not isinstance(passes, int):
        raise ValueError('Unique nonempty case keys and positive integer passes required')
    rng = random.Random(seed)
    result = []
    for _ in range(passes):
        rows = list(keys)
        rng.shuffle(rows)
        result.extend(rows)
    return result


def build_solver(family, device='cuda', seed=2026, *, injected_hash_decoder=None):
    """Construct a candidate only; caller owns release/resource/locking gates.

    CPU decoder injection is solely for interface checks. It cannot qualify
    tiny-cuda-nn gradients, memory, precision or real prediction quality.
    """
    if family not in FAMILIES:
        raise ValueError('Unknown fixed macro candidate')
    if injected_hash_decoder is not None and family != 'HASH_PN':
        raise ValueError('Decoder injection applies only to a CPU Hash fixture')
    import torch
    from .common_grid_solver import CommonGridSolver
    from .native_context_solver import NativeContextSolver
    from .conditioned_field import ConditionedField, ConditionedFieldConfig
    if family == 'FNO9':
        model = CommonGridSolver('FNO9', seed).to(device)
    elif family == 'PP256_NATIVE':
        model = NativeContextSolver(seed).to(device)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = ConditionedField(ConditionedFieldConfig(
                geometry_mode='shape', out_channels=4, seed=seed),
                device=device, decoder=injected_hash_decoder)
    model.macro_family = family
    return model


def encode_case(model, case):
    if model.macro_family == 'HASH_PN':
        return model.encode_geometry(case['cloud'], case['mask']), None
    return model.encode_case(case)


def query_case(model, case, start, stop, context, latent):
    if model.macro_family == 'HASH_PN':
        return model(case['xyz'][start:stop], case['features'][start:stop], context=context)
    return model.query(case, start, stop, context, latent)


def full_case_backward(model, case, allowed_TRAIN_keys, guard=lambda: None, chunk_size=32768):
    """Equal-case, full original-node encoded4 MSE with one global context.

    Each update covers every node of one permitted TRAIN case; decoder chunks
    do not approximate/split the operator context or sample nodes. Hash's
    decoder gradients accumulate across chunks and its geometry encoder is
    backpropagated once. Autograd accumulation has backend rounding limits;
    CPU fixture equality is not proof of actual FP16/official CUDA equality.
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
        raise RuntimeError('Trainable, attached full-case geometry context required')
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
            raise RuntimeError('Operator latent has missing/nonfinite gradient')
        torch.autograd.backward((context, latent), (c.grad, l.grad))
    return total, {'geometry_context_L2': float(c.grad.norm()),
                   'operator_latent_L2': None if l is None else float(l.grad.norm()),
                   'original_nodes_used': count, 'node_sampling': False}


def profile(model):
    """Describe actual constructed model; no inference or dataset access."""
    family = model.macro_family
    fixture = getattr(model, 'backend', None) == 'injected_cpu_interface_decoder'
    return {
        'family': family, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'global_geometry_context_dim': 32, 'local_BC_material_channels': 25,
        'output': 'encoded signed uXYZ and source S; scaler owned by protocol',
        'geometry_sampling': 'same supplied target-independent 1024 cloud/mask',
        'requires_R64_members': family == 'FNO9',
        'precision': 'CPU injected fixture; TCNN precision untested' if fixture else 'native TCNN FP16 with FP32 masters' if family == 'HASH_PN' else 'highest FP32, TF32 disabled by caller',
        'architecture': model.source_config if family == 'HASH_PN' else model.operator.source_config if family == 'PP256_NATIVE' else model.counts(),
        'capacity_matched_across_families': False,
        'continuous_query_policy': 'Declared native25/latent extension needed; original-node accuracy alone is insufficient',
        'actual_CPU_fixture_only': fixture,
    }


__all__ = ['FAMILIES', 'balanced_schedule', 'build_solver', 'encode_case', 'query_case',
           'full_case_backward', 'profile']
