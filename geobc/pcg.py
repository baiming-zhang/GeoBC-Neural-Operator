"""Independent zero-start PCG lanes with periodic true residuals."""
import time
import torch

def solve(apply, rhs, precondition, guard=lambda: None, checkpoint=None,
          tolerance=1e-8, maximum_iterations=10000, seconds=300):
    started = time.monotonic()
    norm = torch.linalg.vector_norm(rhs, dim=(1, 2))
    if not bool((norm > 0).all()):
        raise ValueError('Each lane needs a nonzero reduced RHS')
    lanes = len(rhs)
    x = torch.zeros_like(rhs)
    r = rhs - apply(x)
    z = precondition(r)
    p = z.clone()
    rz = (r*z).sum((1, 2))
    active = torch.ones(lanes, device=rhs.device, dtype=torch.bool)
    iterations = torch.zeros(lanes, device=rhs.device, dtype=torch.int64)
    trace = []
    step = 0
    def save():
        if checkpoint is not None:
            checkpoint({'x': x, 'r': r, 'p': p, 'rz': rz, 'active': active,
                        'iterations': iterations, 'trace': trace, 'step': step,
                        'seconds': time.monotonic()-started})
    try:
        if not bool(torch.isfinite(rz).all() and (rz > 0).all()):
            raise ArithmeticError('PCG initial preconditioner breakdown')
        for step in range(1, maximum_iterations+1):
            guard()
            if time.monotonic()-started >= seconds-10:
                break
            iterations[active] += 1
            Ap = apply(p)
            pAp = (p*Ap).sum((1, 2))
            if not bool(torch.isfinite(Ap).all() and (pAp[active] > 0).all()):
                raise ArithmeticError('PCG operator breakdown')
            alpha = torch.zeros_like(norm)
            alpha[active] = rz[active]/pAp[active]
            x += alpha[:, None, None]*p
            r -= alpha[:, None, None]*Ap
            recursive = torch.linalg.vector_norm(r, dim=(1, 2))/norm
            if step % 25 == 0 or bool((recursive[active] <= tolerance).any()):
                true_r = rhs-apply(x)
                relative = torch.linalg.vector_norm(true_r, dim=(1, 2))/norm
                if not bool(torch.isfinite(relative).all()):
                    raise ArithmeticError('Nonfinite PCG residual')
                active &= relative > tolerance
                trace.append({'step': step, 'true_relative_residual': relative.detach().cpu().tolist()})
            if not bool(active.any()):
                break
            z = precondition(r)
            new_rz = (r*z).sum((1, 2))
            if not bool(torch.isfinite(new_rz[active]).all() and (new_rz[active] > 0).all()):
                raise ArithmeticError('PCG preconditioner breakdown')
            beta = torch.zeros_like(norm)
            beta[active] = new_rz[active]/rz[active]
            p = z+beta[:, None, None]*p
            p[~active] = 0
            rz = new_rz
            if step % 25 == 0:
                save()
        relative = torch.linalg.vector_norm(rhs-apply(x), dim=(1, 2))/norm
    finally:
        save()
    return x, {'iterations': iterations.cpu().tolist(), 'true_relative_residual': relative.cpu().tolist(),
               'converged': bool((relative <= tolerance).all()), 'seconds': time.monotonic()-started, 'trace': trace}
