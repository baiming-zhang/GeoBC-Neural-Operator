"""Matrix-free curved quadratic-tetrahedron elasticity."""

import math
import time
CHUNK = 8192
EDGES = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))

class ScientificNegative(RuntimeError):
    """Reached numerical/geometry/budget negative; no automatic repair."""

def _sync(torch):
    torch.cuda.synchronize()

def _torch_shape_gradient(torch, bary, device):
    """Exact source-equivalent dN/dr in VTK TET10 edge order, FP64."""
    bary = torch.as_tensor(bary, dtype=torch.float64, device=device)
    derivative = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64, device=device)
    edges = torch.tensor(EDGES, dtype=torch.int64, device=device)
    corner = (4 * bary - 1)[..., :, None] * derivative
    mid = 4 * (bary[..., edges[:, 1], None] * derivative[edges[:, 0]] + bary[..., edges[:, 0], None] * derivative[edges[:, 1]])
    return torch.cat((corner, mid), dim=-2)

def _quadrature(torch, device):
    a, b = ((5 + 3 * math.sqrt(5)) / 20, (5 - math.sqrt(5)) / 20)
    bary = [[a if i == j else b for i in range(4)] for j in range(4)]
    return (_torch_shape_gradient(torch, bary, device), _torch_shape_gradient(torch, [[0.25] * 4], device))

def _deadline(torch, began, prior_seconds, maximum=300):
    _sync(torch)
    if prior_seconds + time.monotonic() - began > maximum:
        raise ScientificNegative('Combined data/operator setup wall exceeded the setup time budget')

def _validate_inputs(torch, inputs, plan):
    n, t = (plan['expected_geometry']['nodes'], plan['expected_geometry']['cells'])
    x, cells, free, fixed, force = (inputs[name] for name in ('vertices_mm', 'cells10', 'free_node_ids', 'fixed_node_ids', 'force'))
    if x.shape != (n, 3) or cells.shape != (t, 10) or force.shape != (n, 3) or (x.dtype != torch.float64) or (force.dtype != torch.float64) or (cells.dtype != torch.int64) or (free.ndim != 1) or (fixed.ndim != 1) or (free.dtype != torch.int64) or (fixed.dtype != torch.int64) or (inputs['lame'].shape != (2,)) or (inputs['lame'].dtype != torch.float64):
        raise ValueError('Actual elastic input shape/dtype differs from input schema')
    if not torch.isfinite(x).all() or not torch.isfinite(force).all() or (not torch.isfinite(inputs['lame']).all()):
        raise ScientificNegative('Nonfinite geometry/material/force input')
    lam, mu = (float(value) for value in inputs['lame'].tolist())
    if mu <= 0 or 3 * lam + 2 * mu <= 0:
        raise ScientificNegative('Nonpositive isotropic elastic material energy')
    if free.numel() == 0 or inputs['derived_node_ids'].ndim != 1:
        raise ScientificNegative('No supported free identity DOFs or malformed output-only reference map')
    return (lam, mu)

class MatrixFreeElastic:

    def __init__(self, torch, cells, gradients, weights, node_count, lam, mu):
        self.torch, self.cells, self.gradients, self.weights = (torch, cells, gradients, weights)
        self.node_count, self.lam, self.mu, self.calls = (node_count, lam, mu, 0)
        self.apply_seconds = 0.0

    def apply(self, displacement):
        torch = self.torch
        _sync(torch)
        began = time.monotonic()
        result = torch.zeros_like(displacement)
        eye = torch.eye(3, dtype=torch.float64, device=displacement.device)
        for start in range(0, len(self.cells), CHUNK):
            stop = min(start + CHUNK, len(self.cells))
            cells, gradient = (self.cells[start:stop], self.gradients[start:stop])
            derivative = torch.einsum('nia,nqib->nqab', displacement[cells], gradient)
            strain = 0.5 * (derivative + derivative.transpose(-1, -2))
            trace = strain.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            stress = 2 * self.mu * strain + self.lam * trace[..., None, None] * eye
            local = torch.einsum('nqab,nqib,nq->nia', stress, gradient, self.weights[start:stop])
            result.index_add_(0, cells.reshape(-1), local.reshape(-1, 3))
        _sync(torch)
        self.calls += 1
        self.apply_seconds += time.monotonic() - began
        return result

def _build_geometry(torch, inputs, plan, began, prior_seconds):
    x, cells = (inputs['vertices_mm'], inputs['cells10'])
    device = x.device
    D, Dc = _quadrature(torch, device)
    n, t = (len(x), len(cells))
    gradients = torch.empty((t, 4, 10, 3), dtype=torch.float64, device=device)
    weights = torch.empty((t, 4), dtype=torch.float64, device=device)
    diagonal = torch.zeros((n, 3, 3), dtype=torch.float64, device=device)
    lam, mu = _validate_inputs(torch, inputs, plan)
    length = plan['expected_geometry']['characteristic_length_mm']
    minimum_det, maximum_condition = (math.inf, 0.0)
    geometry_seconds, preconditioner_seconds = (0.0, 0.0)
    eye = torch.eye(3, dtype=torch.float64, device=device)
    guard = plan['recipe']['geometry_guard']
    for start in range(0, t, CHUNK):
        stop = min(start + CHUNK, t)
        _sync(torch)
        tick = time.monotonic()
        Xe = x[cells[start:stop]]
        J = torch.einsum('nic,qir->nqcr', Xe, D)
        center = torch.einsum('nic,qir->nqcr', Xe, Dc)
        matrices = torch.cat((center, J), dim=1)
        if not torch.isfinite(matrices).all():
            raise ScientificNegative(f'Nonfinite full geometry at chunk{start}')
        normalized = matrices / length
        try:
            determinant = torch.linalg.det(normalized)
            singular = torch.linalg.svdvals(normalized)
        except torch.linalg.LinAlgError as error:
            raise ScientificNegative(f'Geometry determinant/SVD failed at chunk{start}: {error}') from error
        condition = singular[..., 0] / singular[..., -1]
        valid = torch.isfinite(condition) & torch.isfinite(determinant) & (determinant.abs() > guard['minimum_abs_detJ_over_length_cubed']) & (condition < guard['maximum_J_condition'])
        consistent = (torch.sign(determinant) == torch.sign(determinant[:, :1])).all(dim=1)
        if not valid.all() or not consistent.all():
            raise ScientificNegative(f'Full center/fourQ geometry invalid or orientation inconsistent at chunk{start}')
        minimum_det = min(minimum_det, float(determinant.abs().min()))
        maximum_condition = max(maximum_condition, float(condition.max()))
        try:
            gradient = torch.linalg.solve(J.transpose(-1, -2), D.transpose(-1, -2)[None].expand(stop - start, -1, -1, -1)).transpose(-1, -2)
        except torch.linalg.LinAlgError as error:
            raise ScientificNegative(f'Physical gradient solve failed at chunk{start}: {error}') from error
        weight = torch.linalg.det(J).abs() / 24.0
        if not torch.isfinite(gradient).all() or not torch.isfinite(weight).all() or (not (weight > 0).all()):
            raise ScientificNegative('Nonfinite physical gradients/integration weights')
        gradients[start:stop], weights[start:stop] = (gradient, weight)
        _sync(torch)
        geometry_seconds += time.monotonic() - tick
        tick = time.monotonic()
        gg = gradient.square().sum(dim=-1)
        local = mu * gg[..., None, None] * eye + (lam + mu) * gradient[..., :, None] * gradient[..., None, :]
        block = (weight[..., None, None, None] * local).sum(dim=1)
        diagonal.index_add_(0, cells[start:stop].reshape(-1), block.reshape(-1, 3, 3))
        _sync(torch)
        preconditioner_seconds += time.monotonic() - tick
        _deadline(torch, began, prior_seconds)
    tick = time.monotonic()
    free_block = diagonal[inputs['free_node_ids']]
    chol, info = torch.linalg.cholesky_ex(free_block, check_errors=False)
    if (info != 0).any() or not torch.isfinite(chol).all():
        raise ScientificNegative('Reduced3x3 block Jacobi Cholesky failed; no shift/fallback')
    _sync(torch)
    preconditioner_seconds += time.monotonic() - tick
    operator = MatrixFreeElastic(torch, cells, gradients, weights, n, lam, mu)
    return (operator, chol, diagonal, {'all_cells': t, 'center_points': t, 'fourQ_points': t * 4, 'minimum_abs_detJ_over_length_cubed': minimum_det, 'maximum_J_condition': maximum_condition, 'orientation': 'Per-cell consistent center/fourQ; consistently negative cells allowed', 'geometry_seconds': geometry_seconds, 'preconditioner_seconds': preconditioner_seconds})

def _restore_full(torch, z, inputs):
    result = torch.zeros_like(inputs['vertices_mm'])
    result[inputs['free_node_ids']] = z
    nodes = inputs['derived_node_ids']
    for index in range(len(nodes)):
        start, stop = (int(value) for value in inputs['derived_rowptr'][index:index + 2].tolist())
        sources = inputs['derived_source_ids'][start:stop]
        result[nodes[index]] = torch.einsum('lij,lj->i', inputs['derived_blocks'][start:stop], result[sources])
    return result


def _rigid_columns(torch, x, length):
    centered = (x - x.mean(dim=0)) / length
    eye = torch.eye(3, dtype=torch.float64, device=x.device)
    return [eye[j].expand_as(x).clone() for j in range(3)] + [torch.cross(eye[j].expand_as(x), centered, dim=-1) for j in range(3)]

def _algebra_guards(torch, operator, diagonal, inputs, plan, began, prior_seconds):
    tolerance = plan['recipe']['input_operator']['relative_algebra_guard_tolerance']
    vectors = _rigid_columns(torch, inputs['vertices_mm'], plan['expected_geometry']['characteristic_length_mm'])
    rigid_records = []
    for index, vector in enumerate(vectors):
        applied = operator.apply(vector)
        scale = torch.linalg.vector_norm(torch.einsum('nij,nj->ni', diagonal, vector))
        if not torch.isfinite(applied).all() or not torch.isfinite(scale) or scale <= 0:
            raise ScientificNegative('Invalid rigid-null scale/operator')
        relative = float(torch.linalg.vector_norm(applied) / scale)
        rigid_records.append({'column': index, 'relative_null': relative, 'denominator': float(scale), 'denominator_definition': 'norm(full nodal diagonal times rigid column)'})
        if relative > tolerance:
            raise ScientificNegative(f'Rigid-null relative guard failed at column{index}: {relative}')
        _deadline(torch, began, prior_seconds)
    constrained = torch.stack(vectors, dim=-1)[inputs['fixed_node_ids']].reshape(-1, 6)
    singular = torch.linalg.svdvals(constrained)
    rank_ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    if not math.isfinite(rank_ratio) or rank_ratio <= tolerance:
        raise ScientificNegative('Actual fixed translations do not eliminate all six physical rigid motions')
    generator = torch.Generator(device=inputs['vertices_mm'].device).manual_seed(2026)
    u = torch.zeros_like(inputs['vertices_mm'])
    v = torch.zeros_like(u)
    free = inputs['free_node_ids']
    u[free] = torch.randn((len(free), 3), dtype=torch.float64, device=u.device, generator=generator)
    v[free] = torch.randn((len(free), 3), dtype=torch.float64, device=u.device, generator=generator)
    Ku, Kv = (operator.apply(u), operator.apply(v))
    denominator = torch.linalg.vector_norm(u) * torch.linalg.vector_norm(Kv) + torch.linalg.vector_norm(v) * torch.linalg.vector_norm(Ku)
    energy_u, energy_v = ((u * Ku).sum(), (v * Kv).sum())
    if not torch.isfinite(Ku).all() or not torch.isfinite(Kv).all() or (not torch.isfinite(denominator)) or (denominator <= 0) or (not torch.isfinite(energy_u)) or (not torch.isfinite(energy_v)):
        raise ScientificNegative('Nonfinite random operator algebra')
    symmetry = float(((u * Kv).sum() - (v * Ku).sum()).abs() / denominator)
    eu = float(energy_u / (torch.linalg.vector_norm(u) * torch.linalg.vector_norm(Ku)))
    ev = float(energy_v / (torch.linalg.vector_norm(v) * torch.linalg.vector_norm(Kv)))
    if symmetry > tolerance or eu < -tolerance or ev < -tolerance or (energy_u <= 0) or (energy_v <= 0):
        raise ScientificNegative('Reduced random symmetry/positive-energy guard failed')
    if operator.calls != 8:
        raise RuntimeError('Exactly eight full K applications are required for operator guards')
    _deadline(torch, began, prior_seconds)
    return {'passed': True, 'K_applications': 8, 'rigid_null': rigid_records, 'constraint_rigid_rank': 6, 'constraint_smallest_over_largest_singular_value': rank_ratio, 'symmetry_relative': symmetry, 'symmetry_denominator': float(denominator), 'random_normalized_energies': [eu, ev], 'seed': 2026, 'scope': 'Sampled full-operator/reduced random algebra; no proof of complete spectrum or engineering validity'}
