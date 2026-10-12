"""Constraint-masked Galerkin coarse preconditioner with 32 Jacobi steps."""

import math
import time
from .elasticity import ScientificNegative
EDGES = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))

def negative(condition, message):
    if not bool(condition):
        raise ScientificNegative(message)

def make_mapping(torch, inputs, ledger):
    """Actual topology only: no coordinate midpoint or GRID-ID arithmetic guess."""
    cells, free = (inputs['cells10'], inputs['free_node_ids'])
    n = len(inputs['vertices_mm'])
    device = cells.device
    corner_ids = torch.unique(cells[:, :4].reshape(-1), sorted=True)
    corner_mask = torch.zeros(n, dtype=torch.bool, device=device)
    corner_mask[corner_ids] = True
    mids = cells[:, 4:].reshape(-1)
    negative(not corner_mask[mids].any(), 'Global node is both CTETRA corner and midpoint')
    edge_index = torch.tensor(EDGES, dtype=torch.int64, device=device)
    endpoints = torch.sort(cells[:, :4][:, edge_index].reshape(-1, 2), dim=1).values
    edge_keys = endpoints[:, 0] * n + endpoints[:, 1]
    order = torch.argsort(mids)
    sm, sk, se = (mids[order], edge_keys[order], endpoints[order])
    negative(not ((sm[1:] == sm[:-1]) & (sk[1:] != sk[:-1])).any(), 'One midpoint has multiple unordered global corner edges')
    order_edge = torch.argsort(edge_keys)
    ek, em = (edge_keys[order_edge], mids[order_edge])
    negative(not ((ek[1:] == ek[:-1]) & (em[1:] != em[:-1])).any(), 'One unordered corner edge has multiple midpoint IDs')
    keep = torch.ones(len(sm), dtype=torch.bool, device=device)
    keep[1:] = sm[1:] != sm[:-1]
    unique_mid, unique_ends = (sm[keep], se[keep])
    free_map = torch.full((n,), -1, dtype=torch.int64, device=device)
    free_map[free] = torch.arange(len(free), dtype=torch.int64, device=device)
    coarse_corners = corner_ids[free_map[corner_ids] >= 0]
    negative(len(coarse_corners) > 0, 'No actual free CTETRA corner coarse nodes')
    coarse_map = torch.full((n,), -1, dtype=torch.int64, device=device)
    coarse_map[coarse_corners] = torch.arange(len(coarse_corners), dtype=torch.int64, device=device)
    rows = [free_map[coarse_corners]]
    cols = [coarse_map[coarse_corners]]
    values = [torch.ones(len(coarse_corners), dtype=torch.float64, device=device)]
    mid_rows = free_map[unique_mid]
    endpoint_cols = coarse_map[unique_ends]
    valid = (mid_rows[:, None] >= 0) & (endpoint_cols >= 0)
    rows.append(mid_rows[:, None].expand(-1, 2)[valid])
    cols.append(endpoint_cols[valid])
    values.append(torch.full((int(valid.sum()),), 0.5, dtype=torch.float64, device=device))
    indices = torch.stack((torch.cat(rows), torch.cat(cols)))
    Pcoo = torch.sparse_coo_tensor(indices, torch.cat(values), (len(free), len(coarse_corners)), dtype=torch.float64, device=device).coalesce()
    P = Pcoo.to_sparse_csr()
    PT = Pcoo.transpose(0, 1).coalesce().to_sparse_csr()
    fixed_mid_free_corner = (mid_rows < 0) & (endpoint_cols >= 0).any(dim=1)
    facts = {'all_original_nodes': n, 'all_cells': len(cells), 'original_corner_nodes': len(corner_ids), 'unique_midpoint_nodes': len(unique_mid), 'free_corner_nodes': len(coarse_corners), 'coarse_vector_dofs': 3 * len(coarse_corners), 'fine_free_nodes': len(free), 'P_scalar_nnz': Pcoo._nnz(), 'corner_mid_disjoint': True, 'mid_to_unique_unordered_edge': True, 'unordered_edge_to_unique_mid': True, 'free_corner_injection_rows': len(coarse_corners), 'coarse_full_column_rank_reason': 'Independent actual free-corner identity rows', 'fixed_mid_with_any_free_corner_count': int(fixed_mid_free_corner.sum()), 'strict_nested_P1_at_fixed_rows': not bool(fixed_mid_free_corner.any()), 'coarse_space_name': 'constraint-masked P1-derived algebraic coarse', 'global_mapping_source': 'Accepted cached cells10 and actual free_node_ids; no coordinates, remeshing or GRIDID-1'}
    ledger.check()
    ledger.account()
    return (P, PT, coarse_corners, coarse_map, free_map, facts)

def assemble_coarse(torch, inputs, arrays, coarse_map, free_map, ledger):
    """Ac=P^T A P; fixed fine-row masks are applied BEFORE local gradients."""
    cells, gradient, weight = (inputs['cells10'], arrays['gradients'], arrays['weights'])
    lam, mu = (float(value) for value in inputs['lame'].tolist())
    device = cells.device
    nc = int((coarse_map >= 0).sum())
    local_P = torch.zeros((10, 4), dtype=torch.float64, device=device)
    local_P[:4] = torch.eye(4, dtype=torch.float64, device=device)
    for i, (a, b) in enumerate(EDGES):
        local_P[4 + i, a] = local_P[4 + i, b] = 0.5
    eye = torch.eye(3, dtype=torch.float64, device=device)
    components = torch.arange(3, dtype=torch.int64, device=device)
    row_chunks, col_chunks, value_chunks = ([], [], [])
    for start in range(0, len(cells), 8192):
        ledger.check()
        ledger.account()
        stop = min(start + 8192, len(cells))
        local = cells[start:stop]
        free_mask = (free_map[local] >= 0).to(torch.float64)
        gc = torch.einsum('nqia,ij,ni->nqja', gradient[start:stop], local_P, free_mask)
        dot = torch.einsum('nqia,nqja->nqij', gc, gc)
        blocks = mu * torch.einsum('nqij,ab->nqiajb', dot, eye)
        blocks = blocks + mu * torch.einsum('nqib,nqja->nqiajb', gc, gc)
        blocks = blocks + lam * torch.einsum('nqia,nqjb->nqiajb', gc, gc)
        local_K = torch.einsum('nqiajb,nq->niajb', blocks, weight[start:stop]).reshape(-1, 12, 12)
        negative(torch.isfinite(local_K).all(), 'Nonfinite Galerkin local stiffness')
        corner_cols = coarse_map[local[:, :4]]
        dofs = (corner_cols[..., None] * 3 + components).reshape(-1, 12)
        valid_dofs = (corner_cols >= 0)[..., None].expand(-1, -1, 3).reshape(-1, 12)
        valid_pairs = valid_dofs[:, :, None] & valid_dofs[:, None, :]
        row_chunks.append(dofs[:, :, None].expand(-1, 12, 12)[valid_pairs])
        col_chunks.append(dofs[:, None, :].expand(-1, 12, 12)[valid_pairs])
        value_chunks.append(local_K[valid_pairs])
        del gc, dot, blocks, local_K, valid_pairs
        ledger.check()
        ledger.account()
    indices = torch.stack((torch.cat(row_chunks), torch.cat(col_chunks)))
    values = torch.cat(value_chunks)
    negative(torch.isfinite(values).all(), 'Nonfinite Galerkin COO values')
    raw_nnz = len(values)
    ledger.check()
    ledger.account()
    Ac = torch.sparse_coo_tensor(indices, values, (3 * nc, 3 * nc), dtype=torch.float64, device=device).coalesce()
    del row_chunks, col_chunks, value_chunks, indices, values
    ledger.check()
    ledger.account()
    delta = (Ac - Ac.transpose(0, 1)).coalesce()
    symmetry = float(torch.linalg.vector_norm(delta.values()) / torch.linalg.vector_norm(Ac.values()))
    negative(math.isfinite(symmetry) and symmetry <= 1e-09, 'Galerkin COO symmetry guard failed')
    Ac = (0.5 * (Ac + Ac.transpose(0, 1))).coalesce()
    ids, values = (Ac.indices(), Ac.values())
    diagonal = torch.zeros(3 * nc, dtype=torch.float64, device=device)
    diagonal_mask = ids[0] == ids[1]
    diagonal.index_add_(0, ids[0, diagonal_mask], values[diagonal_mask])
    negative(torch.isfinite(diagonal).all() and (diagonal > 0).all(), 'Coarse scalar diagonal is nonpositive/nonfinite')
    scaled_abs = values.abs() / torch.sqrt(diagonal[ids[0]] * diagonal[ids[1]])
    row_sum = torch.zeros_like(diagonal)
    row_sum.index_add_(0, ids[0], scaled_abs)
    gamma = float(row_sum.max())
    negative(math.isfinite(gamma) and gamma > 0, 'Nonfinite/nonpositive symmetric-scaled Gershgorin Gamma')
    result = Ac.to_sparse_csr()
    facts = {'raw_COO_entries': raw_nnz, 'coalesced_CSR_nnz': Ac._nnz(), 'vector_shape': [3 * nc, 3 * nc], 'raw_COO_symmetry_relative': symmetry, 'roundoff_symmetrization_after_guard': True, 'coarse_minimum_scalar_diagonal': float(diagonal.min()), 'symmetric_scaled_Gershgorin_Gamma': gamma, 'omega': 1 / gamma, 'Ac_definition': 'P^T A P using accepted curved-P2/4Q gradients with fine fixed rows masked before gc', 'straight_P1_stiffness_used': False}
    ledger.check()
    ledger.account()
    return (result, diagonal, gamma, facts)

class AdditiveCoarse:

    def __init__(self, torch, P, PT, Ac, diagonal, gamma, fine_cholesky, ledger):
        self.torch, self.P, self.PT, self.Ac = (torch, P, PT, Ac)
        self.diagonal, self.omega, self.chol, self.ledger = (diagonal, 1 / gamma, fine_cholesky, ledger)

    def prolong(self, value):
        self.ledger.data['P_calls'] += 1
        return self.torch.sparse.mm(self.P, value)

    def restrict(self, value):
        self.ledger.data['PT_calls'] += 1
        return self.torch.sparse.mm(self.PT, value)

    def coarse_apply(self, value):
        self.ledger.data['coarse_matvec_calls'] += 1
        return self.torch.sparse.mm(self.Ac, value[:, None]).squeeze(1)

    def apply(self, value):
        torch, ledger = (self.torch, self.ledger)
        ledger.sync()
        began = time.monotonic()
        ledger.data['preconditioner_calls'] += 1
        tick = time.monotonic()
        fine = torch.cholesky_solve(value[..., None], self.chol).squeeze(-1)
        ledger.sync()
        ledger.data['block_preconditioner_seconds'] += time.monotonic() - tick
        tick = time.monotonic()
        rhs = self.restrict(value).reshape(-1)
        ledger.sync()
        ledger.data['transfer_seconds'] += time.monotonic() - tick
        tick = time.monotonic()
        zc = torch.zeros_like(rhs)
        for _ in range(32):
            zc = zc + self.omega * (rhs - self.coarse_apply(zc)) / self.diagonal
        ledger.sync()
        ledger.data['coarse_inverse_seconds'] += time.monotonic() - tick
        tick = time.monotonic()
        result = fine + self.prolong(zc.reshape(-1, 3))
        ledger.sync()
        ledger.data['transfer_seconds'] += time.monotonic() - tick
        ledger.data['preconditioner_wall_seconds'] += time.monotonic() - began
        ledger.check_memory()
        return result
