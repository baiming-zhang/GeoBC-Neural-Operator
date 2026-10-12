"""Bezier-hull spatial index and curved-cell point location."""

from .query import shape

class LocatorNegative(ValueError):
    """Fixed scientific capacity/ownership gate failure."""

class PhysicalULocator:

    def __init__(self, torch, vertices, cells, guard):
        self.t, self.x, self.cells, self.guard = (torch, vertices, cells, guard)
        t = torch
        assert vertices.is_cuda and cells.is_cuda
        assert vertices.dtype == t.float64 and cells.dtype == t.int64
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert cells.ndim == 2 and cells.shape[1] == 10
        assert bool(t.isfinite(vertices).all()) and bool(((cells >= 0) & (cells < len(vertices))).all())
        self.bins = 64
        edges = t.tensor(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)), device=vertices.device)
        low, high = ([], [])
        for start in range(0, len(cells), 8192):
            guard()
            nodes = vertices[cells[start:start + 8192]]
            controls = 2 * nodes[:, 4:] - 0.5 * (nodes[:, edges[:, 0]] + nodes[:, edges[:, 1]])
            hull = t.cat((nodes[:, :4], controls), 1)
            low.append(hull.amin(1))
            high.append(hull.amax(1))
        self.low, self.high = (t.cat(low), t.cat(high))
        self.domain_low, self.domain_high = (self.low.amin(0), self.high.amax(0))
        self.extent = self.domain_high - self.domain_low
        assert bool((self.extent > 0).all())
        lo = t.floor((self.low - self.domain_low) / self.extent * 64).to(t.int64).clamp(0, 63)
        hi = t.floor((self.high - self.domain_low) / self.extent * 64).to(t.int64).clamp(0, 63)
        scaled = (self.low - self.domain_low) / self.extent * 64
        lo = t.where(scaled == t.floor(scaled), (lo - 1).clamp(0, 63), lo)
        span = hi - lo + 1
        if int(span.max()) > 16:
            raise LocatorNegative('cell bin span exceeds fixed16 per axis')
        refs = int(span.prod(1).sum())
        if refs > 8000000:
            raise LocatorNegative('cell-bin references exceed fixed8million')
        keys, owners = ([], [])
        for start in range(0, len(cells), 8192):
            guard()
            stop = min(start + 8192, len(cells))
            sp = span[start:stop]
            mx = sp.amax(0).tolist()
            offsets = t.cartesian_prod(*(t.arange(v, device=vertices.device) for v in mx))
            if offsets.ndim == 1:
                offsets = offsets.reshape(-1, 3)
            mask = (offsets[None] < sp[:, None]).all(-1)
            ij = mask.nonzero()
            bins = lo[start:stop][ij[:, 0]] + offsets[ij[:, 1]]
            keys.append(bins[:, 0] * 4096 + bins[:, 1] * 64 + bins[:, 2])
            owners.append(ij[:, 0] + start)
        key, owner = (t.cat(keys), t.cat(owners))
        assert len(key) == refs
        order = t.argsort(key, stable=True)
        self.owners = owner[order]
        counts = t.bincount(key, minlength=64 ** 3)
        self.ptr = t.cat((t.zeros(1, dtype=t.int64, device=vertices.device), counts.cumsum(0)))
        self.metadata = dict(original_cells=len(cells), indexed_cells=len(cells), cell_bin_references=refs, maximum_axis_span=int(span.max()), occupied_bins=int((counts > 0).sum()), maximum_bucket_candidates=int(counts.max()), domain_low=self.domain_low.tolist(), domain_high=self.domain_high.tolist(), dtype='float64', device='cuda', bounds='quadratic_Bezier_control_hull', original_reference_owners_used=False)
        guard()

    def _inverse_hits(self, xyz, owner):
        """Reject individual candidate inverses without rejecting other owners."""
        t = self.t
        nodes = self.x[self.cells[owner]]
        scale = t.linalg.vector_norm(nodes.amax(1) - nodes.amin(1), dim=-1)
        valid = t.isfinite(scale) & (scale > 0)
        r = t.full((len(owner), 3), 0.25, dtype=t.float64, device=self.x.device)
        orientation = t.zeros(len(owner), dtype=t.float64, device=self.x.device)
        eye = t.eye(3, dtype=t.float64, device=self.x.device).expand(len(owner), -1, -1)
        for iteration in range(16):
            self.guard()
            N, D, _ = shape(t, r)
            mapped = t.einsum('qi,qia->qa', N, nodes)
            J = t.einsum('qia,qib->qab', nodes, D)
            valid &= t.isfinite(J).all((1, 2)) & t.isfinite(mapped).all(1)
            safeJ = t.where(valid[:, None, None], J, eye)
            sv = t.linalg.svdvals(safeJ)
            det = t.linalg.det(safeJ)
            valid &= (sv[:, -1] > 0) & (sv[:, 0] <= 1000000000000.0 * sv[:, -1])
            valid &= det.abs() > 1e-14 * scale ** 3
            if iteration == 0:
                orientation = det.sign()
            else:
                valid &= det.sign() == orientation
            safeJ = t.where(valid[:, None, None], J, eye)
            rhs = t.where(valid[:, None], mapped - xyz, t.zeros_like(xyz))
            delta, info = t.linalg.solve_ex(safeJ, rhs, check_errors=False)
            valid &= (info == 0) & t.isfinite(delta).all(1)
            r = t.where(valid[:, None], r - delta, t.full_like(r, 0.25))
        N, _, b = shape(t, r)
        mapped = t.einsum('qi,qia->qa', N, nodes)
        valid &= t.isfinite(mapped).all(1) & t.isfinite(b).all(1)
        valid &= t.linalg.vector_norm(mapped - xyz, dim=-1) <= 1e-09 * scale
        valid &= ((b >= -1e-10) & (b <= 1 + 1e-10)).all(1)
        strict = valid & ((b > 1e-10) & (b < 1 - 1e-10)).all(1)
        return (valid, strict)

    def locate(self, xyz):
        t = self.t
        assert xyz.is_cuda and xyz.dtype == t.float64 and (xyz.ndim == 2) and (xyz.shape[1] == 3)
        assert bool(t.isfinite(xyz).all())
        found = t.full((len(xyz),), -1, dtype=t.int64, device=self.x.device)
        records = []
        for start in range(0, len(xyz), 64):
            self.guard()
            q = xyz[start:start + 64]
            inside = ((q >= self.domain_low) & (q <= self.domain_high)).all(1)
            bins = t.floor((q - self.domain_low) / self.extent * 64).to(t.int64).clamp(0, 63)
            key = bins[:, 0] * 4096 + bins[:, 1] * 64 + bins[:, 2]
            count = t.where(inside, self.ptr[key + 1] - self.ptr[key], t.zeros_like(key))
            if int(count.max()) > 4096:
                raise LocatorNegative('query candidates exceed fixed4096')
            qi = t.repeat_interleave(t.arange(len(q), device=q.device), count)
            if len(qi):
                cumulative = count.cumsum(0) - count
                offset = t.arange(len(qi), device=q.device) - t.repeat_interleave(cumulative, count)
                owner = self.owners[self.ptr[key[qi]] + offset]
                mask = ((q[qi] >= self.low[owner]) & (q[qi] <= self.high[owner])).all(1)
                qi, owner = (qi[mask], owner[mask])
                valid, strict = self._inverse_hits(q[qi], owner) if len(qi) else (t.zeros(0, dtype=t.bool, device=q.device),) * 2
            else:
                owner = t.zeros(0, dtype=t.int64, device=q.device)
                valid = strict = t.zeros(0, dtype=t.bool, device=q.device)
            for j in range(len(q)):
                hits = owner[(qi == j) & valid].unique(sorted=True)
                strict_hits = owner[(qi == j) & strict].unique(sorted=True)
                if len(strict_hits) > 1:
                    status = 'ambiguous_negative'
                elif len(strict_hits) == 1:
                    status = 'strict_interior'
                    found[start + j] = strict_hits[0]
                elif len(hits):
                    status = 'boundary_tie'
                    found[start + j] = hits[0]
                else:
                    status = 'not_found' if bool(inside[j]) else 'outside_global_box'
                records.append(dict(status=status, bucket_candidates=int(count[j]), inverse_candidates=int((qi == j).sum()), owners=hits.tolist(), strict_owners=strict_hits.tolist(), selected_owner=int(found[start + j])))
        return (found, records)
