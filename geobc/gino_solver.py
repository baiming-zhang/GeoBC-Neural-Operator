"""Pinned official GINO with explicit cached-CSR case/query execution.

Official embeddings, kernel MLPs, lifting/FNO and projection remain intact.
Only neighbor retrieval and CSR reduction are adapted. Import this module only
in the separately released resource process, never during a source check.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .conditioned_field import PointNetGeometryEncoder

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "_geobc_neuralop_b307030"
COMMIT = "b307030d9b7ee530823d6a05bddb8238ba22a743"


def load_official():
    """Private package paths avoid upstream package training/data imports."""
    root = ROOT / "third_party/neuraloperator/neuralop"
    for suffix, directory in (("", root), (".models", root / "models"), (".layers", root / "layers")):
        name = PREFIX + suffix
        if name in sys.modules:
            if list(sys.modules[name].__path__) != [str(directory)]:
                raise RuntimeError("Private official namespace path changed")
            continue
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    gino = importlib.import_module(PREFIX + ".models.gino")
    integral = importlib.import_module(PREFIX + ".layers.integral_transform")
    if not hasattr(integral, "_geobc_reference_segment_csr"):
        integral._geobc_reference_segment_csr = integral.segment_csr
    integral.segment_csr = gpu_segment_csr
    return gino.GINO, integral


def gpu_segment_csr(src, indptr, reduction, use_scatter=False):
    """GPU segmented sum/mean, including the official zero-empty-row policy.

    A batch loop (one case here) replaces the upstream Python query-row loop.
    No torch-scatter, dense distance matrix, or row/neighbor fallback occurs.
    """
    if src.device.type != "cuda" or src.dtype != torch.float32 or reduction not in ("sum", "mean"):
        raise ValueError("Resource backend requires CUDA float32 sum/mean")
    batched = src.ndim == 3
    if src.ndim not in (2, 3) or indptr.dtype != torch.int64:
        raise ValueError("Expected official unbatched/batched CSR feature schema")
    data = src if batched else src.unsqueeze(0)
    splits = indptr if indptr.ndim == 2 else indptr.unsqueeze(0)
    if len(data) != len(splits):
        raise ValueError("Batched CSR offsets differ")
    output = []
    for sample, offsets in zip(data, splits):
        summed = torch.segment_reduce(sample, "sum", offsets=offsets, axis=0, initial=0.)
        if reduction == "mean":
            counts = offsets[1:] - offsets[:-1]
            # Empty official neighborhoods have zero sum and zero output.
            summed = summed / counts.clamp_min(1).to(sample.dtype)[:, None]
        output.append(summed)
    result = torch.stack(output)
    return result if batched else result[0]


class DisabledNeighborSearch(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("GINO resource adapter requires the validated CSR; native cdist/search is disabled")


def slice_csr(csr, begin, end):
    first, last = int(csr["row_ptr"][begin]), int(csr["row_ptr"][end])
    return {"row_ptr": csr["row_ptr"][begin:end+1] - first,
            "neighbors_index": csr["neighbors_index"][first:last],
            "squared_distances": csr["squared_distances"][first:last]}


def select_csr_rows(csr, rows):
    """Gather complete original rows without dropping any selected neighbor."""
    starts = csr["row_ptr"][rows]
    lengths = csr["row_ptr"][rows+1] - starts
    pointer = torch.cat((torch.zeros(1, dtype=torch.int64, device=rows.device), lengths.cumsum(0)))
    count = int(pointer[-1])
    row_ids = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), lengths)
    positions = starts[row_ids] + torch.arange(count, device=rows.device) - pointer[:-1][row_ids]
    return {"row_ptr": pointer, "neighbors_index": csr["neighbors_index"][positions],
            "squared_distances": csr["squared_distances"][positions]}


@dataclass
class GINOCaseState:
    latent: torch.Tensor
    grid_xyz: torch.Tensor
    point_xyz: torch.Tensor
    point_csr: dict
    input_points: int


class GINOSolver(nn.Module):
    def __init__(self, *, seed=2026, query_chunk=2048):
        super().__init__()
        if seed != 2026 or query_chunk != 2048:
            raise ValueError("This resource plan fixes seed2026 and query chunks2048")
        torch.manual_seed(seed)
        self.encoder = PointNetGeometryEncoder(descriptor_dim=32, point_width=64)
        OfficialGINO, self.integral_module = load_official()
        self.solver = OfficialGINO(in_channels=25, out_channels=4, fno_in_channels=25,
            latent_feature_channels=32, fno_n_modes=(12, 12, 12), fno_hidden_channels=16,
            fno_n_layers=4, in_gno_radius=.033, out_gno_radius=.033,
            gno_weighting_function="half_cos", gno_use_open3d=False, gno_use_torch_scatter=False)
        self.solver.gno_in.neighbor_search = DisabledNeighborSearch()
        self.solver.gno_out.neighbor_search = DisabledNeighborSearch()
        self.query_chunk = query_chunk
        self.source_config = {"official_commit": COMMIT, "version": "2.0.0", "namespace": PREFIX,
            "official_class": "GINO", "input_channels": 25, "output_channels": 4,
            "FNO_input_before_latent": 25, "latent_feature_channels": 32,
            "FNO_input_after_latent": 57, "FNO_hidden": 16, "FNO_modes": [12,12,12], "FNO_layers": 4,
            "seed": seed, "grid_resolution": 64, "input_query_chunk": query_chunk,
            "default_official_parameters": "Remaining GINO constructor defaults, including positional embeddings and kernel MLPs, remain upstream defaults",
            "constructor_overrides": "Open3D/scatter flags False; their searches/reducer are explicitly replaced, not used as fallbacks",
            "case_encoder": "Existing PointNetGeometryEncoder xyz-only1024/width64/global32; live differentiable broadcast to every latent voxel",
            "input_reduction": "Unweighted mean; empty rows zero; cached input dist64 is not passed as weights",
            "output_reduction": "Weighted sum; cached squared-distance64 supplied as official neighbors.weights to original half_cos(squared_distance, sq_radius=.033^2)",
            "runtime_adapter": "Private IntegralTransform module.segment_csr replacement uses GPU torch.segment_reduce sum then count division for mean; official reference retained for small-real-CSR parity",
            "kernel_source_unchanged": True, "neighbor_search_executed": False,
            "checkpoint": "Input IntegralTransform chunks use use_reentrant=False; full edge set retained; no kernel embedding activation for all6.5M edges is saved at once",
            "query_api": "query(state,original_node_ids) uses fixed stored dist64 for parameter gradients. Explicit {xyz,csr} recomputes differentiable squared distances from query/neighbor coordinates; valid radius CSR is supplied externally, never nearest-node copying or implicit cdist. Native query-position gradients/continuous-query behavior remain unvalidated by this resource task",
            "resource_scope": "Full one-CAD input encode, first512 original-node queries, target-independent output.square.mean backward; not full-output training or a continuity/accuracy check"}

    def _embedded_integral(self, block, y_embed, x_embed, f_y, csr, *, output_weights):
        neighbors = {"neighbors_index": csr["neighbors_index"], "neighbors_row_splits": csr["row_ptr"]}
        if output_weights:
            neighbors["weights"] = csr["squared_distances"]
        return block.integral_transform(y=y_embed, x=x_embed, f_y=f_y, neighbors=neighbors)

    def _chunk_integral(self, block, y, x, f_y, csr, *, output_weights, checkpointed):
        y_embed = block.pos_embedding(y) if block.pos_embedding is not None else y
        x_embed = block.pos_embedding(x) if block.pos_embedding is not None else x
        parts = []
        for begin in range(0, len(x), self.query_chunk):
            end = min(begin+self.query_chunk, len(x))
            row = slice_csr(csr, begin, end)
            def execute(y_values, x_values, function, indices, pointers, distances):
                block_csr = {"neighbors_index": indices, "row_ptr": pointers, "squared_distances": distances}
                return self._embedded_integral(block, y_values, x_values, function, block_csr, output_weights=output_weights)
            args = (y_embed, x_embed[begin:end], f_y, row["neighbors_index"], row["row_ptr"], row["squared_distances"])
            if checkpointed and torch.is_grad_enabled():
                value = checkpoint(execute, *args, use_reentrant=False)
            else:
                value = execute(*args)
            parts.append(value)
        return torch.cat(parts, dim=1)

    def encode_case(self, case):
        xyz, features, grid = case["xyz"], case["features"], case["grid_xyz"]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1 or features.shape != (len(xyz),25) or grid.shape != (64**3,3):
            raise ValueError("Encode requires every original point with25 input features on R64")
        context = self.encoder(case["cloud"], case["mask"])
        if context.shape != (1,32):
            raise ValueError("Expected the shared per-case global32")
        lifted = self._chunk_integral(self.solver.gno_in, xyz, grid, features.unsqueeze(0), case["grid_csr"], output_weights=False, checkpointed=True)
        in_p = lifted.reshape(1,64,64,64,25)
        latent_features = context[:,None,None,None,:].expand(1,64,64,64,32)
        latent_grid = self.solver.latent_embedding(torch.cat((in_p, latent_features), dim=-1))
        latent = latent_grid.permute(0,*self.solver.in_coord_dim_reverse_order,1).reshape(1,-1,16)
        if self.solver.out_gno_tanh in ("latent_embed", "both"):
            latent = torch.tanh(latent)
        return GINOCaseState(latent, grid, xyz, case["point_csr"], len(xyz))

    def query(self, state, q):
        if isinstance(q, torch.Tensor):
            if q.ndim != 1 or q.dtype != torch.int64 or bool(((q < 0) | (q >= state.input_points)).any()):
                raise ValueError("Cached-node query requires bounded int64 original node IDs")
            coordinates = state.point_xyz[q]
            csr = select_csr_rows(state.point_csr, q)
        else:
            coordinates, csr = q["xyz"], q["csr"]
            if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(csr["row_ptr"]) != len(coordinates)+1:
                raise ValueError("Explicit query coordinates and their radius CSR differ")
            lengths = csr["row_ptr"][1:] - csr["row_ptr"][:-1]
            owners = torch.repeat_interleave(coordinates.to(torch.float64), lengths, dim=0)
            neighbors = state.grid_xyz[csr["neighbors_index"]].to(torch.float64)
            # A fixed cached distance would omit the half_cos position derivative.
            # Neighbor membership itself is externally supplied and discrete.
            csr = {**csr, "squared_distances": (owners-neighbors).square().sum(dim=-1)}
        value = self._chunk_integral(self.solver.gno_out, state.grid_xyz, coordinates, state.latent, csr, output_weights=True, checkpointed=False)
        return self.solver.projection(value.permute(0,2,1)).permute(0,2,1)[0]

    def reducer_parity(self, case, state, *, atol=1e-6, rtol=1e-5):
        """Two <=8-row real CSR checks of original vs replaced reduction.

        Source neighbors are compacted without removing any selected edge.
        Gradients compare source features and every official kernel parameter.
        These autograd.grad checks are separate from the main resource backward.
        """
        checks = {}
        grid_degree = case["grid_csr"]["row_ptr"][1:] - case["grid_csr"]["row_ptr"][:-1]
        nonempty = torch.nonzero(grid_degree > 0).flatten()[:7]
        empty = torch.nonzero(grid_degree == 0).flatten()[:1]
        input_rows = torch.cat((empty, nonempty))
        fixtures = (("input_mean", self.solver.gno_in, case["xyz"], case["grid_xyz"][input_rows],
                     case["features"].unsqueeze(0), select_csr_rows(case["grid_csr"], input_rows), False),
                    ("output_half_cos_sum", self.solver.gno_out, state.grid_xyz, case["xyz"][:8],
                     state.latent.detach(), slice_csr(state.point_csr,0,8), True))
        for name, block, y, x, functions, csr, weighted in fixtures:
            unique, inverse = torch.unique(csr["neighbors_index"], sorted=True, return_inverse=True)
            compact = {"row_ptr": csr["row_ptr"], "neighbors_index": inverse, "squared_distances": csr["squared_distances"]}
            y_embed = block.pos_embedding(y[unique]) if block.pos_embedding is not None else y[unique]
            x_embed = block.pos_embedding(x) if block.pos_embedding is not None else x
            parameters = list(block.integral_transform.channel_mlp.parameters())
            outputs, gradients = [], []
            for reference in (True, False):
                function = functions[:,unique,:].detach().clone().requires_grad_(True)
                self.integral_module.segment_csr = self.integral_module._geobc_reference_segment_csr if reference else gpu_segment_csr
                try:
                    out = self._embedded_integral(block, y_embed, x_embed, function, compact, output_weights=weighted)
                    grad = torch.autograd.grad(out.square().mean(), [function]+parameters)
                finally:
                    self.integral_module.segment_csr = gpu_segment_csr
                outputs.append(out.detach()); gradients.append([value.detach() for value in grad])
            value_ok = bool(torch.allclose(outputs[0], outputs[1], atol=atol, rtol=rtol))
            grad_ok = all(bool(torch.allclose(a,b,atol=atol,rtol=rtol)) for a,b in zip(*gradients))
            checks[name] = {"rows": len(x), "complete_selected_edges": len(csr["neighbors_index"]),
                "compact_actual_source_points": len(unique), "value_max_abs_difference": float((outputs[0]-outputs[1]).abs().max()),
                "gradient_max_abs_difference": max(float((a-b).abs().max()) if a.numel() else 0. for a,b in zip(*gradients)),
                "values_passed": value_ok, "gradients_passed": grad_ok, "atol": atol, "rtol": rtol,
                "finite": all(bool(torch.isfinite(value).all()) for seq in gradients for value in seq),
                "source": "Official IntegralTransform unchanged; only segment_csr function is temporarily restored/replaced",
                "empty_input_row_included": bool(len(empty)) if not weighted else False}
        return {"passed": all(row["values_passed"] and row["gradients_passed"] and row["finite"] for row in checks.values()),
                "checks": checks, "autograd_grad_calls": 4, "optimizer_steps": 0, "synthetic_rows": 0}
