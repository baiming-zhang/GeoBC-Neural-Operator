"""Complete original-node multiscale mesh solver for a GeoBC screen.

The data producer supplies fixed, target-independent W/R/P operators. Nine
distinct learned message blocks are shared across four V-cycles. This is a
new mesh adaptation, not an exact implementation of MGKN or BSMS. The model
reads no fields, fits no scaler, and exposes no arbitrary off-node query API.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .conditioned_field import PointNetGeometryEncoder

LATENT_DIM = 32
CONTEXT_DIM = 32
CLOUD_POINTS = 1024
FEATURE_DIM = 25
V_CYCLES = 4
EDGE_CHUNK = 65536
OUTPUT_CHANNELS = ("ux_mm", "uy_mm", "uz_mm", "source_signed_stress_MPa")
BLOCK_OPERATORS = {
    "fine_before": "W0", "down01": "R01", "mid_before": "W1",
    "down12": "R12", "bottom": "W2", "up12": "P12",
    "mid_after": "W1", "up01": "P01", "fine_after": "W0",
}
OPERATOR_LEVELS = {
    "W0": (0, 0), "W1": (1, 1), "W2": (2, 2),
    "R01": (0, 1), "R12": (1, 2), "P12": (2, 1), "P01": (1, 0),
}


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _scaler_metadata(scaler: dict) -> dict:
    """Validate supplied TRAIN-only JSON; do not inspect or fit targets."""
    if (not isinstance(scaler, dict) or scaler.get("channels") != [0, 1, 2, 3]
            or len(scaler.get("mean", [])) != 4 or len(scaler.get("std", [])) != 4):
        raise ValueError("Expected the existing four-field TRAIN-only scaler")
    mean, std = [float(x) for x in scaler["mean"]], [float(x) for x in scaler["std"]]
    stress_scale = float(scaler.get("stress_scale", 0.))
    if (not all(math.isfinite(x) for x in mean+std+[stress_scale])
            or min(std) <= 0 or stress_scale <= 0):
        raise ValueError("Invalid shared signed-field scaler metadata")
    return {"mean": mean, "std": std, "stress_scale": stress_scale,
            "scaler_sha256": _digest(scaler), "scaler_fits": 0}


class _MessageBlock(nn.Module):
    """Weighted complete-edge aggregate followed by affine residual LN32."""

    def __init__(self) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(68, 64), nn.GELU(), nn.Linear(64, 32))
        self.norm = nn.LayerNorm(32, elementwise_affine=True)

    def _edge_messages(self, sender: Tensor, receiver: Tensor, src: Tensor,
                       dst: Tensor, delta_xyz: Tensor, length: Tensor,
                       weight: Tensor) -> Tensor:
        # Gathers, hidden activations and weights all remain inside checkpoint.
        inputs = torch.cat((sender.index_select(0, src), receiver.index_select(0, dst),
                            delta_xyz, length), dim=-1)
        return self.message(inputs)*weight.unsqueeze(-1)

    def forward(self, sender: Tensor, receiver: Tensor, operator: Mapping) -> Tensor:
        # One receiver buffer per block, never one full-N buffer per edge chunk.
        aggregate = receiver.new_zeros(receiver.shape)
        edges = operator["src"].numel()
        for start in range(0, edges, EDGE_CHUNK):
            end = min(start+EDGE_CHUNK, edges)
            args = (sender, receiver, operator["src"][start:end], operator["dst"][start:end],
                    operator["delta_xyz"][start:end], operator["length"][start:end],
                    operator["weight"][start:end])
            if self.training and torch.is_grad_enabled():
                messages = checkpoint(self._edge_messages, *args, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                messages = self._edge_messages(*args)
            aggregate.index_add_(0, args[3], messages)
        # An isolated W row has zero aggregate, but still receives LN(receiver).
        # There is no identity/BC override, learned step size, or residual alpha.
        return self.norm(receiver+aggregate)


class _MultiscaleSolver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lift = nn.Sequential(nn.Linear(60, 64), nn.GELU(), nn.Linear(64, 32))
        self.blocks = nn.ModuleDict({name: _MessageBlock() for name in BLOCK_OPERATORS})
        self.readout = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 4))

    def _cycle(self, h0: Tensor, h1: Tensor, h2: Tensor,
               hierarchy: Mapping) -> tuple[Tensor, Tensor, Tensor]:
        b = self.blocks
        h0 = b["fine_before"](h0, h0, hierarchy["W0"])
        h1 = b["down01"](h0, h1, hierarchy["R01"])
        h1 = b["mid_before"](h1, h1, hierarchy["W1"])
        h2 = b["down12"](h1, h2, hierarchy["R12"])
        h2 = b["bottom"](h2, h2, hierarchy["W2"])
        # Receiver states are the current cycle's fine/mid skips; sender states
        # are the just-updated bottom/mid. Coarse states persist into next cycle.
        h1 = b["up12"](h2, h1, hierarchy["P12"])
        h1 = b["mid_after"](h1, h1, hierarchy["W1"])
        h0 = b["up01"](h1, h0, hierarchy["P01"])
        h0 = b["fine_after"](h0, h0, hierarchy["W0"])
        return h0, h1, h2

    def encode(self, point_inputs: Tensor, hierarchy: Mapping) -> Tensor:
        h0 = self.lift(point_inputs)
        # Fixed complete hard-mass restriction gives differentiable linear
        # seeds once. No dense operator and no learned down block in this seed.
        h1 = torch.sparse.mm(hierarchy["R01"]["csr"], h0)
        h2 = torch.sparse.mm(hierarchy["R12"]["csr"], h1)
        for _ in range(V_CYCLES):
            if self.training and torch.is_grad_enabled():
                h0, h1, h2 = checkpoint(self._cycle, h0, h1, h2, hierarchy,
                                        use_reentrant=False, preserve_rng_state=False)
            else:
                h0, h1, h2 = self._cycle(h0, h1, h2, hierarchy)
        return h0


class MultiscaleMeshField(nn.Module):
    """Full-case encode -> node states -> optionally chunked encoded readout.

    The only live dependency needed by readout is H0. A trainer may accumulate
    all node readout losses through an H0 leaf, then call original_H0.backward
    once with the accumulated leaf gradient. Do not detach inside encode_case.
    """

    def __init__(self, scaler: dict, seed: int = 2026,
                 device: str | torch.device = "cuda") -> None:
        super().__init__()
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.seed = seed
        self.method = "multiscale_mesh"
        self.backend = "torch_complete_mesh_message_passing"
        self._scaler = _scaler_metadata(scaler)
        # Same old xyz-only PointNet constructor/creation order as prior solver
        # screens. This does not claim full-model matching across architectures.
        torch.manual_seed(seed)
        self.encoder = PointNetGeometryEncoder(descriptor_dim=32, point_width=64)
        self.solver = _MultiscaleSolver()
        self.to(device)

    def parameter_counts(self) -> dict:
        encoder = sum(p.numel() for p in self.encoder.parameters())
        lift = sum(p.numel() for p in self.solver.lift.parameters())
        blocks = {name: sum(p.numel() for p in block.parameters())
                  for name, block in self.solver.blocks.items()}
        head = sum(p.numel() for p in self.solver.readout.parameters())
        return {"PointNet": encoder, "lift": lift, "message_blocks": blocks,
                "message_blocks_total": sum(blocks.values()), "readout": head,
                "solver": lift+sum(blocks.values())+head,
                "total": sum(p.numel() for p in self.parameters()),
                "all_parameters_real": True, "complex_parameters": 0,
                "count_policy": "actual public Parameter.numel, not capacity matching"}

    def gradient_components(self) -> dict[str, nn.Module]:
        # Nonstored ModuleLists make .parameters() available to the trainer;
        # they do not register aliases in this model/state_dict/optimizer.
        b = self.solver.blocks
        return {"encoder": self.encoder, "lift": self.solver.lift,
                "fineMP": nn.ModuleList([b["fine_before"], b["fine_after"]]),
                "down": nn.ModuleList([b["down01"], b["down12"]]),
                "up": nn.ModuleList([b["up12"], b["up01"]]),
                "coarse": nn.ModuleList([b["mid_before"], b["bottom"], b["mid_after"]]),
                "readout": self.solver.readout}

    @property
    def source_config(self) -> dict:
        own_path = Path(__file__).resolve()
        encoder_path = own_path.with_name("conditioned_field.py")
        return {
            "method": self.method, "backend": self.backend, "seed": self.seed,
            "source": "geobc/multiscale_mesh_field.py",
            "source_sha256": hashlib.sha256(own_path.read_bytes()).hexdigest(),
            "encoder_source": "geobc/conditioned_field.py:PointNetGeometryEncoder",
            "encoder_source_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
            "parameter_counts": self.parameter_counts(),
            "scaler": copy.deepcopy(self._scaler),
            "input": {"node": "canonical normalized xyz3 + all original25 + xyz-cloud global32",
                      "lift_dimensions": [60, 64, 32], "force_slots": list(range(7, 13)),
                      "force_slots_retained": True, "cloud_shape": [1, 1024, 3],
                      "cloud_target_independent": True, "cloud_load_independent": True,
                      "global_dimension": 32, "node_order": "all original aligned FEM nodes"},
            "encoder": {"point_mlp": [3, 32, 64, 64], "head": [128, 64, 32],
                        "activation": "ReLU", "pooling": "masked max + masked mean",
                        "output": "non-affine LN32 then tanh"},
            "mesh": {"levels": 3, "node_counts": "N0 original; N1/N2 actual variable connected aggregates",
                     "graph": "all10 TETRA10 nodes with original corner-midpoint subedges",
                     "coarse_graph": "actual cut multiplicity; no new geometric shortcut",
                     "roles": "input memberships preserved by unsmoothed ownership, not hard Dirichlet BC",
                     "coordinates": "canonical isotropic /L, unchanged; no second normalization",
                     "edge_delta": "receiver_xyz[dst] - sender_xyz[src]",
                     "edge_length": "norm(edge_delta) in canonical normalized coordinates",
                     "CSR_orientation": "row=receiver/dst, column=sender/src",
                     "W_weights": "actual symmetric cut multiplicity / receiver weighted degree",
                     "R_weights": "complete hard ownership mass pooling; mass is V/10 proxy",
                     "P_weights": "complete two-pass positive Jacobi smoothed aggregation map",
                     "initial_seeds": ["h0=lift(inputs)", "h1=R01@h0", "h2=R12@h1"],
                     "initial_seed_autograd": "torch.sparse.mm through fixed CSR to node states",
                     "coarse_state_policy": "persistent states across all cycles, initialized only once",
                     "support_policy": "no neighbor truncation, resampling, hard BC override or stiffness interpretation"},
            "V_cycle": {"repetitions": 4, "nine_block_order": list(BLOCK_OPERATORS),
                        "block_operators": dict(BLOCK_OPERATORS),
                        "parameter_sharing": "each of nine blocks is distinct; only shared across four cycles",
                        "message_dimensions": [68, 64, 32],
                        "message_input_order": ["sender32", "receiver32", "receiver-minus-sender3", "length1"],
                        "message_hidden_activation": "GELU", "message_output_activation": "linear",
                        "update": "LN32_affine(receiver + sum_edge weight * message)",
                        "LayerNorms": "nine independent affine LN32, one per block",
                        "isolated_W_rows": "zero aggregate, still LN(receiver); no identity special case",
                        "down_receiver": "current coarse state",
                        "up_receiver": "this cycle's corresponding mid/fine skip state",
                        "sender": "just updated source-level state",
                        "learned_residual_alpha": False},
            "output": {"head_dimensions": [32, 128, 4], "hidden_activation": "GELU",
                       "channels": list(OUTPUT_CHANNELS),
                       "values": "four shared TRAIN-scaler encoded targets; trainer decodes physical mm/MPa",
                       "response_basis": False, "signed_scalar_stress_linear": False,
                       "stress_tensor_or_constitutive_formula": False},
            "memory": {"edge_chunk": EDGE_CHUNK, "message_checkpoint": "non-reentrant per edge chunk",
                       "V_cycle_checkpoint": "non-reentrant per cycle during gradient-enabled training",
                       "preserve_rng_state": False, "stochastic_layers": False,
                       "aggregation": "one Ndstx32 accumulator per block with chunk index_add_",
                       "E_by_32_by_32": False, "dense_N_by_N": False,
                       "per_chunk_full_node_accumulator": False,
                       "readout_leaf_chainrule": "only final H0 is needed; one complete encode backward after all readout chunks",
                       "actual_GPU_memory_and_sparse_backward": "deferred to first formal full-case step"},
            "native_offnode": False,
            "limitations": ["new multiscale mesh adaptation, not exact MGKN/BSMS reproduction",
                            "no strict capacity or full-model initialization matching with prior solver families",
                            "graph weights are not stiffness or FEM quadrature message kernels",
                            "membership flags are not node-wise Dirichlet values; no BC enforcement claim",
                            "node solver has no arbitrary off-node25 extension or continuous-gradient guarantee",
                            "no training, accuracy, resource or engineering qualification follows from source readiness"],
        }

    def _validate_hierarchy(self, xyz: Tensor, hierarchy: Mapping) -> None:
        if not isinstance(hierarchy, Mapping):
            raise ValueError("Expected the target-independent hierarchy mapping")
        model_device = next(self.solver.parameters()).device
        sizes = []
        for level in range(3):
            coordinates = hierarchy.get(f"xyz{level}")
            if (not isinstance(coordinates, Tensor) or coordinates.ndim != 2
                    or coordinates.shape[1] != 3 or len(coordinates) < 1
                    or coordinates.dtype != torch.float32 or coordinates.device != model_device):
                raise ValueError(f"xyz{level} must be nonempty float32 [N{level},3] on model device")
            sizes.append(len(coordinates))
        if sizes[0] != len(xyz):
            raise ValueError("Hierarchy does not preserve all original node rows")
        for name, (source_level, destination_level) in OPERATOR_LEVELS.items():
            operator = hierarchy.get(name)
            shape = (sizes[destination_level], sizes[source_level])
            if not isinstance(operator, Mapping) or tuple(operator.get("shape", ())) != shape:
                raise ValueError(f"{name} shape must be (Ndst,Nsrc)={shape}")
            src = operator.get("src")
            if not isinstance(src, Tensor) or src.ndim != 1:
                raise ValueError(f"{name}.src must be int64[E]")
            edges = len(src)
            for key, expected_shape, expected_dtype in (
                    ("src", (edges,), torch.int64), ("dst", (edges,), torch.int64),
                    ("weight", (edges,), torch.float32), ("delta_xyz", (edges, 3), torch.float32),
                    ("length", (edges, 1), torch.float32),
                    ("indptr", (shape[0]+1,), torch.int64)):
                value = operator.get(key)
                if (not isinstance(value, Tensor) or tuple(value.shape) != expected_shape
                        or value.dtype != expected_dtype or value.device != model_device):
                    raise ValueError(f"{name}.{key} shape/dtype/device mismatch")
            csr = operator.get("csr")
            if (not isinstance(csr, Tensor) or csr.layout != torch.sparse_csr
                    or tuple(csr.shape) != shape or csr.dtype != torch.float32 or csr.device != model_device):
                raise ValueError(f"{name}.csr must be float32 sparse CSR with receiver rows")
        # Actual bounds, source/xyz identities, row sums and complete support are
        # producer/loader guards; this shape check performs no full-array scan.

    def encode_case(self, xyz: Tensor, features25: Tensor, cloud_xyz: Tensor,
                    mask: Tensor, hierarchy: Mapping) -> Tensor:
        device = next(self.solver.parameters()).device
        if (not isinstance(xyz, Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1
                or not xyz.is_floating_point() or xyz.device != device
                or not isinstance(features25, Tensor) or features25.shape != (len(xyz), 25)
                or not features25.is_floating_point() or features25.device != device
                or not isinstance(cloud_xyz, Tensor) or cloud_xyz.shape != (1, 1024, 3)
                or not cloud_xyz.is_floating_point() or cloud_xyz.device != device
                or not isinstance(mask, Tensor) or mask.shape != (1, 1024)
                or mask.dtype != torch.bool or mask.device != device):
            raise ValueError("Expected original xyz[N,3], features25[N,25], cloud[1,1024,3], bool mask")
        self._validate_hierarchy(xyz, hierarchy)
        context = self.encoder(cloud_xyz.float(), mask)
        point_inputs = torch.cat((xyz.float(), features25.float(), context.expand(len(xyz), -1)), dim=-1)
        states = self.solver.encode(point_inputs, hierarchy)
        if states.shape != (len(xyz), 32):
            raise ValueError("Full-node final state shape changed")
        return states

    def readout(self, H0: Tensor) -> Tensor:
        if (not isinstance(H0, Tensor) or H0.ndim != 2 or H0.shape[1] != 32
                or len(H0) < 1 or not H0.is_floating_point()
                or H0.device != next(self.solver.parameters()).device):
            raise ValueError("readout requires [N,32] final node states on model device")
        return self.solver.readout(H0.float())

    def forward_case(self, xyz: Tensor, features25: Tensor, cloud_xyz: Tensor,
                     mask: Tensor, hierarchy: Mapping) -> Tensor:
        return self.readout(self.encode_case(xyz, features25, cloud_xyz, mask, hierarchy))

    def forward(self, xyz: Tensor, features25: Tensor, cloud_xyz: Tensor,
                mask: Tensor, hierarchy: Mapping) -> Tensor:
        return self.forward_case(xyz, features25, cloud_xyz, mask, hierarchy)

