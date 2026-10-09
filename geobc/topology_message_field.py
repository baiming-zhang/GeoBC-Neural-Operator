"""Isolated two-level mesh-message displacement-basis fitting candidate.

Both controls share the same fixed graph hierarchy and positive R/P maps.
Only learned-message A1/A2 are identity or normalized real cut adjacency.
The field queries original node IDs; arbitrary-xyz interpolation is undefined.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Literal

import torch
from torch import Tensor, nn

from .bc_role_field import PointNetBCRoleEncoder, _encoder_record
from .hash_bandwidth_field import SplitHashBandwidthDecoder
from .physics_field import combine_basis, REFERENCE_LOAD_ORDER, COMPONENT_ORDER
from .tcnn_fields import TCNNFieldConfig

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "05fd5a5afba10871c58d18fde2e31997b85b1c1aa695b828e4ee416c43e43e68"
CAD_ORDER = ("101_428", "101_479", "101_56", "101_78")
EXPECTED_TOPOLOGY_PARAMETERS = 9856
EXPECTED_FULL_PARAMETERS = 7797152


def _require(value, message):
    if not value:
        raise ValueError(message)


def _path(value):
    value = str(value).replace("\\", "/")
    if os.name == "nt" and value.startswith("/mnt/"):
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    result = Path(value)
    return result if result.is_absolute() else ROOT / result


def _sha(path):
    return hashlib.sha256(_path(path).read_bytes()).hexdigest()


def _match_stat(path, record):
    actual = _path(path).stat()
    _require(actual.st_size == record["bytes"] and actual.st_mtime_ns == record["mtime_ns"],
             f"Frozen input stat changed: {path}")


def load_topology_manifest(directory, *, plan_path=ROOT / "notes/geobc_topology_message_plan.json"):
    """JSON/small SHA/stat only. No NPZ, target, model or native construction."""
    directory = _path(directory)
    _require(_sha(plan_path) == PLAN_SHA, "Frozen topology-message plan changed")
    plan = json.loads(_path(plan_path).read_text(encoding="utf-8"))
    evidence = plan["input_evidence"]
    _require(plan["status"] == "predeclared" and plan["scope"]["CAD_order"] == list(CAD_ORDER)
             and directory / "manifest.json" == _path(evidence["cache_manifest"]), "Only frozen four-TRAIN topology cache allowed")
    _require(_sha(directory / "manifest.json") == evidence["cache_manifest_sha256"]
             and _sha(evidence["result"]) == evidence["result_sha256"]
             and _sha(evidence["preparation_source"]) == evidence["preparation_source_sha256"], "Frozen preparation identity differs")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest["status"] == "completed" and manifest["feasibility_pass"] is True
             and manifest["outcome"] == "feasible" and set(manifest["CADs"]) == set(CAD_ORDER)
             and manifest["contract"]["CAD_order"] == list(CAD_ORDER), "Input feasibility must be completed for exactly these CADs")
    for cad in CAD_ORDER:
        record = manifest["CADs"][cad]
        _require(record["status"] == "completed" and record["labels_read"] == 0 and record["target_independent"]
                 and record["facts"]["N0"] == evidence["CADs"][cad]["N0"]
                 and record["facts"]["N1"] == 2048 and record["facts"]["N2"] == 128
                 and record["file_identity"] == evidence["CADs"][cad]["file_identity"], "Selected completed hierarchy identity differs")
        _match_stat(directory / record["file"], record["file_identity"])
        for key in ("prepared", "gradient_geometry", "interface_inputs"):
            _match_stat(record["source"][key]["path"], record["source"][key])
    return manifest


@dataclass(frozen=True)
class TopologyInput:
    """Fixed per-CAD tensors, all nontrainable, in explicit original node order."""
    cad_id: str
    descriptor47: Tensor
    R12: Tensor
    P12: Tensor
    P01: Tensor
    A1_neighbors: Tensor
    A2_neighbors: Tensor
    node_count: int
    metadata: dict

    @property
    def device(self):
        return self.descriptor47.device

    def record(self):
        return self.metadata


def _csr(stored, name, shape, *, stochastic=False):
    """Read only named CSR input members, preserving full stored support."""
    import numpy as np
    from scipy.sparse import csr_matrix
    indptr, indices, data = (stored[name + suffix] for suffix in ("_indptr", "_indices", "_data"))
    actual_shape = tuple(int(value) for value in stored[name + "_shape"])
    _require(actual_shape == tuple(shape) and indptr.dtype == indices.dtype == np.dtype("int64")
             and indptr.shape == (shape[0] + 1,) and indptr[0] == 0 and indptr[-1] == len(indices) == len(data)
             and (np.diff(indptr) >= 0).all() and (indices >= 0).all() and (indices < shape[1]).all()
             and np.isfinite(data).all() and (data > 0).all(), f"Invalid complete input CSR {name}")
    _require(data.dtype == np.dtype("float32") if stochastic else data.dtype == np.dtype("float64"),
             f"Stored CSR dtype differs: {name}")
    result = csr_matrix((data, indices, indptr), shape=shape)
    _require(result.has_sorted_indices and result.has_canonical_format, f"Uncoalesced/unsorted {name}")
    if stochastic:
        error = abs(np.asarray(result.astype(np.float64).sum(1)).ravel() - 1).max()
        _require(error <= 1e-6 and (np.diff(indptr) > 0).all(), f"Incomplete/nonconstant-preserving {name}")
    else:
        _require((result - result.T).nnz == 0 and (result.diagonal() == 0).all()
                 and (data == np.round(data)).all(), f"Invalid actual cut adjacency {name}")
    return result


def _neighbor_operator(W):
    import numpy as np
    from scipy.sparse import diags
    degree = np.asarray(W.sum(1)).ravel()
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = 1 / degree[degree > 0]
    A = (diags(inverse) @ W + diags((degree == 0).astype(np.float64))).tocsr()
    A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
    _require(np.isfinite(A.data).all() and (A.data > 0).all()
             and abs(np.asarray(A.sum(1)).ravel() - 1).max() <= 1e-12, "Neighbor message operator is not stochastic")
    return A.astype(np.float32)


def _torch_csr(matrix, device):
    return torch.sparse_csr_tensor(torch.as_tensor(matrix.indptr, dtype=torch.long, device=device),
        torch.as_tensor(matrix.indices, dtype=torch.long, device=device),
        torch.as_tensor(matrix.data, dtype=torch.float32, device=device), size=matrix.shape, device=device)


def build_descriptor47(features43, R01, xyz1_normalized, mass1_mm3):
    """CPU static assembly: hard mass mean of masked43, centroid3, mass ratio1."""
    import numpy as np
    if isinstance(features43, Tensor):
        features43 = features43.detach().cpu().numpy()
    features = np.asarray(features43)
    _require(features.dtype == np.dtype("float32") and features.shape == (R01.shape[1], 43)
             and np.isfinite(features).all(), "Finite canonical float32 original43 required")
    masked = features.copy()
    masked[:, 7:13] = 0
    mass = np.asarray(mass1_mm3, np.float64)
    centers = np.asarray(xyz1_normalized, np.float32)
    _require(mass.shape == (R01.shape[0],) and centers.shape == (R01.shape[0], 3)
             and np.isfinite(mass).all() and (mass > 0).all() and np.isfinite(centers).all(), "Invalid aggregate geometry")
    descriptor = np.concatenate((np.asarray(R01 @ masked, np.float32), centers,
                                (mass / mass.mean()).astype(np.float32)[:, None]), axis=1)
    _require(descriptor.shape == (R01.shape[0], 47) and np.isfinite(descriptor).all()
             and (descriptor[:, 7:13] == 0).all(), "Descriptor47 assembly failed")
    return descriptor


def load_topology_input(directory, cad_id, *, xyz, features43, manifest=None, device="cpu"):
    """Only frozen TRAIN input cache/maps and caller-supplied static43 are read.

    Descriptor assembly happens once on CPU. Both message modes use the same
    returned object. The caller verifies all three load inputs match after the
    force/moment mask; no targets or original raw geometry are loaded here.
    """
    import numpy as np
    directory = _path(directory)
    manifest = manifest or load_topology_manifest(directory)
    _require(cad_id in CAD_ORDER and cad_id in manifest["CADs"], "Only the four declared TRAIN CADs may load arrays")
    record = manifest["CADs"][cad_id]
    _match_stat(directory / record["file"], record["file_identity"])
    values = xyz.detach().cpu().numpy() if isinstance(xyz, Tensor) else np.asarray(xyz)
    _require(values.dtype == np.dtype("float32") and values.shape == (record["facts"]["N0"], 3)
             and np.isfinite(values).all() and hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
             == record["xyz_sha256"], "Explicit XYZ/node order differs from hierarchy")
    F = features43.detach().cpu().numpy() if isinstance(features43, Tensor) else np.asarray(features43)
    n0, n1, n2 = (record["facts"][key] for key in ("N0", "N1", "N2"))
    with np.load(directory / record["file"], allow_pickle=False) as stored:
        _require(str(stored["cad_id"].item()) == cad_id and str(stored["xyz_sha256"].item()) == record["xyz_sha256"]
                 and str(stored["contract_sha256"].item()) == record["contract_sha256"], "Cache scalar identity changed")
        embedded = json.loads(str(stored["record_json"].item()))
        _require(embedded["source"] == record["source"] and embedded["labels_read"] == 0
                 and embedded["target_independent"], "Cache source record changed")
        roles = stored["role_bits0"]
        _require(F.dtype == np.dtype("float32") and F.shape == (n0, 43) and np.isfinite(F).all()
                 and (((F[:, :7] == 0) | (F[:, :7] == 1)).all())
                 and np.array_equal((F[:, :7].astype(np.uint8) * (2 ** np.arange(7, dtype=np.uint8))).sum(1).astype(np.uint8), roles),
                 "Original role inputs differ from the frozen hard hierarchy")
        R01 = _csr(stored, "R01", (n1, n0), stochastic=True)
        R12 = _csr(stored, "R12", (n2, n1), stochastic=True)
        P12 = _csr(stored, "P12", (n1, n2), stochastic=True)
        P01 = _csr(stored, "P01", (n0, n1), stochastic=True)
        W1, W2 = _csr(stored, "W1", (n1, n1)), _csr(stored, "W2", (n2, n2))
        descriptor = build_descriptor47(F, R01, stored["xyz1_normalized"], stored["mass1_mm3"])
    metadata = {"cad_id": cad_id, "node_count": n0, "fine_aggregates": n1, "coarse_aggregates": n2,
        "cache_file": record["file_identity"], "xyz_sha256": record["xyz_sha256"],
        "input_contract_sha256": record["contract_sha256"], "descriptor47_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
        "descriptor_policy": "hardR01(masked original43),centroid3,mass1/mean(mass1)",
        "descriptor_shape": list(descriptor.shape), "P01_nnz": P01.nnz, "P12_nnz": P12.nnz,
        "R12_nnz": R12.nnz, "W1_nnz": W1.nnz, "W2_nnz": W2.nnz,
        "labels_read": 0, "target_independent": True, "new_scales_fitted": False,
        "actual_preparation_cache_bytes": manifest["cache_bytes"],
        "maps": "fixed complete float32 CSR; original cut weights row-normalized, isolated identity"}
    return TopologyInput(cad_id, torch.as_tensor(descriptor, dtype=torch.float32, device=device),
        _torch_csr(R12, device), _torch_csr(P12, device), _torch_csr(P01, device),
        _torch_csr(_neighbor_operator(W1), device), _torch_csr(_neighbor_operator(W2), device), n0, metadata)


@dataclass(frozen=True)
class TopologyMessageFieldConfig:
    message_mode: Literal["self_only", "neighbors"] = "self_only"
    mode: Literal["linear_basis"] = "linear_basis"
    geometry_mode: Literal["shape"] = "shape"
    condition_dim: int = 43
    descriptor_dim: int = 32
    topology_dim: int = 32
    hidden_dim: int = 128
    hidden_layers: int = 2
    cloud_points: int = 1024
    seed: int = 2026
    bc_role_mode: Literal["actual"] = "actual"
    hash_bandwidth_mode: Literal["coarse8"] = "coarse8"
    distance_mode: Literal["euclidean"] = "euclidean"

    def __post_init__(self):
        _require(self.message_mode in ("self_only", "neighbors") and self.mode == "linear_basis"
                 and self.geometry_mode == "shape" and self.condition_dim == 43 and self.descriptor_dim == self.topology_dim == 32
                 and self.hidden_dim == 128 and self.hidden_layers == 2 and self.cloud_points == 1024 and self.seed == 2026
                 and self.bc_role_mode == "actual" and self.hash_bandwidth_mode == "coarse8" and self.distance_mode == "euclidean",
                 "Only the fixed predeclared topology-message layout and modes are supported")

    def decoder_config(self):
        return TCNNFieldConfig(out_channels=9, input_mode="both", condition_dim=107,
                               hidden_dim=128, hidden_layers=2, seed=self.seed)

    def record(self):
        return {**asdict(self), "method": "GeoBC two-level mesh-message TRAIN4 fitting pilot",
            "plan_sha256": PLAN_SHA, "claim": "conditional neighbor-message utility; not generalization or a new algorithm",
            "input_layout": "force-masked original25/local13/Euclidean5=43, topology32, global32=107",
            "basis_shape": "[N,3 reference loads,3 displacement components]", "output_channels": 9,
            "reference_load_order": list(REFERENCE_LOAD_ORDER), "component_order": list(COMPONENT_ORDER),
            "load_feature_policy": "zero original force/moment slots7:13 before descriptor restriction and decoder",
            "physical_decode": "external frozen common zero-centered u_scale_mm; no scale fitting here",
            "topology": {"input": 47, "width": 32, "linears": [[47,32],*[ [64,32] for _ in range(4)]],
                "linear_order": ["embed", "coarse0", "coarse1", "fine0", "fine1"], "parameters": 9856,
                "normalization": "six nonaffine LayerNorm32 eps1e-5; zero parameters",
                "updates": "Z+0.5*tanh(LN(Linear(concat(Z,A@Z))))",
                "coarse_fine_fusion": "0.5*(Z1_0+P12@Z2_2)", "query": "all-node tanh(LN(P01@Y1_2)) then explicit node-ID gather",
                "pair_difference": "A1/A2 identity or real cut-weight row-normalized adjacency only",
                "shared_spatial_mixing": "self_only still uses common graph hierarchy and R/P; not a no-topology control",
                "differentiable_context_policy": "recompute once per unique active CAD and optimizer step, never detach tokens or reuse after update"},
            "continuous_scope": "original-node IDs only; arbitrary-xyz feature/interpolation adapter undefined",
            "constructor_order": "global PointNet, embed, coarse0,coarse1,fine0,fine1, then split decoder; caller resets seed2026",
            "old_query_projection_retained": False, "geometry_target_independent": True, "geometry_load_independent": True,
            "stored_parameter_estimate": 7797152, "native_parameter_offsets_inferred": False,
            "native_layout": "masked Hash24 + Identity107 + thirteen ones -> public Network144/9; native padded output16"}


@contextmanager
def _timer(result, key, enabled, device):
    if not enabled:
        yield
        return
    if device.type == "cuda":
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        yield
        end.record(); end.synchronize()
        result[key] = begin.elapsed_time(end) / 1000
        result["method"] = "synchronized CUDA events only when requested"
    else:
        begin = time.perf_counter()
        yield
        result[key] = time.perf_counter() - begin
        result["method"] = "CPU perf_counter"


class TopologyMessageEncoder(nn.Module):
    def __init__(self, message_mode="self_only"):
        super().__init__()
        _require(message_mode in ("self_only", "neighbors"), "Unknown topology message mode")
        self.message_mode = message_mode
        self.embed = nn.Linear(47, 32)
        self.coarse0 = nn.Linear(64, 32)
        self.coarse1 = nn.Linear(64, 32)
        self.fine0 = nn.Linear(64, 32)
        self.fine1 = nn.Linear(64, 32)
        self.embed_norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-5)
        self.coarse0_norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-5)
        self.coarse1_norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-5)
        self.fine0_norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-5)
        self.fine1_norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-5)
        self.query_norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-5)

    def _update(self, z, A, linear, norm):
        mixed = z if self.message_mode == "self_only" else torch.sparse.mm(A, z)
        return z + .5 * torch.tanh(norm(linear(torch.cat((z, mixed), -1))))

    def forward(self, item: TopologyInput, *, record_timing=False):
        timings = {}
        with _timer(timings, "embedding_seconds", record_timing, item.device):
            fine = torch.tanh(self.embed_norm(self.embed(item.descriptor47)))
        with _timer(timings, "restriction_and_coarse_messages_seconds", record_timing, item.device):
            coarse = torch.sparse.mm(item.R12, fine)
            coarse = self._update(coarse, item.A2_neighbors, self.coarse0, self.coarse0_norm)
            coarse = self._update(coarse, item.A2_neighbors, self.coarse1, self.coarse1_norm)
        with _timer(timings, "fusion_and_fine_messages_seconds", record_timing, item.device):
            fine = .5 * (fine + torch.sparse.mm(item.P12, coarse))
            fine = self._update(fine, item.A1_neighbors, self.fine0, self.fine0_norm)
            fine = self._update(fine, item.A1_neighbors, self.fine1, self.fine1_norm)
        with _timer(timings, "fullnode_prolongation_and_normalization_seconds", record_timing, item.device):
            query = torch.tanh(self.query_norm(torch.sparse.mm(item.P01, fine)))
        return query, fine, timings


@dataclass(frozen=True)
class TopologyContextBundle:
    global_tensor: Tensor
    node_contexts: tuple[Tensor, ...]
    fine_tokens: tuple[Tensor, ...]
    cad_keys: tuple[str, ...]
    timings: dict

    @property
    def shape(self): return self.global_tensor.shape
    @property
    def ndim(self): return self.global_tensor.ndim
    @property
    def device(self): return self.global_tensor.device
    @property
    def dtype(self): return self.global_tensor.dtype
    @property
    def requires_grad(self): return self.global_tensor.requires_grad
    @property
    def grad(self): return self.global_tensor.grad
    def __len__(self): return len(self.global_tensor)
    def __getitem__(self, index):
        if isinstance(index, str): return getattr(self, index)
        _require(isinstance(index, (int, slice)), "Bundle supports explicit integer/slice CAD rows")
        selected = list(range(len(self)))[index]
        selected = [selected] if isinstance(selected, int) else selected
        globals_ = self.global_tensor[index]
        if globals_.ndim == 1: globals_ = globals_.unsqueeze(0)
        return TopologyContextBundle(globals_, tuple(self.node_contexts[i] for i in selected),
            tuple(self.fine_tokens[i] for i in selected), tuple(self.cad_keys[i] for i in selected), self.timings)
    def retain_grad(self):
        for tensor in (self.global_tensor, *self.node_contexts, *self.fine_tokens):
            if tensor.requires_grad: tensor.retain_grad()


class TopologyMessageField(nn.Module):
    def __init__(self, config=None, device="cuda", decoder=None):
        super().__init__()
        self.config = config or TopologyMessageFieldConfig()
        _require(isinstance(self.config, TopologyMessageFieldConfig), "TopologyMessageFieldConfig required")
        # This exact module/RNG order is common to both message modes.
        self.encoder = PointNetBCRoleEncoder(32)
        self.topology_encoder = TopologyMessageEncoder(self.config.message_mode)
        expected = self.config.decoder_config()
        self.decoder = decoder if decoder is not None else SplitHashBandwidthDecoder(expected, "coarse8", device=device)
        _require(isinstance(self.decoder, SplitHashBandwidthDecoder) and self.decoder.config == expected
                 and self.decoder.mode == "coarse8", "Frozen split coarse8 decoder/config required")
        self.backend = self.decoder.backend
        self.to(device)

    def native_parameter_groups(self): return self.decoder.native_parameter_groups()

    def topology_parameter_groups(self):
        return {name: list(getattr(self.topology_encoder, name).parameters())
                for name in ("embed", "coarse0", "coarse1", "fine0", "fine1")}

    def validate_native_layout(self):
        record = self.decoder.validate_native_layout()
        count = sum(p.numel() for p in self.parameters())
        _require(count == EXPECTED_FULL_PARAMETERS and sum(p.numel() for p in self.topology_encoder.parameters()) == 9856,
                 "Actual topology/native model parameter count differs")
        return {**record, "full_model_parameters": count, "topology_parameters": 9856,
                "old_query_projection_parameters": 0}

    @property
    def source_config(self):
        return {**self.config.record(), "global_encoder": _encoder_record(), "decoder_implementation": self.decoder.source_config,
            "actual_decoder_backend": self.backend, "encoder_parameters": sum(p.numel() for p in self.encoder.parameters()),
            "topology_encoder_parameters": sum(p.numel() for p in self.topology_encoder.parameters()),
            "native_encoding_parameters": sum(p.numel() for p in self.decoder.encoding.parameters()),
            "native_network_parameters": sum(p.numel() for p in self.decoder.network.parameters()),
            "full_model_parameters": sum(p.numel() for p in self.parameters()),
            "native_parameter_buffers": ["decoder.encoding.params", "decoder.network.params"],
            "input_feasibility": {"elapsed_seconds": 13.101552986999991, "four_CAD_cache_bytes": 342364814,
                "N1_N2": [2048,128], "status": "measured input preparation only; model/native gradients await formal first step"}}

    def encode_context(self, clouds5, masks, topology_inputs, *, record_timing=False):
        _require(clouds5.ndim == 3 and clouds5.shape[1:] == (self.config.cloud_points, 5)
                 and len(topology_inputs) == len(clouds5) and len(set(item.cad_id for item in topology_inputs)) == len(topology_inputs),
                 "Explicit unique ordered CAD inputs and fixed actual BC clouds required")
        timings = {"per_CAD": {}}
        with _timer(timings, "global_encoder_seconds", record_timing, clouds5.device):
            global_tensor = self.encoder(clouds5.detach().float(), masks)
        nodes, tokens, keys = [], [], []
        for item in topology_inputs:
            _require(item.device == global_tensor.device and not item.descriptor47.requires_grad,
                     "Static input maps and model must share device; input descriptor cannot be trainable")
            query, fine, timing = self.topology_encoder(item, record_timing=record_timing)
            nodes.append(query); tokens.append(fine); keys.append(item.cad_id)
            timings["per_CAD"][item.cad_id] = timing
        return TopologyContextBundle(global_tensor, tuple(nodes), tuple(tokens), tuple(keys), timings)

    def condition_features(self, features):
        _require(features.ndim == 2 and features.shape[1] == 43, "Query features must contain exactly original43")
        masked = features.detach().float().clone()
        masked[:, 7:13] = 0
        return masked

    def response_basis(self, xyz, features, *, context, node_indices, cad_index=None):
        _require(isinstance(context, TopologyContextBundle) and xyz.ndim == 2 and xyz.shape[1] == 3
                 and len(xyz) > 0 and features.shape == (len(xyz),43) and xyz.device == context.device,
                 "Explicit bundle, original-node xyz and query43 required")
        _require(node_indices.dtype == torch.long and node_indices.shape == (len(xyz),) and node_indices.device == xyz.device,
                 "Explicit original node_indices long[N] required")
        if cad_index is None:
            _require(len(context) == 1, "Multiple CAD contexts require explicit cad_index")
            cad_index = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
        _require(cad_index.dtype == torch.long and cad_index.shape == (len(xyz),) and cad_index.device == xyz.device
                 and not bool((cad_index < 0).any()) and not bool((cad_index >= len(context)).any()), "Invalid explicit CAD rows")
        positions, queries = [], []
        for row in range(len(context)):
            ids = torch.nonzero(cad_index == row, as_tuple=False).flatten()
            if not len(ids): continue
            selected = node_indices.index_select(0, ids)
            _require(not bool((selected < 0).any()) and not bool((selected >= len(context.node_contexts[row])).any()), "Node ID outside original CAD")
            positions.append(ids); queries.append(context.node_contexts[row].index_select(0, selected))
        query32 = xyz.new_empty((len(xyz),32), dtype=torch.float32).index_copy(0, torch.cat(positions), torch.cat(queries))
        global32 = context.global_tensor.index_select(0, cad_index)
        conditions = torch.cat((self.condition_features(features), query32, global32), -1)
        flat = self.decoder(xyz, conditions)
        _require(flat.shape == (len(xyz),9), "Native decoder must emit basis9")
        return flat.reshape(len(xyz),3,3)

    def forward(self, xyz, features, alpha, *, context, node_indices, cad_index=None):
        return combine_basis(self.response_basis(xyz,features,context=context,node_indices=node_indices,cad_index=cad_index),alpha)


__all__ = ["TopologyMessageFieldConfig", "TopologyMessageField", "TopologyMessageEncoder",
           "TopologyInput", "TopologyContextBundle", "load_topology_manifest", "load_topology_input", "build_descriptor47"]
