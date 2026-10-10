"""Supplied-array, all-native-node curved TETRA10 tensor recovery.

This module performs no imports of scientific libraries, I/O, solver calls,
target access, clock resets or model operations. The caller supplies the pinned
geometry module, Torch/NumPy, immutable physical arrays and budget guard.
"""
from __future__ import annotations

import math

SCIENTIFIC_DIGEST = "0cdfd4ae71d2de1ce26d6bb6d5fe517d1920d7cd66d49eb7001b4ad49c8fa5b8"
GEOMETRY_SOURCE = "geobc/continuous_query_geometry.py"
GEOMETRY_SOURCE_SHA256 = "95599c0001c2f50f1cabd5f2e74d91396f71072378a77b31763ee086e6a79c27"
CELL_CHUNK = 8192
GRADIENT_IDENTITY_TOLERANCE = 1e-9
TENSOR_SYMMETRY_TOLERANCE = 1e-12
TENSOR_CHANNELS = ("xx", "yy", "zz", "xy", "xz", "yz")
REFERENCE_BARYCENTRIC = (
    (1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.),
    (.5, .5, 0., 0.), (0., .5, .5, 0.), (.5, 0., .5, 0.),
    (.5, 0., 0., .5), (0., .5, 0., .5), (0., 0., .5, .5),
)


class ScientificNegative(RuntimeError):
    """Whole representation is invalid; callers must not drop cells/nodes."""


def require(condition, message):
    if not bool(condition):
        raise ScientificNegative(message)


def source_config():
    """Static source contract; reading it does not touch arrays or libraries."""
    return {
        "scientific_digest": SCIENTIFIC_DIGEST,
        "geometry_source": GEOMETRY_SOURCE,
        "geometry_source_sha256": GEOMETRY_SOURCE_SHA256,
        "API": "recover_tensor(torch,np,geometry,vertices_mm,cells10,nodal_U,lame_MPa,quadrature_weights,expected_valid,guard,chunk=8192)",
        "cell_chunk": CELL_CHUNK,
        "reference_barycentric": [list(row) for row in REFERENCE_BARYCENTRIC],
        "VTK_mid_edge_order": [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
        "lame_order": ["lambda_MPa", "mu_MPa"],
        "tensor_channels": list(TENSOR_CHANNELS),
        "shear_convention": "Tensor shear, no engineering factor2 in offdiagonal sigma slots",
        "physical_formula": "J[c,r]=sum_a X[a,c]*dN[a]/dr[r]; solve J^T grad_X N=grad_r N; G[i,j]=sum_a U[a,i]*grad_X[j]N[a]",
        "constitutive_formula": "epsilon=(G+G^T)/2; sigma=2*mu*epsilon+lambda*trace(epsilon)*I",
        "incident_weight": "Positive accepted fourQ volume sum /10, same single weight for each native node of its incident cell",
        "coverage": "mass>0 must exactly equal supplied immutable old valid mask",
        "geometry_guard": "All10 per-cell node J finite, determinant>0, solve_ex info0, finite physical gradients/tensors; no drop/fallback",
        "gradient_identity_relative_tolerance": GRADIENT_IDENTITY_TOLERANCE,
        "tensor_symmetry_relative_tolerance": TENSOR_SYMMETRY_TOLERANCE,
        "array_types": "CUDA FP64 X/U/lame/fourQ weights; CUDA int64 cells10; CUDA bool expectedvalid",
        "caller_budget": "guard() at each chunk boundary; caller owns all setup/compute/I/O/clock and resource accounting",
        "scientific_negative_exception": "geobc.elastic_nodal_tensor.ScientificNegative",
        "no_file_target_solver_model_access": True,
        "claim": "Chosen all-node curved-P2 constitutive/recovery proxy, not source H3D/GPSTRESS recovery or signed scalar definition",
    }


def _validate_inputs(torch, vertices_mm, cells10, nodal_U, lame_MPa,
                     quadrature_weights, expected_valid, guard, chunk):
    require(callable(guard), "Caller budget guard must be callable")
    require(type(chunk) is int and chunk == CELL_CHUNK, "Frozen8192-cell chunk changed")
    require(vertices_mm.ndim == 2 and vertices_mm.shape[1] == 3 and len(vertices_mm) > 0,
            "Physical vertices must be nonempty[N,3]")
    require(cells10.ndim == 2 and cells10.shape[1] == 10 and len(cells10) > 0,
            "Accepted VTK tetra10 incidence must be nonempty[T,10]")
    n, t = len(vertices_mm), len(cells10)
    require(nodal_U.shape == (n, 3) and lame_MPa.shape == (2,)
            and quadrature_weights.shape == (t, 4) and expected_valid.shape == (n,),
            "Supplied physical field/material/volume/valid schemas differ")
    arrays = (vertices_mm, cells10, nodal_U, lame_MPa, quadrature_weights, expected_valid)
    require(vertices_mm.device.type == "cuda" and all(value.device == vertices_mm.device for value in arrays),
            "All supplied tensors must share one CUDA device")
    require(all(value.dtype == torch.float64 for value in (vertices_mm, nodal_U, lame_MPa, quadrature_weights))
            and cells10.dtype == torch.int64 and expected_valid.dtype == torch.bool,
            "Frozen FP64/int64/bool input dtype changed")
    require(torch.isfinite(vertices_mm).all() and torch.isfinite(nodal_U).all()
            and torch.isfinite(lame_MPa).all() and torch.isfinite(quadrature_weights).all(),
            "Nonfinite supplied physical geometry/U/material/weights")
    require((quadrature_weights > 0).all(), "Accepted physical fourQ weights must all be positive")
    require(cells10.min() >= 0 and cells10.max() < n, "VTK incidence outside original node order")
    lam, mu = lame_MPa[0], lame_MPa[1]
    require(mu > 0 and 3 * lam + 2 * mu > 0, "Nonpositive isotropic shear or bulk modulus")
    return n, t, lam, mu


def recover_tensor(torch, np, geometry, vertices_mm, cells10, nodal_U,
                   lame_MPa, quadrature_weights, expected_valid, guard, chunk=8192):
    """Return (sigma[N,6] FP64, valid[N] bool, small diagnostics).

    Ten native barycentric queries are evaluated for every supplied element.
    The original nodal U is a predicted physical solution, never a target. The
    accepted4Q weights are used solely for positive incident-volume weighting;
    old4Q gradients/tensors are not recomputed. A failed guard raises one whole
    ScientificNegative, and no partial sigma or changed validity is returned.
    """
    guard()
    n, t, lam, mu = _validate_inputs(torch, vertices_mm, cells10, nodal_U,
                                    lame_MPa, quadrature_weights, expected_valid, guard, chunk)
    reference = np.asarray(REFERENCE_BARYCENTRIC, dtype=np.float64)
    require(np.array_equal(np.asarray(geometry.REFERENCE_BARYCENTRIC), reference),
            "Pinned geometry VTK10 native reference order differs")
    # Only a10-point analytic reference table is evaluated on CPU; real fields
    # stay CUDA FP64. These identities are part of this one real recovery stage.
    derivative_cpu = geometry.shape_gradient(reference)
    shape_cpu = geometry.quadratic_shape(reference)
    require(derivative_cpu.shape == (10, 10, 3) and shape_cpu.shape == (10, 10)
            and np.isfinite(derivative_cpu).all() and np.isfinite(shape_cpu).all(),
            "Pinned shape/gradient native tables invalid")
    cardinal_error = float(np.max(np.abs(shape_cpu - np.eye(10))))
    shape_partition_error = float(np.max(np.abs(shape_cpu.sum(axis=1) - 1)))
    require(cardinal_error <= 1e-12 and shape_partition_error <= 1e-12,
            "Native VTK10 shape cardinal/partition identity failed")
    derivative = torch.as_tensor(derivative_cpu, dtype=torch.float64, device=vertices_mm.device)
    eye = torch.eye(3, dtype=torch.float64, device=vertices_mm.device)
    accumulated = torch.zeros((n, 6), dtype=torch.float64, device=vertices_mm.device)
    mass = torch.zeros(n, dtype=torch.float64, device=vertices_mm.device)
    partition_error = affine_error = symmetry_error = 0.0
    minimum_det = math.inf
    checked_cells = checked_queries = chunks = 0
    minimum_weight = math.inf
    with torch.no_grad():
        for start in range(0, t, chunk):
            guard(); stop = min(start + chunk, t); count = stop - start
            incidence = cells10[start:stop]
            X = vertices_mm[incidence]
            U = nodal_U[incidence]
            # Subtract one cell-constant translation: derivatives still use
            # physical mm, and exact derivative partition makes this equivalent
            # to sum_a X_a*dN_a and sum_a U_a*gradN_a without a new scale.
            X_relative = X - X[:, :1]
            U_relative = U - U[:, :1]
            J = torch.einsum("nac,qar->nqcr", X_relative, derivative)
            require(torch.isfinite(J).all(), "Nonfinite all10 native Jacobian at cellchunk" + str(start))
            determinant = torch.linalg.det(J)
            require(torch.isfinite(determinant).all() and (determinant > 0).all(),
                    "Nonpositive/nonfinite native-node Jacobian at cellchunk" + str(start))
            rhs = derivative.transpose(-1, -2)[None].expand(count, -1, -1, -1)
            solved, info = torch.linalg.solve_ex(J.transpose(-1, -2), rhs, check_errors=False)
            require((info == 0).all() and torch.isfinite(solved).all(),
                    "All10 native physical-gradient solve failed at cellchunk" + str(start))
            gradients = solved.transpose(-1, -2)  # [cell,query,localbasis,physicalcoordinate]
            gradient_norm = torch.linalg.vector_norm(gradients, dim=(-2, -1))
            require(torch.isfinite(gradient_norm).all() and (gradient_norm > 0).all(),
                    "Invalid physical-gradient identity scale")
            # Both checks are dimensionless. There is no floor/clipping of a
            # nonzero denominator, and no normalized-coordinate derivative.
            partition = torch.linalg.vector_norm(gradients.sum(dim=2), dim=-1) / gradient_norm
            affine = torch.einsum("nai,nqaj->nqij", X_relative, gradients)
            affine_relative = torch.linalg.matrix_norm(affine - eye, ord="fro") / math.sqrt(3.0)
            require(torch.isfinite(partition).all() and torch.isfinite(affine_relative).all(),
                    "Nonfinite gradient partition/physical-affine mapping identity")
            chunk_partition = float(partition.max().item())
            chunk_affine = float(affine_relative.max().item())
            require(chunk_partition <= GRADIENT_IDENTITY_TOLERANCE and chunk_affine <= GRADIENT_IDENTITY_TOLERANCE,
                    "Native-node gradient partition or affine mapping failed at cellchunk" + str(start))
            G = torch.einsum("nai,nqaj->nqij", U_relative, gradients)
            epsilon = 0.5 * (G + G.transpose(-1, -2))
            trace = epsilon.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            sigma = 2 * mu * epsilon + lam * trace[..., None, None] * eye
            require(torch.isfinite(G).all() and torch.isfinite(epsilon).all() and torch.isfinite(sigma).all(),
                    "Nonfinite native physical gradient/strain/tensor at cellchunk" + str(start))
            sigma_norm = torch.linalg.matrix_norm(sigma, ord="fro")
            asymmetry = torch.linalg.matrix_norm(sigma - sigma.transpose(-1, -2), ord="fro")
            require(torch.isfinite(sigma_norm).all() and torch.isfinite(asymmetry).all(),
                    "Nonfinite native tensor symmetry scale")
            require(((sigma_norm > 0) | (asymmetry == 0)).all(), "Undefined zero-tensor symmetry")
            # A truly zero tensor has exactly zero symmetry error; replace only
            # its0 divisor by1, never impose a floor on finite nonzero stress.
            symmetry = asymmetry / torch.where(sigma_norm > 0, sigma_norm, torch.ones_like(sigma_norm))
            chunk_symmetry = float(symmetry.max().item())
            require(math.isfinite(chunk_symmetry) and chunk_symmetry <= TENSOR_SYMMETRY_TOLERANCE,
                    "Native tensor symmetry failed at cellchunk" + str(start))
            six = torch.stack((sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2],
                               sigma[..., 0, 1], sigma[..., 0, 2], sigma[..., 1, 2]), dim=-1)
            volume = quadrature_weights[start:stop].sum(dim=1)
            weight = volume / 10.0
            require(torch.isfinite(weight).all() and (weight > 0).all(), "Incident Vc/10 is nonpositive/nonfinite")
            ids = incidence.reshape(-1)
            accumulated.index_add_(0, ids, (six * weight[:, None, None]).reshape(-1, 6))
            mass.index_add_(0, ids, weight[:, None].expand(-1, 10).reshape(-1))
            checked_cells += count; checked_queries += count * 10; chunks += 1
            minimum_det = min(minimum_det, float(determinant.min().item()))
            minimum_weight = min(minimum_weight, float(weight.min().item()))
            partition_error = max(partition_error, chunk_partition)
            affine_error = max(affine_error, chunk_affine)
            symmetry_error = max(symmetry_error, chunk_symmetry)
            del X, U, X_relative, U_relative, J, determinant, rhs, solved, info, gradients
            del gradient_norm, partition, affine, affine_relative, G, epsilon, trace, sigma
            del sigma_norm, asymmetry, symmetry, six, volume, weight, ids, incidence
            guard()
        require(checked_cells == t and checked_queries == t * 10,
                "Recovery did not evaluate every supplied element/native node")
        require(torch.isfinite(mass).all() and (mass >= 0).all() and torch.isfinite(accumulated).all(),
                "Incident volume/tensor accumulation is nonfinite or negative")
        valid = mass > 0
        require(torch.equal(valid, expected_valid),
                "All-native-node incidence coverage differs from immutable old validity")
        require((accumulated[~valid] == 0).all(), "Invalid nodes have unexpected accumulated tensors")
        result = torch.zeros_like(accumulated)
        result[valid] = accumulated[valid] / mass[valid, None]
        require(torch.isfinite(result).all(), "Aggregated all-node physical tensor is nonfinite")
        valid_count = int(valid.sum().item())
    guard()
    diagnostics = {
        "passed": True,
        "scientific_digest": SCIENTIFIC_DIGEST,
        "representation": "element_node",
        "all_cells": t,
        "native_queries_per_cell": 10,
        "all_native_element_node_queries": checked_queries,
        "checked_cells": checked_cells,
        "checked_chunks": chunks,
        "cell_chunk": chunk,
        "node_count": n,
        "valid_node_count": valid_count,
        "invalid_node_count": n - valid_count,
        "coverage_exactly_matches_old_valid": True,
        "all10_J_finite_and_strictly_positive": True,
        "all10_solve_ex_info_zero": True,
        "minimum_native_detJ_mm3": minimum_det,
        "minimum_positive_incident_weight_mm3": minimum_weight,
        "gradient_partition_relative_error": partition_error,
        "physical_affine_mapping_relative_error": affine_error,
        "tensor_symmetry_relative_error": symmetry_error,
        "shape_cardinality_absolute_error": cardinal_error,
        "shape_partition_absolute_error": shape_partition_error,
        "gradient_identity_relative_tolerance": GRADIENT_IDENTITY_TOLERANCE,
        "tensor_symmetry_relative_tolerance": TENSOR_SYMMETRY_TOLERANCE,
        "sigma_channels": list(TENSOR_CHANNELS),
        "sigma_units": "MPa",
        "X_U_units": "mm",
        "gradient_units": "dimensionless",
        "positive_weight": "sum accepted4Q physical weights /10; one weight per incident native cellnode",
        "invalid_nodes": "Zero tensor slots with explicit valid0; no failed reached node is hidden as invalid0",
        "skipped_cells_or_queries": 0,
        "fallbacks": 0,
        "GT_reads": 0,
        "solver_calls": 0,
        "K_guard_repeats": 0,
        "old_fourQ_tensor_recovery_runs": 0,
        "clock_or_budget_resets": 0,
        "cost_and_memory_measurement_owner": "Caller driver, including helper setup and checks",
        "claim": "Constitutive/recovery proxy, not original nodal stress GT or source signed-scalar exporter",
    }
    return result, valid, diagnostics


__all__ = ["ScientificNegative", "SCIENTIFIC_DIGEST", "GEOMETRY_SOURCE_SHA256", "source_config", "recover_tensor"]
