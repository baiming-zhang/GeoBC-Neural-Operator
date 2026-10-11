"""Runtime factory for a separately released native-TS shared3 fitting test."""
from .transolver_native_hash_v68 import NativeTransolverHash, FAMILY
from .macro_training import encode_case, query_case


def build_solver(family, device='cuda', seed=2026):
    if family != FAMILY:
        raise ValueError('Only frozen native originalTS composition')
    return NativeTransolverHash(seed=seed, device=device)


def profile(model):
    return dict(family=model.macro_family,
        parameter_count=sum(p.numel() for p in model.parameters()),
        architecture=model.source_config, global_geometry_context_dim=32,
        whole_case_operator_latent_channels=16, local_BC_material_channels=25,
        requires_R64_members=False, actual_CPU_fixture_only=False,
        capacity_matched_to_originalTS_R64_v65=True,
        capacity_matched_to_FNO_Hash=False,
        precision=model.source_config['precision'],
        independent_offnode_FEM_accuracy_measured=False)
