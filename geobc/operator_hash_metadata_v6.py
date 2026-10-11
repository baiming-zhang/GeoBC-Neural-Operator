"""Metadata-only repair; original model math and failed sources are preserved."""
from .operator_conditioned_hash import OperatorConditionedHashSolver as Original
from .operator_hash_training import (balanced_schedule, encode_case, query_case,
                                     full_case_backward)


class MetadataFixedOperatorHash(Original):
    @property
    def source_config(self):
        return dict(method='Whole-case FNO-conditioned continuous Hash field',
                    claim='Candidate composition; no novel-algorithm or final-solver claim',
                    global_input='Original target-free R64 occupancy, xyz, native25 and CAD descriptor',
                    operator=self.operator.config.record(),
                    decoder=self.decoder_config.record(), counts=self.counts(),
                    backend=self.backend, sample_identity=False,
                    reused_FNO_initial_encoder_operator=True,
                    condition_order=['native25', 'global_geometry32', 'whole_case_operator_latent16'],
                    continuity='Fixed context; trilinear latent and Hash interpolation, explicit native25 extension',
                    precision='CPU injected fixture' if self.backend=='injected_cpu_interface_decoder' else
                              'FP32 encoder/operator with native TCNN FP16 decoder; mixed precision differs from MLP',
                    offnode_FEM_accuracy_measured=False)


def build_solver(family, device='cuda', seed=2026):
    if family != 'FNO_GLOBAL_HASH':
        from .operator_hash_training import build_solver as original
        return original(family, device=device, seed=seed)
    model = MetadataFixedOperatorHash(seed=seed, device=device)
    model.macro_family = family
    return model


def profile(model):
    from .operator_hash_training import profile as original
    result = original(model)
    if getattr(model, 'backend', None)=='injected_cpu_interface_decoder':
        result['actual_CPU_fixture_only'] = True
        result['precision'] = 'CPU injected fixture; native TCNN unqualified'
    return result
