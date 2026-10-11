"""Same fresh geometry encoder/TCNN decoder, official original TS backbone."""
import torch
from .operator_hash_metadata_v6 import MetadataFixedOperatorHash
from .transformers import build_transformer
from .fno_stress_specialist_v54 import tree_digest
from .macro_training import encode_case,query_case

FAMILY='TS_ORIGINAL_GLOBAL_HASH'


class TransolverConditionedHash(MetadataFixedOperatorHash):
    def __init__(self,seed=2026,device='cuda'):
        # Build the exact untrained FNO-conditioned reference; the actual
        # driver verifies its full bytes against the already observed v61.
        super().__init__(seed=seed,device=device)
        reference_digest=tree_digest(self.state_dict())
        shared={k:tree_digest(getattr(self,k).state_dict())for k in ('encoder','head')}
        cuda_devices=[torch.cuda.current_device()]if torch.device(device).type=='cuda'else[]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            operator=build_transformer('transolver',61,16,device=device,checkpoint_blocks=True)
        self.operator=operator
        self.arm=FAMILY
        self.macro_family=FAMILY
        assert {k:tree_digest(getattr(self,k).state_dict())for k in shared}==shared
        self.initial_replacement=dict(reference_FNO_Hash_model_digest=reference_digest,
            fresh_encoder_head_digests=shared,only_operator_replaced=True,shared_modules_unchanged=True,
            pretrained_weights_used=False,backbone_draws_preserve_CPU_CUDA_RNG=True)

    @property
    def source_config(self):
        return dict(method='Whole-case original Transolver-conditioned continuous Hash field',
            claim='Candidate composition, no final solver or novel algorithm claim',
            global_input='Same complete target-free R64 61 channels, xyz reordered first; all262144 tokens',
            operator={**self.operator.source_config,
                'input':'xyz3 + occupancy1 + nativeBC/material25 + geometry32, allR64 tokens',
                'output':'16 wholecase latent channels before the continuous decoder'},
            decoder=self.decoder_config.record(),counts=self.counts(),
            backend=self.backend,sample_identity=False,condition_order=['native25','global_geometry32','whole_case_operator_latent16'],
            initialization='Only originalTS backbone replaces fresh reference; encoder and TCNN head bytes unchanged',
            continuous_query_policy='Same fixed wholecase context, trilinear latent and Hash interpolation, explicit native25 extension',
            independent_offnode_FEM_accuracy_measured=False)


def build_solver(family,device='cuda',seed=2026):
    if family!=FAMILY:raise ValueError('Only this frozen originalTS continuous decoder candidate')
    return TransolverConditionedHash(seed=seed,device=device)


def profile(model):
    return dict(family=model.macro_family,parameter_count=sum(p.numel()for p in model.parameters()),
        architecture=model.source_config,global_geometry_context_dim=32,whole_case_operator_latent_channels=16,
        local_BC_material_channels=25,requires_R64_members=True,actual_CPU_fixture_only=False,
        capacity_matched_across_backbones=False,precision='FP32 originalTS/PointNet and native TCNN FP16 computation with FP32 masters',
        independent_offnode_FEM_accuracy_measured=False)
