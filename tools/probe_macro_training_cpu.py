"""One explicit CPU fixture check of shared macro training, with no labels/CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    if not parser.parse_args().probe:
        print('Source-only; --probe checks CPU synthetic contracts, not real architectures.')
        return
    root = Path(__file__).resolve().parents[1]
    output = root/'notes/geobc_macro_training_CPU_probe.json'
    if output.exists():
        raise RuntimeError('Existing CPU attempt must not be repeated/overwritten')
    entry = (time.time(), time.monotonic())
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('CPU120s budget')))
    signal.alarm(120)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    record = dict(status='running', CPU_contract_passed=False, budget_seconds=120,
                  synthetic_forward_operators=True, actual_dataset_input_GT_PT_array_reads=0,
                  actual_CUDA_forwards=0, actual_optimizer_updates=0, official_GPU_qualified=False,
                  actual_training_released=False, families={}, failure=None)
    try:
        from geobc import available_cad_pilot_data as d
        from geobc.macro_training import FAMILIES, balanced_schedule, build_solver, encode_case, query_case, full_case_backward, profile
        manifest = d.read('data/geobc_prospective_dataset_v1/manifest.json')
        original = d.read('notes/geobc_exposure_extension_plan.json')['scientific']
        keys = manifest['train_keys']['40']
        order = balanced_schedule(keys, 200)
        assert d.digest(order) == original['full_order_digest']
        assert d.digest(order[:4800]) == original['parent_order_digest']
        assert set(Counter(order).values()) == {200}
        expanded = balanced_schedule(manifest['train_keys']['100'], 200)
        assert len(expanded) == 60000 and set(Counter(expanded).values()) == {200}
        record['actual_metadata_reads'] = 2
        record['TRAIN40_schedule_equals_existing_frozen_24000'] = True
        record['TRAIN100_schedule_cases'] = 300
        record['TRAIN100_schedule_updates_at200passes'] = 60000
        import torch
        from torch import nn
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(2026)

        class HashFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(60, 64), nn.Tanh(), nn.Linear(64, 4))
            def forward(self, xyz, condition):
                return self.layers(torch.cat((xyz, condition), -1))
        class NativeFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(61, 16)
            def forward(self, tokens):
                return self.linear(tokens)+self.linear(tokens.mean(1, keepdim=True))
        class GridFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(61, 16)
            def forward(self, grid):
                value = self.linear(grid.mean((2, 3, 4)))
                return value[:, :, None, None, None].expand(1, 16, 64, 64, 64)

        encoder_states = []
        for family in FAMILIES:
            model = build_solver(family, device='cpu', injected_hash_decoder=HashFixture() if family == 'HASH_PN' else None)
            record['families'][family] = {'construction_profile': profile(model)}
            encoder_states.append({k: v.clone() for k, v in model.encoder.state_dict().items()})
            if family == 'FNO9':
                model.operator = GridFixture()
            elif family == 'PP256_NATIVE':
                model.operator = NativeFixture()
            case = dict(key='synthetic/hor', scientific_group='TRAIN', xyz=torch.rand(23, 3),
                        features=torch.randn(23, 25), cloud=torch.rand(1, 1024, 3),
                        mask=torch.ones(1, 1024, dtype=torch.bool), target=torch.randn(23, 4))
            if family == 'FNO9':
                case.update(grid=torch.rand(1, 29, 64, 64, 64),
                            query_indices=torch.randint(64**3, (23, 8)), query_weights=torch.full((23, 8), 1/8))
            context, latent = encode_case(model, case)
            prediction = query_case(model, case, 0, 23, context, latent)
            loss = (prediction-case['target']).square().mean()
            loss.backward()
            gradients = {k: v.grad.clone() for k, v in model.named_parameters()}
            checks = []
            for chunk in (7, 23):
                model.zero_grad(set_to_none=True)
                measured, leafs = full_case_backward(model, case, ['synthetic/hor'], chunk_size=chunk)
                assert abs(measured-float(loss.detach())) < 2e-6
                maximum = 0.
                for k, v in model.named_parameters():
                    assert v.grad is not None and torch.isfinite(v.grad).all()
                    torch.testing.assert_close(v.grad, gradients[k], rtol=5e-5, atol=3e-6)
                    maximum = max(maximum, float((v.grad-gradients[k]).abs().max()))
                checks.append(dict(chunk_size=chunk, max_parameter_gradient_absolute_difference=maximum, leafs=leafs))
            rejected = 0
            for forbidden in (dict(case, scientific_group='DEV'), dict(case, scientific_group='FINAL'),
                              dict(case, key='synthetic/new_outside_TRAIN_scope')):
                guard_calls = []
                try:
                    full_case_backward(model, forbidden, ['synthetic/hor'], guard=lambda: guard_calls.append(True))
                except PermissionError:
                    assert not guard_calls
                    rejected += 1
            assert rejected == 3
            record['families'][family].update(full_vs_chunk_gradient_checks=checks, protected_case_rejections_before_encoding=rejected)
            del model, case, context, latent, prediction, gradients
        assert all(encoder_states[0].keys() == x.keys() and all(torch.equal(encoder_states[0][k], x[k])
                   for k in encoder_states[0]) for x in encoder_states[1:])
        record.update(status='completed', CPU_contract_passed=True, actual_initial_geometry_encoder_equal_all_families=True,
                      CUDA_initialized=torch.cuda.is_initialized())
    except Exception as error:
        record.update(status='failed', failure=type(error).__name__+': '+str(error))
        raise
    finally:
        record.update(epoch_seconds=time.time()-entry[0], monotonic_seconds=time.monotonic()-entry[1])
        record['budget_passed'] = max(record['epoch_seconds'], record['monotonic_seconds']) <= 120
        source_names = ['geobc/macro_training.py', 'tools/probe_macro_training_cpu.py',
                        'geobc/native_context_solver.py', 'geobc/common_grid_solver.py',
                        'geobc/conditioned_field.py', 'geobc/tcnn_fields.py']
        record['source_sha256'] = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_names}
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        signal.alarm(0)
    print(json.dumps({k: record[k] for k in ['status', 'CPU_contract_passed', 'epoch_seconds', 'CUDA_initialized']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
