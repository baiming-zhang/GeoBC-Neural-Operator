"""Input-derived dimensional amplitude prior; no targets or scaler fitting.

This is a conditioning hypothesis, not an exact FEM similarity law. Original
native25 features, signed physical labels and all model parameters are kept.
"""
import torch
from .coarse_transolver_v19 import CoarseTransolver
from .coarse_fno_control_v22 import CoarseFNOControl


FORMULA = {
    'U': '(norm(F)/(E*L)+norm(M)/(E*L**2))/(42300/(113800*200)+565000/(113800*200**2))',
    'S': '(norm(F)/L**2+norm(M)/L**3)/(42300/200**2+565000/200**3)',
    'units': 'F N, M Nmm, E MPa=N/mm2, L mm; U numerator mm, S numerator MPa',
    'meaning': 'positive input-only amplitude prior, not exact structural response or new labels',
}


def response_gains(features):
    if features.ndim != 2 or features.shape[1] != 25 or not len(features):
        raise ValueError('Complete original native25 required')
    # These are exactly the globally broadcast source inputs, not interface
    # flags, outputs, nodal labels or a fitted statistic.
    global_values = torch.cat((features[:, 7:14], features[:, 24:25]), dim=1)
    if not torch.isfinite(global_values).all() or not torch.equal(
            global_values, global_values[:1].expand_as(global_values)):
        raise ValueError('Finite case-constant force/moment/E/L required')
    x = global_values[0].to(torch.float64)
    force = torch.linalg.vector_norm(x[:3]*42300.)
    moment = torch.linalg.vector_norm(x[3:6]*565000.)
    elastic_modulus, length = x[6]*113800., x[7]*200.
    if elastic_modulus <= 0 or length <= 0 or force+moment <= 0:
        raise ValueError('Positive E/L and nonzero load required; no epsilon rescue')
    u = (force/(elastic_modulus*length)+moment/(elastic_modulus*length.square())) / (
        42300./(113800.*200.)+565000./(113800.*200.**2))
    s = (force/length.square()+moment/length.pow(3)) / (42300./200.**2+565000./200.**3)
    gains = torch.stack((u, s))
    if not torch.isfinite(gains).all() or not (gains > 0).all():
        raise FloatingPointError('Undefined/nonfinite response gains; no clipping')
    return gains


def prepare_response_case(case):
    features = case['features']
    stamp = (features.data_ptr(), features._version, tuple(features.shape), str(features.device))
    if case.get('_response_gain_v30_stamp') != stamp:
        case['_response_gain_v30'] = response_gains(features)
        case['_response_gain_v30_stamp'] = stamp
    return case['_response_gain_v30']


def scale_encoded_response(encoded, gains, scale):
    """Old encoded -> signed physical -> multiply positive prior -> old encoded.

    FP64 only for the scalar transform, like the existing physical objective;
    backbone/parameters stay FP32. Original means/asinh are not refitted.
    """
    if encoded.ndim != 2 or encoded.shape[1] != 4 or gains.shape != (2,):
        raise ValueError('Four output channels and two positive gains required')
    value = encoded.to(torch.float64)
    mean, std = value.new_tensor(scale.mean), value.new_tensor(scale.std)
    transformed = value*std+mean
    u = (transformed[:, :3]*gains[0]-mean[:3])/std[:3]
    stress = transformed[:, 3:4]
    if scale.stress_scale is None:
        stress = stress*gains[1]
    else:
        # scale cancels between sinh decode and asinh encode. No abs/clamp.
        stress = torch.asinh(stress.sinh()*gains[1])
    result = torch.cat((u, (stress-mean[3])/std[3]), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite signed response; no clipping/rescue')
    return result


class _ResponseGain:
    def encode_case(self, case):
        prepare_response_case(case)
        return super().encode_case(case)

    def query(self, case, start, stop, context, latent):
        encoded = super().query(case, start, stop, context, latent)
        return scale_encoded_response(encoded, case['_response_gain_v30'], self.response_scale)

    def profile(self):
        return dict(super().profile(), response_gain=FORMULA,
                    initial_weights_matched_not_initial_function=True,
                    scalar_response_transform='FP64; old scaler unchanged; output FP64 before persistence FP32',
                    native_inputs_unchanged=True, labels_unchanged=True)


class ResponseGainTransolver(_ResponseGain, CoarseTransolver):
    def __init__(self, scale, seed=2026, device='cuda'):
        super().__init__('transolver', seed, device)
        self.response_scale = scale


class ResponseGainFNO(_ResponseGain, CoarseFNOControl):
    def __init__(self, scale, seed=2026, device='cuda'):
        super().__init__(seed, device)
        self.response_scale = scale
