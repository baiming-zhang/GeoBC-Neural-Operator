"""Original signed U/S output encoding and decoding."""

from dataclasses import dataclass
import numpy as np

@dataclass
class FieldScaler:
    mean: list[float]
    std: list[float]
    stress_scale: float | None
    channels: list[int]

    @classmethod
    def fit(cls, targets, channels, stress_transform='linear'):
        physical = np.concatenate([x[:, channels].astype(np.float64) for x in targets])
        scale = None
        if stress_transform == 'asinh' and 3 in channels:
            j = channels.index(3)
            scale = max(float(np.quantile(np.abs(physical[:, j]), 0.9)), 1e-12)
            physical[:, j] = np.arcsinh(physical[:, j] / scale)
        return cls(physical.mean(0).tolist(), np.maximum(physical.std(0), 1e-08).tolist(), scale, channels)

    def encode(self, target):
        result = np.asarray(target, dtype=np.float64)[:, self.channels].copy()
        if self.stress_scale is not None:
            j = self.channels.index(3)
            result[:, j] = np.arcsinh(result[:, j] / self.stress_scale)
        return ((result - np.asarray(self.mean)) / np.asarray(self.std)).astype(np.float32)

    def decode(self, prediction):
        result = np.asarray(prediction, dtype=np.float64) * np.asarray(self.std) + np.asarray(self.mean)
        if self.stress_scale is not None:
            j = self.channels.index(3)
            result[:, j] = self.stress_scale * np.sinh(result[:, j])
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite physical field after decoding')
        return result
