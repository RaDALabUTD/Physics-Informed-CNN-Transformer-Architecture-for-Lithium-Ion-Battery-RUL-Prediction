from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression


@dataclasses.dataclass
class PhysicsConstants:
    r0: float
    r_ref: float
    eol_fraction: float = 0.80

    @staticmethod
    def fit(resistance_proxy: torch.Tensor, eol_fraction: float = 0.80) -> "PhysicsConstants":
        r0 = float(torch.quantile(resistance_proxy, 0.02))
        r_max = float(torch.quantile(resistance_proxy, 0.98))
        r_ref = max((r_max - r0) / (-math.log(eol_fraction)), 1e-6)
        return PhysicsConstants(r0=r0, r_ref=r_ref, eol_fraction=eol_fraction)


class CapacityRULCurve:
    def __init__(self, x_thresholds: torch.Tensor, y_thresholds: torch.Tensor):
        self.x = x_thresholds
        self.y = y_thresholds

    @staticmethod
    def fit(rul_true: np.ndarray, capacity_frac_true: np.ndarray) -> "CapacityRULCurve":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(rul_true, capacity_frac_true)
        x = torch.tensor(iso.X_thresholds_, dtype=torch.float32)
        y = torch.tensor(iso.y_thresholds_, dtype=torch.float32)
        return CapacityRULCurve(x, y)

    def to(self, device: torch.device) -> "CapacityRULCurve":
        return CapacityRULCurve(self.x.to(device), self.y.to(device))

    def __call__(self, rul_hat: torch.Tensor) -> torch.Tensor:
        x, y = self.x, self.y
        rul_clamped = rul_hat.clamp(x[0], x[-1])
        idx = torch.searchsorted(x, rul_clamped.detach()).clamp(1, len(x) - 1)
        x0, x1 = x[idx - 1], x[idx]
        y0, y1 = y[idx - 1], y[idx]
        frac = (rul_clamped - x0) / (x1 - x0 + 1e-6)
        return y0 + frac * (y1 - y0)


def monotonicity_loss(y_hat: torch.Tensor, segment_len: int, margin: float = 1.0) -> torch.Tensor:
    n = y_hat.shape[0]
    if n < segment_len or n % segment_len != 0:
        return y_hat.new_zeros(())
    y = y_hat.view(-1, segment_len)
    if segment_len < 2:
        return y_hat.new_zeros(())
    diffs = y[:, 1:] - y[:, :-1] + margin
    return torch.mean(torch.relu(diffs) ** 2)


def butler_volmer_loss(
    y_hat: torch.Tensor,
    resistance_proxy: torch.Tensor,
    constants: PhysicsConstants,
    capacity_rul_curve: CapacityRULCurve,
) -> torch.Tensor:
    rul_hat = torch.relu(y_hat)
    q_hat = capacity_rul_curve(rul_hat)
    q_bv_max = torch.exp(-(resistance_proxy - constants.r0) / constants.r_ref).clamp(max=1.0)
    return torch.mean(torch.relu(q_hat - q_bv_max) ** 2)


@dataclasses.dataclass
class LossWeights:
    lambda_mono: float = 0.1
    lambda_bv: float = 0.05
    adaptive_scale: bool = False
    max_scale_ratio: float = 20.0


def _adaptive_lambda(target_frac: float, l_data: torch.Tensor, l_term: torch.Tensor, max_ratio: float) -> torch.Tensor:
    if l_term.item() <= 1e-12:
        return l_term.new_zeros(())
    target_weighted = target_frac * l_data.detach()
    eff_lambda = target_weighted / (l_term.detach() + 1e-12)
    return eff_lambda.clamp(max=max_ratio * 500.0)


def total_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    resistance_proxy: torch.Tensor,
    constants: PhysicsConstants,
    weights: LossWeights,
    segment_len: int,
    capacity_rul_curve: CapacityRULCurve,
    mono_margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_data = torch.mean((y_hat - y_true) ** 2)
    l_mono = monotonicity_loss(y_hat, segment_len, mono_margin) if weights.lambda_mono > 0 else y_hat.new_zeros(())
    l_bv = (
        butler_volmer_loss(y_hat, resistance_proxy, constants, capacity_rul_curve)
        if weights.lambda_bv > 0 else y_hat.new_zeros(())
    )
    if weights.adaptive_scale:
        eff_lambda_mono = _adaptive_lambda(weights.lambda_mono, l_data, l_mono, weights.max_scale_ratio)
        eff_lambda_bv = _adaptive_lambda(weights.lambda_bv, l_data, l_bv, weights.max_scale_ratio)
    else:
        eff_lambda_mono = weights.lambda_mono
        eff_lambda_bv = weights.lambda_bv
    total = l_data + eff_lambda_mono * l_mono + eff_lambda_bv * l_bv
    return total, {
        "data": l_data.item(), "mono": l_mono.item(), "bv": l_bv.item(), "total": total.item(),
        "eff_lambda_mono": float(eff_lambda_mono) if torch.is_tensor(eff_lambda_mono) else eff_lambda_mono,
        "eff_lambda_bv": float(eff_lambda_bv) if torch.is_tensor(eff_lambda_bv) else eff_lambda_bv,
    }
