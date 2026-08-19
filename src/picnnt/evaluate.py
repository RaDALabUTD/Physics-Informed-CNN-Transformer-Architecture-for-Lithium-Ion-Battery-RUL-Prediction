from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats


@dataclasses.dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float
    mape: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    nonzero = np.abs(y_true) > 1e-6
    mape = float(100.0 * np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero]))) if nonzero.any() else float("nan")
    return Metrics(rmse=rmse, mae=mae, r2=r2, mape=mape)


@dataclasses.dataclass
class SignificanceResult:
    t_stat: float
    p_value: float
    cohens_d: float


def paired_significance(errors_a: np.ndarray, errors_b: np.ndarray) -> SignificanceResult:
    t_stat, p_value = stats.ttest_ind(errors_a, errors_b, equal_var=False)
    pooled_std = np.sqrt((errors_a.var(ddof=1) + errors_b.var(ddof=1)) / 2)
    cohens_d = float((errors_b.mean() - errors_a.mean()) / (pooled_std + 1e-12))
    return SignificanceResult(t_stat=float(t_stat), p_value=float(p_value), cohens_d=cohens_d)
