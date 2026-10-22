"""The 20 Haralick features of a normalised GLCM (Oh et al., 2024, Sec. 3.3 / Appendix).

Grey levels are indexed 1..N. Note that ``difference_average`` and ``dissimilarity`` are the
same quantity (IBSI): sum_k k * p_{x-y}(k) = sum_ij |i-j| P(i,j).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

HARALICK_FEATURE_NAMES: List[str] = [
    "autocorrelation",
    "cluster_prominence",
    "cluster_shade",
    "contrast",
    "correlation",
    "difference_average",
    "difference_entropy",
    "difference_variance",
    "dissimilarity",
    "energy",
    "entropy",
    "homogeneity",
    "information_measure_correlation_1",
    "inverse_difference_moment",
    "max_correlation_coefficient",
    "maximum_probability",
    "sum_average",
    "sum_entropy",
    "variance",
    "sum_variance",
]

DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),    # horizontal
    (1, 0),    # vertical
    (1, 1),    # main diagonal
    (-1, 1),   # anti-diagonal
)


def _safe(arr: np.ndarray, eps: float) -> np.ndarray:
    return np.where(arr > 0, arr, eps)


def _safe_log(arr: np.ndarray, eps: float) -> np.ndarray:
    s = _safe(arr, eps)
    return np.where(arr > 0, np.log(s), np.log(eps))


def _marginals(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return P.sum(axis=1), P.sum(axis=0)


def _diff_distribution(P: np.ndarray, n_levels: int) -> np.ndarray:
    levels = np.arange(n_levels, dtype=np.int64)
    diff = np.abs(levels[:, None] - levels[None, :])
    d = diff.ravel()
    return np.bincount(d, weights=P.ravel(), minlength=n_levels)


def _sum_distribution(P: np.ndarray, n_levels: int) -> np.ndarray:
    levels = np.arange(n_levels, dtype=np.int64)
    s = (levels[:, None] + levels[None, :]).ravel()
    return np.bincount(s, weights=P.ravel(), minlength=2 * n_levels - 1)

def compute_features(P: np.ndarray, eps: float = 1e-12,
                     feature_order=HARALICK_FEATURE_NAMES) -> np.ndarray:
    """Compute the 20 Haralick features from a (N, N) probability matrix."""
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square; got shape {P.shape}")
    n_levels = P.shape[0]

    levels_1 = np.arange(1, n_levels + 1, dtype=np.float64)
    I, J = np.meshgrid(levels_1, levels_1, indexing='ij')
    IJ_product = I * J
    I_plus_J = I + J
    I_minus_J = I - J
    I_minus_J_abs = np.abs(I_minus_J)

    gx, gy = _marginals(P)
    mu_x = float(np.sum(levels_1 * gx))
    mu_y = float(np.sum(levels_1 * gy))
    sigma_x2 = float(np.sum((levels_1 - mu_x) ** 2 * gx))
    sigma_y2 = float(np.sum((levels_1 - mu_y) ** 2 * gy))
    sigma_x = float(np.sqrt(max(sigma_x2, 0.0)))
    sigma_y = float(np.sqrt(max(sigma_y2, 0.0)))
    mu_xy = 0.5 * (mu_x + mu_y)

    HX = -float(np.sum(gx * _safe_log(gx, eps)))
    HY = -float(np.sum(gy * _safe_log(gy, eps)))
    HXY = -float(np.sum(P * _safe_log(P, eps)))

    diff_dist = _diff_distribution(P, n_levels)
    sum_dist = _sum_distribution(P, n_levels)
    mu_xy_diff = float(np.sum(np.arange(diff_dist.size, dtype=np.float64) * diff_dist))
    mu_xy_sum = float(
        np.sum((np.arange(sum_dist.size, dtype=np.float64) + 2.0) * sum_dist)
    )

    # Max correlation coefficient
    active = gx > eps
    if active.sum() < 2:
        max_corr_coeff = 0.0
    else:
        G_act = P[np.ix_(active, active)]
        gx_act = gx[active]
        M = G_act @ G_act.T
        denom = gx_act[:, None] * gx_act[None, :]
        denom = np.where(denom > eps, denom, eps)
        Q = M / denom
        eigs = np.linalg.eigvalsh(Q)
        if eigs.size < 2:
            max_corr_coeff = 0.0
        else:
            eigs_sorted = np.sort(eigs)[::-1]
            lam2 = eigs_sorted[1]
            max_corr_coeff = float(min(np.sqrt(lam2), 1.0)) if lam2 > 0 else 0.0

    # Information measure of correlation 1
    gx_2d = gx[:, None]
    gy_2d = gy[None, :]
    gx_gy = gx_2d * gy_2d
    HXY1 = -float(np.sum(P * _safe_log(gx_gy, eps)))
    denom_imc1 = max(HX, HY)
    imc1 = (HXY - HXY1) / denom_imc1 if denom_imc1 >= eps else 0.0

    diff_norm = np.arange(diff_dist.size, dtype=np.float64)
    sum_norm = np.arange(sum_dist.size, dtype=np.float64) + 2.0

    # NOTE: relative to the usual definitions (cluster shade = 3rd moment, cluster prominence =
    # 4th moment of i+j-mu_x-mu_y) the two names below are swapped. Kept as-is so that the
    # reported feature columns are reproduced exactly.
    values = {
        "autocorrelation": float(np.sum(IJ_product * P)),
        "cluster_prominence": float(np.sum((I_plus_J - 2 * mu_xy) ** 3 * P)),
        "cluster_shade": float(np.sum((I_plus_J - 2 * mu_xy) ** 4 * P)),
        "contrast": float(np.sum((diff_norm ** 2) * diff_dist)),
        "correlation": (
            float(np.sum(((I - mu_x) / sigma_x) * ((J - mu_y) / sigma_y) * P))
            if sigma_x >= eps and sigma_y >= eps else 0.0
        ),
        "difference_average": float(np.sum(diff_norm * diff_dist)),
        "difference_entropy": float(-np.sum(diff_dist * _safe_log(diff_dist, eps))),
        "difference_variance": float(np.sum((diff_norm - mu_xy_diff) ** 2 * diff_dist)),
        "dissimilarity": float(np.sum(I_minus_J_abs * P)),
        "energy": float(np.sum(P * P)),
        "entropy": float(-np.sum(P * _safe_log(P, eps))),
        "homogeneity": float(np.sum(P / (1.0 + I_minus_J ** 2))),
        "information_measure_correlation_1": imc1,
        "inverse_difference_moment": float(np.sum(P / (1.0 + I_minus_J_abs))),
        "max_correlation_coefficient": max_corr_coeff,
        "maximum_probability": float(P.max()),
        "sum_average": float(np.sum(sum_norm * sum_dist)),
        "sum_entropy": float(-np.sum(sum_dist * _safe_log(sum_dist, eps))),
        "variance": float(np.sum((I - mu_xy) ** 2 * P)),
        "sum_variance": float(np.sum((sum_norm - mu_xy_sum) ** 2 * sum_dist)),
    }
    return np.array([values[name] for name in feature_order], dtype=np.float64)
