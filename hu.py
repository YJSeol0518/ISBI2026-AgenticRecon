"""Attenuation <-> HU conversion and ROI windowing."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

MU_WATER = 0.019          # mm^-1 at ~70 keV


def mu_to_hu(mu: np.ndarray, mu_water: float = MU_WATER, mu_air: float = 0.0) -> np.ndarray:
    return ((np.asarray(mu, np.float32) - mu_water) / (mu_water - mu_air) * 1000.0).astype(np.float32)


def roi_window(hu: np.ndarray, mask: Optional[np.ndarray] = None, lo_pct: float = 2.0,
               hi_pct: float = 98.0, min_width: float = 50.0) -> Tuple[float, float]:
    """ROI-adaptive HU window (2nd-98th percentile, at least ``min_width`` HU wide)."""
    v = hu[mask] if mask is not None and mask.any() else hu.ravel()
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if hi - lo < min_width:
        c = 0.5 * (lo + hi)
        lo, hi = c - min_width / 2, c + min_width / 2
    return float(lo), float(hi)


def quantize(hu: np.ndarray, lo: float, hi: float, levels: int = 256) -> np.ndarray:
    q = (np.clip(hu, lo, hi) - lo) / max(hi - lo, 1e-8)
    return np.clip((q * (levels - 1)).astype(np.int32), 0, levels - 1)
