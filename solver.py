"""SART with a spatially varying quadratic regulariser.

    x_0 = FDK(y)
    D   = A^T A 1 + eps
    x_{k+1} = x_k - ( A^T(A x_k - y) / D + B * (x_k - Q(x_k)) ) / (1 + B)

``Q`` is the neighbourhood average in ``projector.quadpenalty`` and ``B`` is the
regularisation-strength field beta(r). A scalar ``B`` gives the ordinary global-beta solver.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .projector import bp, filtering, fp, quadpenalty

EPS = 1e-6


def fdk(sino: np.ndarray, param: Dict[str, Any], filter_type: str = "ram-lak") -> np.ndarray:
    """Filtered back-projection (also the SART initialisation)."""
    return bp(filtering(sino, param, filter_type=filter_type), param)


def build_D(param: Dict[str, Any]) -> np.ndarray:
    ones = np.ones((param["ny"], param["nx"]), dtype=np.float32)
    return bp(fp(ones, param), param) + EPS


def reconstruct(sino: np.ndarray, param: Dict[str, Any], beta_field,
                n_iter: int = 30, D: Optional[np.ndarray] = None,
                x0: Optional[np.ndarray] = None) -> np.ndarray:
    sino = np.asarray(sino, dtype=np.float32)
    x = fdk(sino, param) if x0 is None else np.array(x0, np.float32)
    D = build_D(param) if D is None else D
    B = np.asarray(beta_field, dtype=np.float32)
    if B.ndim == 0:
        B = np.full_like(x, float(B))
    denom = (1.0 + B).astype(np.float32)
    for _ in range(n_iter):
        grad = bp(fp(x, param) - sino, param)
        x_new = x - (grad / D + B * (x - quadpenalty(x, 1))) / denom
        x = x_new.astype(np.float32)
    return x
