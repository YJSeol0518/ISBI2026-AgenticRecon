"""External full-dose reference bank (per-organ cloud of patch descriptors) and divergences.

The bank of an organ is the pooled set of 50-D patch descriptors from full-dose scans of a
reference cohort (other patients).

* ``pearson_divergence_1d``   12-bin Pearson chi-square between two 1-D samples
* ``pearson_divergence_*``    the same per descriptor, and its log1p-mean
* ``sliced_wasserstein``      distributional distance in a whitened PCA space (used only by the
                              global-beta background rule, see ``search.global_sweep``)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .descriptors import N_DESC, descriptor_names


@dataclass
class OrganBank:
    organ: str
    feats: np.ndarray                  # (n_patches, N_DESC) raw descriptors
    center: np.ndarray                 # (N_DESC,) robust median
    scale: np.ndarray                  # (N_DESC,) robust IQR-based scale
    comps: np.ndarray                  # (N_DESC, k) PCA basis in scaled space
    evals: np.ndarray                  # (k,)
    n_subjects: int = 0
    hu_window: Tuple[float, float] = (0.0, 1.0)
    meta: Dict = None

    def whiten(self, F: np.ndarray) -> np.ndarray:
        Z = (F - self.center) / self.scale
        return (Z @ self.comps) / np.sqrt(self.evals + 1e-6)

    @property
    def ref_white(self) -> np.ndarray:
        if not hasattr(self, "_rw"):
            self._rw = self.whiten(self.feats)
        return self._rw


def _robust_scale(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.median(F, 0)
    q75, q25 = np.percentile(F, [75, 25], axis=0)
    scale = (q75 - q25) / 1.349
    scale[scale < 1e-6] = np.maximum(F.std(0), 1e-6)[scale < 1e-6]
    return center.astype(np.float64), scale.astype(np.float64)


def fit_bank(organ: str, feats: np.ndarray, n_subjects: int = 0, var_keep: float = 0.97,
             max_k: int = 16, hu_window: Tuple[float, float] = (0.0, 1.0),
             meta: Optional[Dict] = None) -> OrganBank:
    F = np.asarray(feats, dtype=np.float64)
    center, scale = _robust_scale(F)
    Z = np.clip((F - center) / scale, -8, 8)
    C = np.cov(Z, rowvar=False)
    C += 1e-3 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])   # shrinkage
    w, V = np.linalg.eigh(C)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    k = int(np.searchsorted(np.cumsum(w) / w.sum(), var_keep) + 1)
    k = int(np.clip(k, 4, min(max_k, len(w))))
    return OrganBank(organ=organ, feats=F.astype(np.float32), center=center, scale=scale,
                     comps=V[:, :k], evals=w[:k], n_subjects=n_subjects, hu_window=hu_window,
                     meta=meta or {})


# --------------------------------------------------------------------------
# divergences
# --------------------------------------------------------------------------
def sliced_wasserstein(A: np.ndarray, B: np.ndarray, n_proj: int = 96, seed: int = 0,
                       n_quant: int = 64) -> float:
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(A.shape[1], n_proj))
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    qs = (np.arange(n_quant) + 0.5) / n_quant
    return float(np.abs(np.quantile(A @ P, qs, axis=0) - np.quantile(B @ P, qs, axis=0)).mean())


def pearson_divergence_1d(a: np.ndarray, b: np.ndarray, n_bins: int = 12) -> float:
    r"""Binned Pearson chi-square  sum_i (p_i - q_i)^2 / (q_i + 1e-6).

    ``a`` is the test sample, ``b`` the reference; both are histogrammed over their common
    range and normalised to probability mass functions.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=edges)
    q, _ = np.histogram(b, bins=edges)
    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    return float(np.sum((p - q) ** 2 / (q + 1e-6)))


def pearson_divergence_per_feature(bank: OrganBank, F: np.ndarray,
                                   names: Optional[List[str]] = None) -> Dict[str, float]:
    """Pearson chi-square of every descriptor's marginal vs the bank (all 50 by default)."""
    names = names or descriptor_names()
    F = np.asarray(F, np.float64)
    if len(F) == 0:
        return {n: float("nan") for n in names}
    return {n: pearson_divergence_1d(F[:, i], bank.feats[:, i].astype(np.float64))
            for i, n in enumerate(names)}


def pearson_divergence_aggregate(bank: OrganBank, F: np.ndarray) -> float:
    """mean_k log(1 + chi2_k) over the 50 descriptors (Ours 1 objective, Eq. 2)."""
    d = pearson_divergence_per_feature(bank, F)
    vals = np.array([v for v in d.values() if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(np.log1p(vals))) if len(vals) else float("nan")


def divergence(bank: OrganBank, F: np.ndarray, seed: int = 0) -> Dict[str, float]:
    if len(F) == 0:
        return {"sw": float("nan"), "pearson": float("nan"), "n_patches": 0}
    W = bank.whiten(np.asarray(F, np.float64))
    return {"sw": sliced_wasserstein(bank.ref_white, W, seed=seed),
            "pearson": pearson_divergence_aggregate(bank, F),
            "n_patches": int(len(F))}


# --------------------------------------------------------------------------
# persistence: <name>.npz (arrays) + <name>.json (cohort info and organ HU priors)
# --------------------------------------------------------------------------
def save_banks(banks: Dict[str, OrganBank], path, hu_priors: Optional[Dict] = None,
               extra: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob, meta = {}, {}
    for k, b in banks.items():
        for a in ("feats", "center", "scale", "comps", "evals"):
            blob[f"{k}/{a}"] = getattr(b, a)
        meta[k] = {"n_subjects": b.n_subjects, "hu_window": list(b.hu_window),
                   "n_patches": int(len(b.feats)), "k": int(b.comps.shape[1]), "meta": b.meta}
    np.savez_compressed(path, __meta__=json.dumps(meta), **blob)
    info = {"hu_priors": hu_priors or {}, "descriptor_names": descriptor_names(), "n_desc": N_DESC}
    info.update(extra or {})
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))


def load_banks(path) -> Tuple[Dict[str, OrganBank], Dict[str, Dict[str, float]]]:
    """Returns ``(banks, hu_priors)``."""
    z = np.load(str(path), allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    banks = {k: OrganBank(organ=k, feats=z[f"{k}/feats"], center=z[f"{k}/center"],
                          scale=z[f"{k}/scale"], comps=z[f"{k}/comps"], evals=z[f"{k}/evals"],
                          n_subjects=m["n_subjects"], hu_window=tuple(m["hu_window"]),
                          meta=m.get("meta", {}))
             for k, m in meta.items()}
    priors = json.loads(Path(str(path)).with_suffix(".json").read_text())["hu_priors"]
    return banks, priors
