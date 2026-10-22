"""GT-free per-organ audit of a reconstruction: texture, projection consistency and HU stability.

    L = w_tex * sum_o pi_o D_tex(o) + w_prj * sum_o pi_o G_prj(o) + w_hu * sum_o pi_o D_hu(o)

* ``D_tex``  divergence of the organ's patch descriptors from the external reference bank
             (``tex_metric='pearson'``: Ours 1 objective; ``'sw'``: sliced Wasserstein)
* ``G_prj``  log(chi2_o / chi2_o,floor) of the noise-weighted projection residual restricted to the
             organ's ray footprint, relative to an (almost) unregularised fit of the same data
* ``D_hu``   drift of the organ's median HU from the LDCT plus distance to the cohort HU prior

Only the measured quarter-dose sinogram, its FDK image and the external cohort are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .hu import mu_to_hu, roi_window
from .masks import OrganPartition
from .projector import fp
from .texture.bank import OrganBank, divergence
from .texture.descriptors import DEFAULT_SPEC, describe_region


# ---------------------------------------------------------------------------
# projection-domain consistency
# ---------------------------------------------------------------------------
def estimate_noise_var(sino: np.ndarray, min_var: float = 1e-10) -> np.ndarray:
    """Per-detector-channel noise variance from adjacent-view differences (robust MAD)."""
    d = np.diff(np.asarray(sino, np.float64), axis=0)
    med = np.median(d, axis=0, keepdims=True)
    mad = np.median(np.abs(d - med), axis=0)
    var = (1.4826 * mad) ** 2 / 2.0
    var = np.maximum(var, np.maximum(min_var, 1e-3 * np.median(var)))
    return var.astype(np.float32)


@dataclass
class ConsistencyModel:
    param: Dict[str, Any]
    sino: np.ndarray
    noise_var: np.ndarray                                              # (nu,)
    footprints: Dict[str, np.ndarray] = field(default_factory=dict)   # organ -> (nview, nu) weights
    floor: Dict[str, float] = field(default_factory=dict)             # achievable chi2 per organ

    @classmethod
    def build(cls, sino: np.ndarray, param: Dict[str, Any], masks: Dict[str, np.ndarray],
              min_weight_pct: float = 60.0) -> "ConsistencyModel":
        fps = {}
        for name, m in masks.items():
            if m is None or not m.any():
                continue
            w = fp(m.astype(np.float32), param)
            thr = np.percentile(w[w > 0], min_weight_pct) if (w > 0).any() else 0.0
            w = np.where(w >= thr, w, 0.0)          # rays that genuinely traverse the organ
            if w.sum() > 0:
                fps[name] = (w / w.sum()).astype(np.float32)
        return cls(param=param, sino=np.asarray(sino, np.float32),
                   noise_var=estimate_noise_var(sino), footprints=fps)

    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        r = fp(np.asarray(x, np.float32), self.param) - self.sino
        r2n = (r.astype(np.float64) ** 2) / self.noise_var[None, :]
        out = {"chi2_global": float(r2n.mean())}
        for name, w in self.footprints.items():
            out[f"chi2_{name}"] = float((w * r2n).sum())
        return out

    def set_floor(self, x_floor: np.ndarray) -> None:
        c = self.evaluate(x_floor)
        self.floor = {k[5:]: max(v, 1e-9) for k, v in c.items() if k.startswith("chi2_")}
        self.floor["global"] = max(c["chi2_global"], 1e-9)


def discrepancy(chi2: float, floor: Optional[float] = None) -> float:
    """Signed log(chi2 / floor)."""
    if not np.isfinite(chi2) or chi2 <= 0:
        return float("inf")
    if floor is None or not np.isfinite(floor) or floor <= 0:
        return abs(float(np.log(chi2)))
    return float(np.log(chi2 / floor))


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
#: organ weights pi_o of the audit loss (texture-critical classes dominate)
DEFAULT_ORGAN_WEIGHTS = {"liver": 1.0, "lung": 1.0, "bone": 1.0, "soft": 0.35, "fat": 0.2, "organ": 0.35}


@dataclass
class AuditWeights:
    tex: float = 1.0
    prj: float = 1.0
    hu: float = 0.5
    tex_metric: str = "sw"           # 'sw' or 'pearson'


@dataclass
class OrganReport:
    organ: str
    n_patches: int = 0
    d_tex: float = float("nan")
    g_prj: float = float("nan")
    d_hu: float = float("nan")


@dataclass
class AuditResult:
    organs: Dict[str, OrganReport]
    loss: float
    loss_tex: float
    loss_prj: float
    loss_hu: float


class OrganAuditor:
    """Per-organ HU windows and the HU anchor are fixed once from the LDCT (FDK) image."""

    def __init__(self, part: OrganPartition, banks: Dict[str, OrganBank], cons: ConsistencyModel,
                 hu_priors: Dict[str, Dict[str, float]], organs: Sequence[str],
                 x_ldct: np.ndarray, weights: Optional[AuditWeights] = None,
                 organ_weights: Optional[Dict[str, float]] = None, mu_water: float = 0.019,
                 max_patches: int = 260, seed: int = 0):
        self.banks, self.cons, self.hu_priors = banks, cons, hu_priors
        self.organs = list(organs)
        self.w = weights or AuditWeights()
        self.mu_water = mu_water
        self.max_patches = max_patches
        self.seed = seed
        self.masks = {o: part.mask(o) for o in self.organs}
        ow = organ_weights or DEFAULT_ORGAN_WEIGHTS
        tot = sum(ow.values())
        self.pi = {o: ow.get(o, 0.0) / tot for o in self.organs}
        hu0 = mu_to_hu(x_ldct, mu_water)
        self.windows = {o: roi_window(hu0, self.masks[o]) for o in self.organs}
        self.baseline_hu = {o: float(np.median(hu0[self.masks[o]])) for o in self.organs}

    def audit(self, x: np.ndarray) -> AuditResult:
        hu = mu_to_hu(x, self.mu_water)
        cons = self.cons.evaluate(x)
        rng = np.random.default_rng(self.seed)      # shared across organs, in self.organs order
        reports: Dict[str, OrganReport] = {}
        for o in self.organs:
            m = self.masks[o]
            spec = DEFAULT_SPEC.get(o, DEFAULT_SPEC["soft"])
            F = describe_region(hu, m, spec, self.windows[o], self.max_patches, rng)
            rep = OrganReport(organ=o, n_patches=len(F))
            bank = self.banks.get(o)
            if bank is not None and len(F):
                rep.d_tex = divergence(bank, F, seed=self.seed)[self.w.tex_metric]

            v = hu[m]
            med = float(np.median(v))
            iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
            d_prior = d_drift = float("nan")
            pr = self.hu_priors.get(o)
            if pr:
                d_prior = abs(med - pr["median"]) / max(pr.get("iqr", 1.0), 5.0)
            d_drift = abs(med - self.baseline_hu[o]) / max(iqr, 5.0)
            num = den = 0.0
            for val, w in ((d_drift, 0.8), (d_prior, 0.2)):
                if np.isfinite(val):
                    num += w * val
                    den += w
            rep.d_hu = float(num / den) if den > 0 else float("nan")

            chi2 = cons.get(f"chi2_{o}", float("nan"))
            rep.g_prj = discrepancy(chi2, self.cons.floor.get(o)) if np.isfinite(chi2) else float("nan")
            reports[o] = rep

        def agg(attr: str) -> float:
            num = tot = 0.0
            for o, r in reports.items():
                v = getattr(r, attr)
                if np.isfinite(v):
                    num += self.pi[o] * v
                    tot += self.pi[o]
            return num / tot if tot > 0 else float("nan")

        l_tex, l_prj, l_hu = agg("d_tex"), agg("g_prj"), agg("d_hu")
        loss = (self.w.tex * (l_tex if np.isfinite(l_tex) else 10.0)
                + self.w.prj * (l_prj if np.isfinite(l_prj) else 10.0)
                + self.w.hu * (l_hu if np.isfinite(l_hu) else 10.0))
        return AuditResult(organs=reports, loss=float(loss), loss_tex=float(l_tex),
                           loss_prj=float(l_prj), loss_hu=float(l_hu))
