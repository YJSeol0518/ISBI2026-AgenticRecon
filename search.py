"""Organ-wise coordinate search over beta: Ours 1 (texture-guided BO) and Ours 2 (feature-resolved BO).

Common protocol (both methods)
------------------------------
* Searched classes: liver, lung, bone, soft (those present on the slice). The schedule visits
  liver/lung/bone twice per cycle and soft once; one class is probed per reconstruction while
  the others stay at their current incumbent.
* Background classes fat / organ are not searched. Their beta follows a tissue rule from a
  GT-free 9-point global-beta sweep (``global_sweep``): with g = argmin of the audit loss,
  beta_fat = 3 g, beta_soft = 1.5 g (not used while soft is searched), beta_organ = g.
* Every reconstruction is SART (30 iterations) with the organ-blended beta field.
* The incumbent of a class is the probed beta with the best *observed* objective. The final image
  is one more reconstruction with all incumbents.
* Only the low-dose sinogram, its FDK (LDCT) image and the external reference bank are used.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .case import Case
from .hu import mu_to_hu, roi_window
from .optim.generic_bo import ConstantOptimiser
from .optim.response_model import FeatureResponseModel
from .solver import reconstruct
from .texture.bank import pearson_divergence_1d, pearson_divergence_aggregate
from .texture.descriptors import DEFAULT_SPEC, HARALICK_FEATURE_NAMES, describe_region

PRIORITY = ("liver", "lung", "bone")
SEARCH = ("liver", "lung", "bone", "soft")
BG = ("fat", "organ")
SWEEP_BETAS = (0.003, 0.008, 0.015, 0.025, 0.04, 0.06, 0.1, 0.2, 0.5)
EPS = 1e-6


def make_cycle(classes: Sequence[str], priority: Sequence[str], n: int) -> List[str]:
    hi = [c for c in priority if c in classes]
    lo = [c for c in classes if c not in hi]
    seq = hi + hi + lo if hi else list(classes)
    return [seq[i % len(seq)] for i in range(n)]


# ---------------------------------------------------------------------------
# GT-free global-beta sweep: background rule (both methods) and Ours 2 seed
# ---------------------------------------------------------------------------
def seed_score(case: Case, x: np.ndarray, organs=PRIORITY, seed: int = 0, max_patches: int = 260) -> float:
    """Mean over liver/lung/bone of the 50-descriptor log1p-mean Pearson divergence to the bank."""
    hu = mu_to_hu(x, case.mu_water)
    hu0 = mu_to_hu(case.x_fdk, case.mu_water)
    scores = []
    for o in organs:
        if o not in case.organs or o not in case.banks:
            continue
        mask = case.part.mask(o)
        if mask.sum() < 400:
            continue
        F = describe_region(hu, mask, DEFAULT_SPEC.get(o, DEFAULT_SPEC["soft"]), roi_window(hu0, mask),
                            max_patches, np.random.default_rng(seed))
        if len(F):
            scores.append(pearson_divergence_aggregate(case.banks[o], F))
    return float(np.mean(scores)) if scores else float("nan")


def global_sweep(case: Case, betas: Sequence[float] = SWEEP_BETAS, n_iter: int = 30,
                 verbose: bool = True) -> Dict:
    """9 global-beta reconstructions (not counted in the method budgets).

    Returns ``g_best`` (argmin of the GT-free audit loss: sliced-Wasserstein texture + projection
    consistency + HU stability) and ``seed_beta`` (argmin of ``seed_score``).
    """
    aud = case.auditor(tex_metric="sw")
    rows = []
    for b in betas:
        x = case.reconstruct(float(np.float32(b)), n_iter=n_iter)
        a = aud.audit(x)
        rows.append({"beta": float(b), "audit_loss": a.loss, "seed_score": seed_score(case, x)})
        if verbose:
            print(f"  [sweep] beta={b:<6g} audit_loss={a.loss:.4f} seed_score={rows[-1]['seed_score']:.4f}",
                  flush=True)
    loss = np.array([r["audit_loss"] for r in rows], float)
    g_best = float(betas[int(np.nanargmin(loss))])
    seed_beta = min(rows, key=lambda r: r["seed_score"])["beta"]
    return {"rows": rows, "g_best": g_best, "seed_beta": seed_beta}


def background_betas(g_best: Optional[float]) -> Dict[str, float]:
    g = g_best or 0.03
    return {"fat": 3.0 * g, "soft": 1.5 * g, "organ": 1.0 * g}


class _Harness:
    """Shared bookkeeping: which classes are searched, the background rule, the field builder."""

    def __init__(self, case: Case, g_best: Optional[float], init_beta: float, n_iter: int):
        self.case = case
        self.search = [o for o in SEARCH if o in case.organs]
        self.bg = [o for o in BG if o in case.organs]
        self.g = g_best or 0.03
        self.tissue = background_betas(g_best)
        self.fb = case.beta_field(feather=3.0, beta_default=init_beta)
        self.n_iter = n_iter
        self.incumbent = {o: init_beta for o in self.search}
        self.n_recon = 0

    def betas(self, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        b = {c: self.incumbent[c] for c in self.search}
        b.update({c: self.tissue.get(c, self.g) for c in self.bg})
        b.update(override or {})
        return b

    def recon(self, betas: Dict[str, float]) -> np.ndarray:
        self.n_recon += 1
        return reconstruct(self.case.sino, self.case.param, self.fb(betas), n_iter=self.n_iter, D=self.case.D)

    def final(self) -> Tuple[Dict[str, float], np.ndarray]:
        b = self.betas()
        return b, self.recon(b)


# ---------------------------------------------------------------------------
# Ours 1: texture-guided BO
# ---------------------------------------------------------------------------
def run_ours1(case: Case, g_best: Optional[float], budget: int = 22, n_iter: int = 30,
              init_beta: float = 0.01, seed: int = 0, verbose: bool = True):
    """Per class an independent 1-D GP-EI optimiser on log(beta) in [1e-4, 1] (first probe 0.01,
    then 2 random), minimising D_o = mean_k log(1 + chi2_k) over the 50 descriptors of the class
    vs the reference bank. ``budget`` probes + 1 final reconstruction."""
    h = _Harness(case, g_best, init_beta, n_iter)
    aud = case.auditor(tex_metric="pearson", seed=seed)
    opt = {o: ConstantOptimiser(initial_constants={"beta": init_beta}) for o in h.search}
    best = {o: None for o in h.search}
    nxt = {o: init_beta for o in h.search}
    trace = []
    for step, o in enumerate(make_cycle(h.search, PRIORITY, budget), 1):
        beta_o = float(nxt[o])
        x = h.recon(h.betas({o: beta_o}))
        a = aud.audit(x)
        rep = a.organs.get(o)
        d = float(rep.d_tex) if rep is not None and np.isfinite(rep.d_tex) else float(a.loss_tex)
        improved = best[o] is None or d < best[o]
        if improved:
            best[o], h.incumbent[o] = d, beta_o
        trace.append({"step": step, "organ": o, "beta": beta_o, "D": d, "improved": improved,
                      "incumbent": dict(h.incumbent)})
        if verbose:
            print(f"    step {step:2d} [{o:6s}] beta={beta_o:.3g} D={d:.4f}{' *' if improved else '  '}"
                  f"  incumbent={h.incumbent[o]:.3g}", flush=True)
        nxt[o] = float(opt[o].propose_next(reward=-d, constants={"beta": beta_o})["beta"])
    betas, x_final = h.final()
    return betas, x_final, trace, h.n_recon


# ---------------------------------------------------------------------------
# Ours 2: feature-resolved BO
# ---------------------------------------------------------------------------
def haralick_pearson(case: Case, aud, organ: str, x: np.ndarray, seed: int = 0,
                     max_patches: int = 260) -> np.ndarray:
    """20 per-feature Pearson chi-square divergences of one organ vs the bank (Haralick order)."""
    F = describe_region(mu_to_hu(x, case.mu_water), aud.masks[organ],
                        DEFAULT_SPEC.get(organ, DEFAULT_SPEC["soft"]), aud.windows.get(organ),
                        max_patches, np.random.default_rng(seed))
    if len(F) == 0:
        return np.full(len(HARALICK_FEATURE_NAMES), np.nan)
    ref = case.banks[organ].feats
    return np.array([pearson_divergence_1d(F[:, i].astype(np.float64), ref[:, i].astype(np.float64))
                     for i in range(len(HARALICK_FEATURE_NAMES))])


def run_ours2(case: Case, g_best: Optional[float], seed_beta: float, budget: int = 18, n_iter: int = 30,
              beta_bounds=(2e-3, 1.5), n_grid: int = 9, init_beta: float = 0.01, lambda_worst: float = 0.25,
              kappa: float = 1.0, surprise_z: float = 2.0, min_surprised: int = 3, extra_pool: int = 4,
              seed: int = 0, verbose: bool = True):
    """Per class 20 per-feature GPs (``FeatureResponseModel``) over a 9-point log grid.

    Probe 1 of a class is ``seed_beta`` (from ``global_sweep``), probe 2 the farthest grid point,
    later probes argmin_u J(mu(u)) - kappa * mean_f sigma_f(u) over untried grid points.
    Self-correction: if >= ``min_surprised`` features deviate more than ``surprise_z`` predictive
    std from the model's prediction, the same class is probed again immediately (at most
    ``extra_pool`` times). ``budget`` + re-probes + 1 final reconstruction.
    """
    h = _Harness(case, g_best, init_beta, n_iter)
    aud = case.auditor(tex_metric="pearson", seed=seed)
    lo, hi = np.log(beta_bounds[0]), np.log(beta_bounds[1])
    grid = np.exp(np.linspace(lo, hi, n_grid))
    names = list(HARALICK_FEATURE_NAMES)
    ldct = {o: haralick_pearson(case, aud, o, case.x_fdk, seed=seed) for o in h.search}
    models = {o: FeatureResponseModel(bounds=(lo, hi), feature_names=names) for o in h.search}
    best_J = {o: None for o in h.search}
    seeded = set()
    order = deque(make_cycle(h.search, PRIORITY, budget))
    trace, step = [], 0
    while order:
        o = order.popleft()
        m = models[o]
        if o not in seeded:
            beta_o = float(np.clip(seed_beta, beta_bounds[0], beta_bounds[1]))
            seeded.add(o)
        else:
            probed = set(m.probed)
            cand = np.array([b for b in grid if np.log(b) not in probed]) if probed else grid
            if len(cand) == 0:
                cand = grid
            beta_o = float(cand[m.propose_next(np.log(cand), lambda_worst=lambda_worst, kappa=kappa)])
        u = np.log(beta_o)
        pred = m.prediction_at(u) if m.n_probes >= 2 else None

        x = h.recon(h.betas({o: beta_o}))
        r_vec = (haralick_pearson(case, aud, o, x, seed=seed) + EPS) / (ldct[o] + EPS)

        n_exceed, reprobe = None, False
        if pred is not None:
            z = np.abs(np.log(np.maximum(r_vec, 0.0) + EPS) - pred[0]) / np.maximum(pred[1], 1e-6)
            n_exceed = int((z > surprise_z).sum())
            reprobe = n_exceed >= min_surprised and extra_pool > 0

        m.add(u, r_vec)
        J = m.observed_objective(r_vec, lambda_worst)
        improved = best_J[o] is None or J < best_J[o]
        if improved:
            best_J[o], h.incumbent[o] = J, beta_o
        step += 1
        trace.append({"step": step, "organ": o, "beta": beta_o, "J": J, "improved": improved,
                      "incumbent": dict(h.incumbent), "r_vec": r_vec.tolist(),
                      "n_surprised_features": n_exceed, "reprobe": reprobe})
        if verbose:
            print(f"    step {step:2d} [{o:6s}] beta={beta_o:.3g} J={J:.4f}{' *' if improved else '  '}"
                  f"  incumbent={h.incumbent[o]:.3g}" + ("  [surprise -> re-probe]" if reprobe else ""),
                  flush=True)
        if reprobe:
            order.appendleft(o)
            extra_pool -= 1
    betas, x_final = h.final()
    return betas, x_final, trace, h.n_recon
