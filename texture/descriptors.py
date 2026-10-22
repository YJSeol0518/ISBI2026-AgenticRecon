"""Multi-family texture descriptors for CT patches.

GLCM/Haralick alone is a fairly blunt instrument for the three tissue types we
care about, so the bank deliberately mixes four complementary families:

===================  ==================================================
family               what it is sensitive to
===================  ==================================================
Haralick (GLCM)      joint grey-level co-occurrence -- overall grain
NPS (radial)         *where in frequency* the fluctuation energy sits;
                     the CT-native way to say "this looks over-smoothed"
gradient / LoG       edge strength at several scales (bone)
LBP (riu2)           micro-structure ordering, contrast-invariant (lung)
first-order          sigma / skew / kurtosis -- noise magnitude and bias
===================  ==================================================

Every descriptor is computed on a *patch*, in HU, quantised with a window
derived from the patch's own organ ROI, so the bank is invariant to the
absolute-scale mismatch between clinical HU and reconstructed attenuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy import ndimage as ndi

from .glcm import compute_features, HARALICK_FEATURE_NAMES, DIRECTIONS
from ..hu import quantize, roi_window

_DY = np.array([d[0] for d in DIRECTIONS], dtype=np.int64)
_DX = np.array([d[1] for d in DIRECTIONS], dtype=np.int64)


@njit(cache=True)
def _glcm_core(image, mask, dy_arr, dx_arr, distance, symmetric, n_levels):
    """Masked, direction-summed (d = 1, 4 directions), symmetric, normalised GLCM."""
    P = np.zeros((n_levels, n_levels), dtype=np.float64)
    H, W = image.shape
    for d in range(len(dy_arr)):
        dy = dy_arr[d] * distance
        dx = dx_arr[d] * distance
        y0 = max(0, -dy); y1 = min(H, H - dy)
        x0 = max(0, -dx); x1 = min(W, W - dx)
        for y in range(y0, y1):
            for x in range(x0, x1):
                if not mask[y, x]:
                    continue
                yy = y + dy; xx = x + dx
                if not mask[yy, xx]:
                    continue
                i = image[y, x]; j = image[yy, xx]
                P[i, j] += 1.0
                if symmetric:
                    P[j, i] += 1.0
    total = P.sum()
    if total > 0.0:
        P /= total
    return P


def build_glcm(q: np.ndarray, mask: np.ndarray, n_levels: int,
               distance: int = 1, symmetric: bool = True) -> np.ndarray:
    return _glcm_core(np.ascontiguousarray(q.astype(np.int64)),
                      np.ascontiguousarray(mask.astype(np.bool_)),
                      _DY, _DX, distance, symmetric, n_levels)

NPS_BINS = 8
LBP_BINS = 10


# --------------------------------------------------------------------------
# LBP (rotation-invariant uniform, P=8, R=1)
# --------------------------------------------------------------------------
@njit(cache=True)
def _lbp_riu2(img, mask, out):
    H, W = img.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if not mask[i, j]:
                continue
            c = img[i, j]
            b0 = 1 if img[i - 1, j - 1] >= c else 0
            b1 = 1 if img[i - 1, j] >= c else 0
            b2 = 1 if img[i - 1, j + 1] >= c else 0
            b3 = 1 if img[i, j + 1] >= c else 0
            b4 = 1 if img[i + 1, j + 1] >= c else 0
            b5 = 1 if img[i + 1, j] >= c else 0
            b6 = 1 if img[i + 1, j - 1] >= c else 0
            b7 = 1 if img[i, j - 1] >= c else 0
            bits = (b0, b1, b2, b3, b4, b5, b6, b7)
            trans = 0
            ones = 0
            for k in range(8):
                ones += bits[k]
                if bits[k] != bits[(k + 1) % 8]:
                    trans += 1
            out[ones if trans <= 2 else 9] += 1.0
    return out


def lbp_hist(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h = np.zeros(LBP_BINS, dtype=np.float64)
    _lbp_riu2(np.ascontiguousarray(patch.astype(np.float64)),
              np.ascontiguousarray(mask), h)
    s = h.sum()
    return (h / s) if s > 0 else h


# --------------------------------------------------------------------------
# Noise power spectrum (radially averaged, detrended)
# --------------------------------------------------------------------------
def _radial_bins(n: int) -> Tuple[np.ndarray, int]:
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    edges = np.linspace(0, 0.5, NPS_BINS + 1)
    idx = np.digitize(r, edges) - 1
    idx[idx >= NPS_BINS] = NPS_BINS - 1
    idx[r == 0] = -1
    return idx.astype(np.int64), NPS_BINS


_RB_CACHE: Dict[int, Tuple[np.ndarray, int]] = {}


def nps_features(patch: np.ndarray) -> np.ndarray:
    """log10 radial NPS (NPS_BINS) + peak frequency + centroid frequency."""
    n = patch.shape[0]
    if n not in _RB_CACHE:
        _RB_CACHE[n] = _radial_bins(n)
    idx, nb = _RB_CACHE[n]

    d = patch - ndi.gaussian_filter(patch, 4.0, mode="nearest")   # detrend anatomy
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]
    P = np.abs(np.fft.fft2(d * w)) ** 2
    prof = np.zeros(nb)
    for b in range(nb):
        m = idx == b
        prof[b] = P[m].mean() if m.any() else 0.0
    tot = prof.sum() + 1e-12
    fc = np.linspace(0.5 / nb / 2, 0.5 - 0.5 / nb / 2, nb)
    peak = fc[int(np.argmax(prof))]
    centroid = float((prof * fc).sum() / tot)
    return np.concatenate([np.log10(prof / tot + 1e-8), [peak, centroid]])


# --------------------------------------------------------------------------
# Full per-patch descriptor
# --------------------------------------------------------------------------
def descriptor_names(gray_levels: int = 32) -> List[str]:
    names = list(HARALICK_FEATURE_NAMES)
    names += [f"nps_b{i}" for i in range(NPS_BINS)] + ["nps_peak_f", "nps_centroid_f"]
    names += ["grad_mean", "grad_std", "grad_p90"]
    names += ["log_e_s1", "log_e_s2", "log_e_s4"]
    names += [f"lbp_{i}" for i in range(LBP_BINS)]
    names += ["amp_std", "hu_skew", "hu_kurt", "hu_noise_sigma"]
    return names


N_DESC = len(descriptor_names())


def patch_descriptor(patch_hu: np.ndarray, mask: np.ndarray,
                     win: Tuple[float, float], gray_levels: int = 32) -> np.ndarray:
    """Descriptor of one patch.  ``win`` is the (lo, hi) HU quantisation window."""
    p = patch_hu.astype(np.float32)
    if not mask.all():                       # fill the small uncovered remainder
        p = p.copy()
        p[~mask] = float(p[mask].mean()) if mask.any() else 0.0

    q = quantize(p, win[0], win[1], gray_levels)
    har = compute_features(build_glcm(q, mask, gray_levels))

    # Everything below the GLCM works on the patch rescaled into the organ's
    # own HU window, which makes the bank invariant to the absolute scale
    # mismatch between clinical HU and reconstructed attenuation.
    span = max(win[1] - win[0], 1e-6)
    pn = (p - win[0]) / span * 1000.0

    nps = nps_features(pn)

    gy, gx = np.gradient(pn)
    g = np.sqrt(gy ** 2 + gx ** 2)[mask]
    scale = float(pn[mask].std()) + 1e-6
    grad = np.array([g.mean() / scale, g.std() / scale, np.percentile(g, 90) / scale])

    logs = []
    for s in (1.0, 2.0, 4.0):
        e = ndi.gaussian_laplace(pn, s, mode="nearest")[mask]
        logs.append(np.log10(float(np.mean(e ** 2)) + 1e-8))

    lbp = lbp_hist(pn, mask)

    v = pn[mask].astype(np.float64)
    mu, sd = v.mean(), v.std() + 1e-8
    # one deliberately absolute feature: the high-pass noise sigma in HU
    hp = (p - ndi.gaussian_filter(p, 2.0, mode="nearest"))[mask]
    noise_sigma = float(1.4826 * np.median(np.abs(hp - np.median(hp))))
    first = np.array([sd, float(np.mean(((v - mu) / sd) ** 3)),
                      float(np.mean(((v - mu) / sd) ** 4)), noise_sigma])

    out = np.concatenate([har, nps, grad, np.array(logs), lbp, first])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class PatchSpec:
    size: int = 32
    stride: int = 16
    min_coverage: float = 0.85
    gray_levels: int = 32


DEFAULT_SPEC: Dict[str, PatchSpec] = {
    "liver": PatchSpec(32, 12, 0.85, 32),
    "lung":  PatchSpec(32, 12, 0.85, 32),
    "bone":  PatchSpec(16, 6, 0.60, 16),     # ribs/vertebrae are thin
    # fat / soft / organ are fragmented once the labelled organs are carved
    # out of them, so they need smaller, more permissive windows or they yield
    # no patches at all on a test slice
    "soft":  PatchSpec(20, 8, 0.70, 24),
    "fat":   PatchSpec(24, 10, 0.75, 24),
    "organ": PatchSpec(20, 8, 0.75, 24),
}


def extract_patches(hu: np.ndarray, mask: np.ndarray, spec: PatchSpec,
                    max_patches: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sliding-window patches whose mask coverage exceeds ``min_coverage``."""
    H, W = hu.shape
    s, st = spec.size, spec.stride
    ys, xs = np.where(mask)
    if ys.size == 0:
        return []
    y0, y1 = max(0, ys.min() - s), min(H - s, ys.max() + 1)
    x0, x1 = max(0, xs.min() - s), min(W - s, xs.max() + 1)
    integ = np.cumsum(np.cumsum(mask.astype(np.int32), 0), 1)

    def cov(y, x):
        a = integ[y + s - 1, x + s - 1]
        b = integ[y - 1, x + s - 1] if y > 0 else 0
        c = integ[y + s - 1, x - 1] if x > 0 else 0
        d = integ[y - 1, x - 1] if y > 0 and x > 0 else 0
        return (a - b - c + d) / float(s * s)

    out = []
    for y in range(int(y0), int(y1) + 1, st):
        for x in range(int(x0), int(x1) + 1, st):
            if y + s > H or x + s > W:
                continue
            if cov(y, x) >= spec.min_coverage:
                out.append((hu[y:y + s, x:x + s], mask[y:y + s, x:x + s]))
    if max_patches is not None and len(out) > max_patches:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(len(out), max_patches, replace=False)
        out = [out[i] for i in sel]
    return out


def describe_region(hu: np.ndarray, mask: np.ndarray, spec: PatchSpec,
                    win: Optional[Tuple[float, float]] = None,
                    max_patches: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """(n_patches, N_DESC) descriptor cloud for one organ region."""
    if win is None:
        win = roi_window(hu, mask)
    patches = extract_patches(hu, mask, spec, max_patches, rng)
    if not patches:
        return np.zeros((0, N_DESC), dtype=np.float32)
    return np.stack([patch_descriptor(p, m, win, spec.gray_levels)
                     for p, m in patches], 0)
