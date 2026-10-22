"""Organ partition of a slice and composition of the spatially varying beta field.

A TotalSegmentator (v2) label slice gives liver / lung / bone / other organs. The unlabelled
remainder of the body is completed by HU thresholds of the LDCT (FDK) image into fat / soft,
and everything outside the body is air, so every pixel belongs to exactly one class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi

# --- TotalSegmentator v2 label groups -------------------------------------
TS_LIVER: List[int] = [5]
TS_LUNG: List[int] = [10, 11, 12, 13, 14]
TS_BONE: List[int] = (
    [25] + list(range(26, 51))          # sacrum + vertebrae C1..L5
    + list(range(92, 116))              # ribs L1-12 / R1-12
    + [116, 117]                        # sternum, costal cartilages
    + list(range(69, 81))               # limb / pelvic bones
)
TS_ORGAN: List[int] = [1, 2, 3, 4, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20,
                       21, 22, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                       61, 62, 63, 64, 65, 66, 67, 68]

#: index 0 (air) is never regularised differently
CLASS_NAMES: Tuple[str, ...] = ("air", "lung", "fat", "soft", "organ", "liver", "bone")
TUNABLE: Tuple[str, ...] = ("lung", "fat", "soft", "organ", "liver", "bone")


@dataclass
class OrganPartition:
    index_map: np.ndarray                    # (H, W) int8 index into CLASS_NAMES
    names: Tuple[str, ...] = CLASS_NAMES
    body: np.ndarray = field(default=None)   # (H, W) bool

    def mask(self, name: str) -> np.ndarray:
        return self.index_map == self.names.index(name)

    def areas(self) -> Dict[str, int]:
        return {n: int((self.index_map == i).sum()) for i, n in enumerate(self.names)}


def body_mask_from_hu(hu: np.ndarray, thresh: float = -350.0, min_frac: float = 0.02) -> np.ndarray:
    """Largest connected component above ``thresh`` HU, holes filled."""
    m = ndi.binary_opening(hu > thresh, np.ones((3, 3)), iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return np.ones_like(hu, dtype=bool)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    if sizes.max() < min_frac * hu.size:
        return np.ones_like(hu, dtype=bool)
    return ndi.binary_fill_holes(lab == (np.argmax(sizes) + 1))


def build_partition(label_slice: np.ndarray, hu: np.ndarray,
                    fat_hu: float = -30.0, bone_hu: float = 250.0) -> OrganPartition:
    """Segmentation labels where defined, HU classes (fat/soft/bone/lung) elsewhere in the body."""
    idx = np.zeros(hu.shape, dtype=np.int8)
    body = body_mask_from_hu(hu)
    idx[body & (hu <= fat_hu)] = CLASS_NAMES.index("fat")
    idx[body & (hu > fat_hu)] = CLASS_NAMES.index("soft")
    idx[body & (hu >= bone_hu)] = CLASS_NAMES.index("bone")
    idx[body & (hu < -500.0)] = CLASS_NAMES.index("lung")

    for labels, name in ((TS_ORGAN, "organ"), (TS_LUNG, "lung"), (TS_LIVER, "liver"), (TS_BONE, "bone")):
        m = np.isin(label_slice, labels)
        if m.any():
            idx[m] = CLASS_NAMES.index(name)

    idx[(idx == 0) & body & (hu > -500.0)] = CLASS_NAMES.index("soft")
    return OrganPartition(index_map=idx, body=body)


def soft_indicators(part: OrganPartition, feather: float = 3.0) -> np.ndarray:
    """(C, H, W) Gaussian-feathered class indicators that sum to one at every pixel."""
    C = len(part.names)
    ind = np.stack([(part.index_map == c).astype(np.float32) for c in range(C)], 0)
    if feather > 0:
        for c in range(C):
            ind[c] = ndi.gaussian_filter(ind[c], feather, mode="nearest")
    s = ind.sum(0, keepdims=True)
    s[s < 1e-8] = 1.0
    return ind / s


class BetaField:
    """Turns per-class betas into the beta(r) field by geometric blending of the indicators:

        B(r) = exp( sum_c w_c(r) log beta_c )

    Classes missing from ``betas`` use ``beta_default``.
    """

    def __init__(self, part: OrganPartition, feather: float = 3.0, beta_default: float = 0.01):
        self.ind = soft_indicators(part, feather)
        self.beta_default = beta_default

    def __call__(self, betas: Dict[str, float]) -> np.ndarray:
        acc = np.zeros(self.ind.shape[1:], dtype=np.float32)
        for c, n in enumerate(CLASS_NAMES):
            acc += self.ind[c] * np.log(max(float(betas.get(n, self.beta_default)), 1e-8))
        return np.exp(acc).astype(np.float32)
