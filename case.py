"""Loading one 2-D case (one sinogram slice) and everything derived from it.

Case file (HDF5): ``sino_quarter`` (nview, nu) float32 low-dose sinogram, ``labels`` (ny, nx)
TotalSegmentator v2 label slice on the reconstruction grid, optional ``deg`` (view angles in
degrees) and optional attribute ``geom`` (JSON overrides of ``geometry.DEFAULT_GEOM``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from .audit import AuditWeights, ConsistencyModel, OrganAuditor
from .geometry import build_param
from .hu import MU_WATER, mu_to_hu
from .masks import TUNABLE, BetaField, OrganPartition, build_partition
from .solver import build_D, fdk, reconstruct
from .texture.bank import OrganBank, load_banks
from .texture.descriptors import DEFAULT_SPEC, extract_patches


@dataclass
class Case:
    name: str
    sino: np.ndarray                     # quarter-dose (LDCT) sinogram, (nview, nu)
    param: Dict[str, Any]
    labels: np.ndarray                   # TotalSegmentator label slice
    part: OrganPartition
    x_fdk: np.ndarray                    # LDCT image (FDK of ``sino``)
    D: np.ndarray
    banks: Dict[str, OrganBank]
    hu_priors: Dict[str, Dict[str, float]]
    organs: List[str]                    # classes present on this slice with a bank and enough patches
    cons: Optional[ConsistencyModel]     # None if loaded with consistency=False
    mu_water: float = MU_WATER
    meta: Dict[str, Any] = field(default_factory=dict)

    def auditor(self, tex_metric: str = "sw", seed: int = 0, max_patches: int = 260) -> OrganAuditor:
        return OrganAuditor(self.part, self.banks, self.cons, self.hu_priors, self.organs, self.x_fdk,
                            weights=AuditWeights(tex_metric=tex_metric), mu_water=self.mu_water,
                            max_patches=max_patches, seed=seed)

    def beta_field(self, feather: float = 3.0, beta_default: float = 0.01) -> BetaField:
        return BetaField(self.part, feather=feather, beta_default=beta_default)

    def reconstruct(self, beta, n_iter: int = 30) -> np.ndarray:
        """SART with a scalar beta or a {class: beta} dict (organ-specific field)."""
        B = self.beta_field()(beta) if isinstance(beta, dict) else np.float32(beta)
        return reconstruct(self.sino, self.param, B, n_iter=n_iter, D=self.D)


def load_case(path, bank_path, min_px: int = 600, min_patches: int = 8, floor_iters: int = 30,
              floor_beta: float = 2e-3, cache_dir=None, consistency: bool = True,
              verbose: bool = True) -> Case:
    """Load a case file and build the partition, auditor inputs and the
    chi-square floor (an almost unregularised SART fit, cached in ``cache_dir`` if given).
    ``consistency=False`` skips the projection-consistency model (enough for reconstruction with
    given betas, not for the search)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        sino = f["sino_quarter"][()].astype(np.float32)
        labels = f["labels"][()]
        deg = f["deg"][()] if "deg" in f else None
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    geom = json.loads(meta.pop("geom")) if "geom" in meta else {}
    param = build_param(deg=deg, **geom)
    x_fdk = fdk(sino, param)
    part = build_partition(labels, mu_to_hu(x_fdk, MU_WATER))
    banks, priors = load_banks(bank_path)

    areas = part.areas()
    organs = [o for o in TUNABLE if o in banks and areas.get(o, 0) >= min_px]
    hu0 = mu_to_hu(x_fdk, MU_WATER)
    organs = [o for o in organs
              if len(extract_patches(hu0, part.mask(o), DEFAULT_SPEC.get(o, DEFAULT_SPEC["soft"]))) >= min_patches]

    D = build_D(param)
    cons = None
    if consistency:
        cons = ConsistencyModel.build(sino, param, {o: part.mask(o) for o in organs})
        fp_cache = (Path(cache_dir) / f"_floor_{path.stem}_it{floor_iters}_b{floor_beta:g}.npy"
                    if cache_dir else None)
        if fp_cache is not None and fp_cache.exists():
            x_floor = np.load(fp_cache)
        else:
            x_floor = reconstruct(sino, param, np.float32(floor_beta), n_iter=floor_iters, D=D)
            if fp_cache is not None:
                fp_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(fp_cache, x_floor.astype(np.float32))
        cons.set_floor(x_floor)

    if verbose:
        print(f"[case {path.stem}] organs={organs}  areas=" + " ".join(f"{o}:{areas[o]}" for o in organs))
    return Case(name=path.stem, sino=sino, param=param, labels=labels, part=part, x_fdk=x_fdk, D=D,
                banks=banks, hu_priors=priors, organs=organs, cons=cons, meta=meta)
