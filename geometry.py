"""2-D fan-beam geometry."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: default 2-D fan-beam geometry (override per case via the ``geom`` attribute)
DEFAULT_GEOM: Dict[str, Any] = dict(
    nx=512, ny=512, dx=1.0, dy=1.0,
    nu=736, du=1.2858, off_a=0.0,
    DSD=1085.6, DSO=595.0,
    nview=2304, scan_angle=360.0,
)


def build_param(deg: Optional[np.ndarray] = None, **overrides) -> Dict[str, Any]:
    """Parameter dict consumed by the projector / FDK functions.

    ``deg`` are the view angles in degrees; if omitted, ``nview`` equally spaced views over
    ``scan_angle`` are used. Any key of ``DEFAULT_GEOM`` can be overridden.
    """
    g = dict(DEFAULT_GEOM)
    g.update(overrides)
    p = dict(g)
    p["da"] = g["du"] / g["DSD"]                      # radians per detector cell
    if deg is None:
        deg = np.linspace(0.0, g["scan_angle"], int(g["nview"]), endpoint=False)
    p["deg"] = np.asarray(deg, dtype=np.float32)
    p["nview"] = int(len(p["deg"]))
    return p
