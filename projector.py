"""Pixel-driven fan-beam forward projector, its adjoint, FDK filtering and the quadratic smoother.

The backprojector accumulates into one private image per thread (fixed, strided view subsets,
summed in thread order), so repeated calls on the same input give the same result. A plain
``prange`` scatter (``img[idx] += val``) loses concurrent updates and is not reproducible.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numba
import numpy as np
from numba import jit, njit, prange
from scipy import fft


# ---------------------------------------------------------------------------
# forward / back projection
# ---------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def _fp_core(img, deg, nview, dsd, dso, nx, ny, dx, dy, nu, da, off_a):
    sino = np.zeros(nu * nview, dtype=np.float32)
    for ii in prange(nu * nview):
        iview = ii // nu
        sinval = math.sin(deg[iview] / 180.0 * math.pi)
        cosval = math.cos(deg[iview] / 180.0 * math.pi)
        a = ((ii % nu) - (nu - 1.0) / 2.0 - off_a) * da
        dist = dy / math.cos(a)
        temp = 0.0
        for iy in range(ny):
            posy = (iy - (ny - 1.0) / 2.0) * dy
            posx = math.tan(a) * (posy + dso)
            rx = (posx * cosval + posy * sinval) / dx + (nx - 1.0) / 2.0
            ry = (-posx * sinval + posy * cosval) / dy + (ny - 1.0) / 2.0
            if rx > 0 and rx < nx - 1 and ry > 0 and ry < ny - 1:
                wx = rx - int(rx)
                wy = ry - int(ry)
                temp += ((1.0 - wx) * (1.0 - wy) * img[int(ry) * nx + int(rx)] +
                         wx * (1.0 - wy) * img[int(ry) * nx + int(rx) + 1] +
                         (1.0 - wx) * wy * img[int(ry + 1) * nx + int(rx)] +
                         wx * wy * img[int(ry + 1) * nx + int(rx) + 1]) * dist
        sino[ii] = temp
    return sino


@njit(parallel=True, cache=True)
def _bp_core(proj, deg, nview, dsd, dso, nx, ny, dx, dy, nu, da, off_a, nthreads):
    acc = np.zeros((nthreads, nx * ny), dtype=np.float64)
    for t in prange(nthreads):
        for iview in range(t, nview, nthreads):
            sinval = math.sin(deg[iview] / 180.0 * math.pi)
            cosval = math.cos(deg[iview] / 180.0 * math.pi)
            base = iview * nu
            for iu in range(nu):
                a = (iu - (nu - 1.0) / 2.0 - off_a) * da
                dist = dy / math.cos(a)
                tana = math.tan(a)
                val0 = proj[base + iu] * dist
                if val0 == 0.0:
                    continue
                for iy in range(ny):
                    posy = (iy - (ny - 1.0) / 2.0) * dy
                    posx = tana * (posy + dso)
                    x = posx * cosval + posy * sinval
                    y = -posx * sinval + posy * cosval
                    rx = x / dx + (nx - 1.0) / 2.0
                    ry = y / dy + (ny - 1.0) / 2.0
                    if rx > 0 and rx < nx - 1 and ry > 0 and ry < ny - 1:
                        ix = int(rx)
                        iyy = int(ry)
                        wx = rx - ix
                        wy = ry - iyy
                        p = iyy * nx + ix
                        acc[t, p] += (1.0 - wx) * (1.0 - wy) * val0
                        acc[t, p + 1] += wx * (1.0 - wy) * val0
                        acc[t, p + nx] += (1.0 - wx) * wy * val0
                        acc[t, p + nx + 1] += wx * wy * val0
    out = np.zeros(nx * ny, dtype=np.float64)
    for t in range(nthreads):
        for i in range(nx * ny):
            out[i] += acc[t, i]
    return out


def fp(img: np.ndarray, param: Dict[str, Any]) -> np.ndarray:
    """Forward projection A x, returns a (nview, nu) sinogram."""
    sino = _fp_core(np.asarray(img).flatten().astype(np.float32), param["deg"].astype(np.float32),
                    param["nview"], param["DSD"], param["DSO"], param["nx"], param["ny"],
                    param["dx"], param["dy"], param["nu"], param["da"], param["off_a"])
    return sino.reshape((param["nview"], param["nu"]))


def bp(proj: np.ndarray, param: Dict[str, Any]) -> np.ndarray:
    """Back projection A^T p (exact adjoint of ``fp``), returns a (ny, nx) image."""
    if proj.shape[0] == param["nu"] and proj.shape[1] == param["nview"]:
        proj = proj.T
    nt = min(int(numba.get_num_threads()), int(param["nview"]))
    img = _bp_core(np.ascontiguousarray(proj, dtype=np.float32).ravel(),
                   np.ascontiguousarray(param["deg"], dtype=np.float32),
                   int(param["nview"]), float(param["DSD"]), float(param["DSO"]),
                   int(param["nx"]), int(param["ny"]), float(param["dx"]), float(param["dy"]),
                   int(param["nu"]), float(param["da"]), float(param["off_a"]), nt)
    return img.reshape((param["ny"], param["nx"])).astype(np.float32)


# ---------------------------------------------------------------------------
# FDK filtering
# ---------------------------------------------------------------------------
def _ramp(n: int) -> np.ndarray:
    nn = np.arange(-n // 2, n // 2)
    h = np.zeros(nn.shape, dtype=np.float32)
    h[n // 2] = 1 / 4
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd]) ** 2
    return h


def _create_filter(filter_type: str, kernel: np.ndarray, order: int) -> np.ndarray:
    filt = np.abs(fft.fft(kernel))[:order // 2 + 1]
    w = 2 * np.pi * np.arange(filt.size) / order
    t = filter_type.lower()
    if t == "ram-lak":
        pass
    elif t == "shepp-logan":
        filt[1:] *= np.sin(w[1:] / 2) / (w[1:] / 2)
    elif t == "cosine":
        filt[1:] *= np.cos(w[1:] / 2)
    elif t == "hamming":
        filt[1:] *= 0.54 + 0.46 * np.cos(w[1:])
    elif t == "hann":
        filt[1:] *= (1 + np.cos(w[1:])) / 2
    else:
        raise ValueError(f"Invalid filter selected: {filter_type}")
    return np.concatenate([filt, filt[-2:0:-1]])


def filtering(proj: np.ndarray, param: Dict[str, Any], filter_type: str = "ram-lak") -> np.ndarray:
    """Fan-beam pre-weighting + ramp filtering. Works on a copy; the input is never modified."""
    nu, nview = param["nu"], param["nview"]
    proj = np.array(proj, dtype=np.float32, copy=True)
    if proj.shape[0] == nu and proj.shape[1] == nview:
        proj = proj.T
    proj = np.moveaxis(proj, 0, -1)                                  # (nu, nview)

    as_ = (-(np.arange(-nu / 2 + 0.5, nu / 2, 1)) - param["off_a"]) * param["da"]   # fan angle [rad]
    weight = (as_[:, np.newaxis] / np.sin(as_[:, np.newaxis])) ** 2
    weight[np.isnan(weight)] = 1
    for i in range(nview):
        proj[:, i] = proj[:, i] * np.cos(as_) * weight.flatten()

    filt_len = max(64, 2 ** int(np.ceil(np.log2(2 * nu))))
    filt = np.tile(_create_filter(filter_type, _ramp(filt_len), filt_len), (nview, 1)).T
    fproj = np.zeros((filt_len, nview), dtype=np.float32)
    fproj[filt_len // 2 - nu // 2:filt_len // 2 + nu // 2, :] = proj
    fproj = np.real(fft.ifft(fft.fft(fproj, axis=0) * filt, axis=0))
    proj = fproj[filt_len // 2 - nu // 2:filt_len // 2 + nu // 2, :] * \
        (2 * np.pi / nview) / (param["da"] * param["DSD"]) / 2 / (param["dy"] ** 2) * param.get("du", 1.0)
    return np.moveaxis(proj, 0, -1)


# ---------------------------------------------------------------------------
# quadratic smoother used by the regulariser
# ---------------------------------------------------------------------------
@jit(nopython=True)
def quadpenalty(img, k=1):
    """Inverse-distance-weighted (2k+1)^2 neighbourhood average (centre weight 1)."""
    sx, sy = img.shape
    tmp = np.zeros((sx + 2 * k, sy + 2 * k), dtype=np.float32)
    tmp[k:sx + k, k:sy + k] = img
    Sumimg = np.zeros((sx, sy), dtype=np.float32)
    weight_sum = 0.0
    for ix in range(-k, k + 1):
        for iy in range(-k, k + 1):
            if ix != 0 or iy != 0:
                weight = 1.0 / np.sqrt(float(ix * ix + iy * iy))
            else:
                weight = 1.0
            weight_sum += weight
            Sumimg += tmp[k + ix:k + sx + ix, k + iy:k + sy + iy] * weight
    return Sumimg / weight_sum
