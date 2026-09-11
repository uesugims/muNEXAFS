"""Reference-spectrum R² matching and RGBY score maps."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray

@dataclass(frozen=True, slots=True)
class SpectralFitResult:
    r2: NDArray[np.float64]          # (reference, y, x)
    rgb_y: NDArray[np.float32]       # (y, x, 4), premultiplied display colors
    references: NDArray[np.float64]  # (reference, energy)
    r2_floor: float
    normalization: str
    effective_references: NDArray[np.float64] | None = None
    raw_references: NDArray[np.float64] | None = None
    analysis_parameters: dict = field(default_factory=dict)

def _normalize(a: NDArray[np.float64], mode: str) -> NDArray[np.float64]:
    if mode == "None": return a
    lo = np.nanmin(a, axis=-1, keepdims=True); hi = np.nanmax(a, axis=-1, keepdims=True)
    if mode == "Min–max": return np.divide(a-lo, hi-lo, out=np.zeros_like(a), where=hi != lo)
    # Area-normalized spectra are useful when absolute OD should be removed.
    return np.divide(a, np.nanmax(np.abs(a), axis=-1, keepdims=True), out=np.zeros_like(a), where=np.nanmax(np.abs(a), axis=-1, keepdims=True) != 0)

def spectral_r2_map(od_stack: NDArray[np.float64], energies_eV: NDArray[np.float64], references: NDArray[np.float64], reference_energies: NDArray[np.float64] | None = None, *, normalization: str = "Min–max", r2_floor: float = 0.9) -> SpectralFitResult:
    od = np.asarray(od_stack, float); refs = np.asarray(references, float)
    if od.ndim != 3 or refs.ndim != 2 or refs.shape[0] not in (3, 4): raise ValueError("od_stack=(energy,y,x), references=(3 or 4,energy)")
    e = np.asarray(energies_eV, float)
    if reference_energies is not None:
        re = np.asarray(reference_energies, float)
        refs = np.asarray([np.interp(e, re, row, left=np.nan, right=np.nan) for row in refs])
    y = _normalize(od.reshape(od.shape[0], -1).T, normalization)
    active_refs = np.isfinite(refs).all(axis=1)
    # None-selected RGBY channels are represented by NaN rows and contribute
    # no score or color to the resulting map.
    safe_refs = np.where(np.isfinite(refs), refs, 0.0)
    r = _normalize(safe_refs, normalization)
    valid = np.isfinite(y).all(axis=1)
    scores = np.full((refs.shape[0], y.shape[0]), np.nan)
    # Standard coefficient of determination: compare residual variance with
    # the variance of the measured spectrum at each pixel.
    denom = np.sum((y-y.mean(axis=1, keepdims=True))**2, axis=1)
    for i in range(refs.shape[0]):
        if not active_refs[i]:
            continue
        residual = np.sum((y-r[i])**2, axis=1)
        ratio = np.divide(residual[valid], denom[valid], out=np.zeros_like(residual[valid]), where=denom[valid] > 0)
        scores[i, valid] = np.where(denom[valid] > 0, 1.0 - ratio, 0.0)
    scores = scores.reshape((refs.shape[0],) + od.shape[1:])
    weights = np.clip((scores-r2_floor) / max(1e-12, 1-r2_floor), 0, 1)
    weights[~np.isfinite(weights)] = 0.0
    if refs.shape[0] == 3: weights = np.concatenate([weights, np.zeros((1,)+weights.shape[1:])])
    rgb_y = np.empty((*od.shape[1:], 4), np.float32); rgb_y[..., 0] = np.clip(weights[0] + weights[3], 0, 1); rgb_y[..., 1] = np.clip(weights[1] + weights[3], 0, 1); rgb_y[..., 2] = weights[2]; rgb_y[..., 3] = np.nanmax(weights, axis=0)
    effective = np.where(active_refs[:, None], r, np.nan)
    return SpectralFitResult(scores, rgb_y, refs, float(r2_floor), normalization,
                             effective_references=effective)
