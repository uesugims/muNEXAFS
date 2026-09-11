"""Transmission and optical-density calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OpticalDensityResult:
    """Derived OD data and the mask of pixels with valid input."""

    optical_density: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    i0: NDArray[np.float64]
    current_correction_applied: bool


def compute_optical_density(
    transmission: NDArray[np.float64],
    i0: NDArray[np.float64],
    *,
    source_current: NDArray[np.float64] | None = None,
    current_reference: float = 300.0,
    apply_current_correction: bool = False,
) -> OpticalDensityResult:
    """Compute ``OD = log(I0 / I)`` without modifying the input arrays.

    ``transmission`` is ``(energy, y, x)`` and ``i0`` is ``(energy,)`` or
    broadcastable to that shape. Invalid and non-positive values become NaN.
    Source-current correction is opt-in; zero or non-finite currents are
    rejected rather than silently divided by zero.
    """

    transmission = np.asarray(transmission, dtype=np.float64)
    i0 = np.asarray(i0, dtype=np.float64)
    if transmission.ndim != 3:
        raise ValueError("transmission must have shape (energy, y, x)")
    if i0.shape not in {(transmission.shape[0],), transmission.shape}:
        raise ValueError("i0 must have shape (energy,) or match transmission")

    corrected_i0 = i0.copy()
    corrected_transmission = transmission.copy()
    corrected = False
    if apply_current_correction:
        if source_current is None:
            raise ValueError("source_current is required for current correction")
        current = np.asarray(source_current, dtype=np.float64)
        if current.shape != (transmission.shape[0],):
            raise ValueError("source_current must have shape (energy,)")
        if not np.isfinite(current).all() or (current <= 0).any():
            raise ValueError("source_current must contain finite positive values")
        scale = current_reference / current
        corrected_transmission = corrected_transmission * scale[:, None, None]
        if corrected_i0.ndim == 1:
            corrected_i0 = corrected_i0 * scale
        else:
            corrected_i0 = corrected_i0 * scale[:, None, None]
        corrected = True

    i0_for_image = (
        corrected_i0[:, None, None] if corrected_i0.ndim == 1 else corrected_i0
    )
    valid = np.isfinite(corrected_transmission) & np.isfinite(i0_for_image)
    valid &= corrected_transmission > 0
    valid &= i0_for_image > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        od = np.where(valid, np.log(i0_for_image / corrected_transmission), np.nan)
    return OpticalDensityResult(od, valid, corrected_i0, corrected)
