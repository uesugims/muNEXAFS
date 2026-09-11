"""Frame-to-frame image registration for STXM stacks.

The registration layer is intentionally independent of Qt.  It estimates a
translation for each frame against a reference frame and returns a shifted
copy, leaving the reader's raw data untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as ndimage_shift
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    registered: NDArray[np.float64]
    shifts_xy: NDArray[np.float64]
    reference_index: int
    reference_mode: str = "fixed"
    upsample_factor: int = 10


def register_translation_stack(
    stack: NDArray[np.float64],
    *,
    reference_index: int = 0,
    upsample_factor: int = 10,
    interpolation_order: int = 1,
    reference_mode: str = "fixed",
) -> RegistrationResult:
    """Register ``(energy, y, x)`` frames by translation.

    ``shifts_xy`` is ``(energy, 2)`` in image coordinates ``(x, y)``.  The
    reference frame has zero shift.  NaNs are replaced by the frame median
    only for phase-correlation estimation; the returned registered stack
    retains NaNs through the interpolation mask convention.
    """

    data = np.asarray(stack, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("stack must have shape (energy, y, x)")
    if not 0 <= reference_index < data.shape[0]:
        raise ValueError("reference_index is out of range")
    if reference_mode not in {"fixed", "previous"}:
        raise ValueError("reference_mode must be 'fixed' or 'previous'")
    reference = _finite_frame(data[reference_index])
    shifts_xy = np.zeros((data.shape[0], 2), dtype=np.float64)
    registered = np.empty_like(data)
    registered[reference_index] = data[reference_index]
    if reference_mode == "previous":
        order = list(range(reference_index + 1, data.shape[0])) + list(range(reference_index - 1, -1, -1))
    else:
        order = list(range(data.shape[0]))
    for index in order:
        if index == reference_index:
            continue
        frame = data[index]
        shift_yx, _, _ = phase_cross_correlation(
            reference, _finite_frame(frame), upsample_factor=upsample_factor
        )
        shifts_xy[index] = (float(-shift_yx[1]), float(-shift_yx[0]))
        registered[index] = ndimage_shift(
            frame, shift=shift_yx, order=interpolation_order, mode="constant", cval=np.nan,
            prefilter=interpolation_order > 1,
        )
        if reference_mode == "previous":
            reference = _finite_frame(registered[index])
    return RegistrationResult(registered, shifts_xy, reference_index, reference_mode, upsample_factor)


def _finite_frame(frame: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(frame, dtype=np.float64).copy()
    finite = np.isfinite(result)
    fill = float(np.nanmedian(result)) if finite.any() else 0.0
    result[~finite] = fill
    return result
