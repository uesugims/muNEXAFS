"""Pre-map calculations used before segmentation."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


def _slice(energies: NDArray[np.float64], energy_range: tuple[float, float]) -> tuple[int, int]:
    low, high = sorted(map(float, energy_range))
    indices = np.flatnonzero((energies >= low) & (energies <= high))
    if len(indices) < 1:
        raise ValueError("energy range contains no frames")
    return int(indices[0]), int(indices[-1]) + 1


def pre_edge_subtract(
    stack: NDArray[np.float64],
    energies: NDArray[np.float64],
    pre_edge_range: tuple[float, float],
) -> NDArray[np.float64]:
    """Return the stack with the per-pixel pre-edge average subtracted.

    Each frame becomes ``max(OD_k - mean(OD over the pre-edge range), 0)``.
    Removing the non-resonant baseline (roughly the thickness/density
    contribution) makes the frames reflect chemical (resonant) absorption, which
    is what the original tool clustered on — the correct input for
    composition-based segmentation, rather than raw OD.  Negative values
    (post-edge below the pre-edge baseline: noise / edge artefacts) are clipped
    to zero to match the original ``max(pmat, 0)`` contrast; NaN pixels are
    preserved so segmentation's finite-mask still excludes them.
    """
    stack = np.asarray(stack, dtype=np.float64)
    start, stop = _slice(energies, pre_edge_range)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pre = np.nanmean(stack[start:stop], axis=0)  # (y, x)
    return np.maximum(stack - pre[None], 0.0)


def net_absorption_map(
    stack: NDArray[np.float64],
    energies: NDArray[np.float64],
    pre_edge_range: tuple[float, float],
    post_edge_range: tuple[float, float],
) -> NDArray[np.float64]:
    """Net absorption summed over the post-edge range.

    For every pixel the pre-edge average OD is subtracted from each post-edge
    frame, and the differences are **summed** over the post-edge frames.  This
    reproduces the original tool's ``diff_sum`` map exactly (a plain per-frame
    sum, not a ``dE``-weighted trapezoidal integral), so the map contrast
    matches.  The pre-edge average is the non-resonant baseline — roughly
    proportional to thickness/density — so removing it leaves only the chemical
    (resonant) absorption.  Two regions with the same chemistry but different
    thickness are therefore grouped together, unlike a plain OD integral which
    separates them by thickness.
    """
    stack = np.asarray(stack, dtype=np.float64)
    pre_start, pre_stop = _slice(energies, pre_edge_range)
    post_start, post_stop = _slice(energies, post_edge_range)
    with warnings.catch_warnings():
        # An all-NaN pixel column (a dead pixel across the whole pre-edge)
        # legitimately averages to NaN; that is handled below, so silence the
        # cosmetic "Mean of empty slice" warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        pre = np.nanmean(stack[pre_start:pre_stop], axis=0)  # (y, x) baseline
    net = stack[post_start:post_stop] - pre[None]        # (k, y, x)
    result = np.nansum(net, axis=0)
    # Match the original diff_sum contrast: negative net absorption (post-edge
    # weaker than the pre-edge baseline — noise / edge artefacts) is clipped to
    # zero before display and thresholding, so the positive absorption spans the
    # full contrast range instead of being compressed by out-of-interest lows.
    result = np.maximum(result, 0.0)
    # A pixel that is invalid (NaN) at every post-edge frame has no net
    # absorption; keep it NaN so segmentation's finite-mask excludes it rather
    # than clustering it as a spurious zero.
    result[np.all(np.isnan(net), axis=0)] = np.nan
    return result


def peak_map(
    stack: NDArray[np.float64], energies: NDArray[np.float64], energy_range: tuple[float, float]
) -> NDArray[np.float64]:
    """Return the peak-position (energy) map for each pixel.

    The map value is the energy at which that pixel reaches its maximum
    intensity, not the maximum intensity itself.  This is what allows a
    color palette to encode the local peak position.
    """
    start, stop = _slice(energies, energy_range)
    selected = np.asarray(stack[start:stop], dtype=np.float64)
    valid = np.isfinite(selected).any(axis=0)
    safe = np.where(np.isfinite(selected), selected, -np.inf)
    peak_index = np.argmax(safe, axis=0)
    result = np.asarray(energies[start:stop], dtype=np.float64)[peak_index]
    return np.where(valid, result, np.nan)
