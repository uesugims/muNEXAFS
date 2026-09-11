"""Synthetic STXM/NEXAFS phantom with several minor phases.

Reproducible and drive-free: it builds a large OD cube from the endmember
spectra saved by ``extract_endmembers.py`` (``data/endmembers.npz``), so the
ground-truth phase locations are known exactly.  Design goals:

* Large canvas (default 320x320) — makes the speed comparison meaningful.
* One dominant matrix phase plus several **minor phases** of decreasing area
  fraction and blob size, down to a tiny sparse one that is hard to find.
* A smooth **thickness/density field** multiplies every pixel's OD, so methods
  that key on total variance (PCA) can be fooled by thickness rather than
  chemistry.
* Photon (Poisson) noise applied at the transmission level, giving realistic,
  absorption-dependent noise controlled by a single ``i0_counts`` budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Phantom:
    energies: np.ndarray          # (E,)
    od_true: np.ndarray           # (E, Y, X) noise-free OD
    od: np.ndarray                # (E, Y, X) measured OD (with Poisson noise)
    abundances: np.ndarray        # (P, Y, X) per-phase abundance (sum ~ thickness)
    masks: np.ndarray             # (P, Y, X) bool ground-truth (phase dominant)
    endmembers: np.ndarray        # (P, E) reference spectra (raw OD)
    names: list[str]
    thickness: np.ndarray         # (Y, X)
    area_fraction: np.ndarray     # (P,)


# area fraction and blob radius (px) per minor phase; index 0 is the matrix.
# A geometric ladder of shrinking area exposes where each method breaks down.
_MINOR_LAYOUT = [
    dict(frac=0.045, r=(9, 16), n=10),    # minor1  ~4.5 %
    dict(frac=0.018, r=(6, 11), n=10),    # minor2  ~1.8 %
    dict(frac=0.007, r=(4, 7), n=14),     # minor3  ~0.7 %
    dict(frac=0.0025, r=(2, 4), n=22),    # minor4  ~0.25 %
    dict(frac=0.0010, r=(2, 3), n=30),    # minor5  ~0.1 %  (tiniest: hardest)
]


def _disk_field(shape, rng, n, r_range, target_frac):
    """Random overlapping disks until the covered fraction reaches target."""
    Y, X = shape
    field = np.zeros(shape, dtype=float)
    yy, xx = np.mgrid[0:Y, 0:X]
    guard = 0
    while field.mean() < target_frac and guard < 4000:
        cy, cx = rng.integers(0, Y), rng.integers(0, X)
        r = rng.integers(r_range[0], r_range[1] + 1)
        field[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1.0
        guard += 1
    return field


def load_endmembers(path: str | Path | None = None):
    path = Path(path) if path else Path(__file__).with_name("data") / "endmembers.npz"
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["energies"], float), np.asarray(d["endmembers"], float), [str(n) for n in d["names"]]


def build_phantom(size: int = 320, i0_counts: float = 350.0, seed: int = 0,
                  minor_level: float = 1.0, endmembers_path: str | Path | None = None) -> Phantom:
    """Build the phantom.

    ``minor_level`` is the peak abundance of a minor phase where it is present:
    each minor phase is *mixed into* the matrix at this fraction (not a pure
    region), so it is a subtle, low-contrast deviation from the matrix — the
    realistic "trace/minor phase" case that is hard for variance methods.
    ``i0_counts`` sets the photon budget (lower = noisier).
    """
    energies, endmembers, names = load_endmembers(endmembers_path)
    P = len(endmembers)
    rng = np.random.default_rng(seed)
    shape = (size, size)

    # Minor-phase abundance blobs (soft edges), each mixed into the matrix at a
    # modest level so it is a low-contrast trace, not a pure region.
    minor = np.zeros((P - 1, *shape))
    for i, cfg in enumerate(_MINOR_LAYOUT[: P - 1]):
        f = _disk_field(shape, rng, cfg["n"], cfg["r"], cfg["frac"])
        minor[i] = minor_level * gaussian_filter(f, 1.0)
    minor = np.clip(minor, 0, 1)
    minor_sum = np.clip(minor.sum(0), 0, 1)
    matrix = 1.0 - minor_sum                                     # matrix fills the rest
    abundances = np.concatenate([matrix[None], minor], axis=0)   # (P, Y, X), sum ~ 1

    # Smooth thickness/density field (0.6 .. 1.4) multiplies every pixel.
    thickness = 1.0 + 0.4 * gaussian_filter(rng.standard_normal(shape), size / 12.0)
    thickness = np.clip(thickness / thickness.mean(), 0.6, 1.4)

    # Noise-free OD = thickness * sum_p abundance_p * endmember_p(E)
    od_true = thickness[None] * np.tensordot(endmembers.T, abundances, axes=(1, 0))  # (E,Y,X)
    od_true = np.clip(od_true, 0, None)

    # Poisson noise at the transmission level: I = Poisson(I0 * exp(-OD)).
    trans = i0_counts * np.exp(-od_true)
    noisy = rng.poisson(np.clip(trans, 0, None)).astype(float)
    noisy = np.maximum(noisy, 0.5)                     # avoid log(0)
    od = np.log(i0_counts / noisy)

    # Ground truth: a minor phase is "present" where its abundance exceeds half
    # its peak level; the matrix mask is the complement of all minors.
    minor_masks = [minor[p - 1] >= 0.5 * minor_level for p in range(1, P)]
    matrix_mask = ~np.any(minor_masks, axis=0)
    masks = np.stack([matrix_mask, *minor_masks])
    area_fraction = masks.reshape(P, -1).mean(1)
    return Phantom(energies, od_true, od, abundances, masks, endmembers, names,
                   thickness, area_fraction)


if __name__ == "__main__":
    ph = build_phantom()
    print("phantom:", ph.od.shape, "phases:", ph.names)
    for n, f in zip(ph.names, ph.area_fraction):
        print(f"  {n:8s} area fraction {100*f:6.2f} %")
