"""Extract realistic NEXAFS endmember spectra from a real STXM scan.

Run once (needs the raw dataset) to produce ``data/endmembers.npz``, which the
phantom generator then uses so the benchmark is reproducible WITHOUT the drive.

    python extract_endmembers.py [HDR]

Default HDR: the UVSOR carbon K-edge scan UV_211123007.  The script clusters the
per-pixel OD spectra, then greedily selects the most mutually distinct cluster
means as endmembers (one matrix-like + several minor phases).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2, whiten

# Make the muaxis package importable when run from the benchmark folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from muaxis.io.stxm import read_stxm_scan
from muaxis.processing import compute_optical_density

DEFAULT_HDR = "/Volumes/Extreme Pro/stxm/UVSOR/211123/UV_211123007/UV_211123007.hdr"
N_CLUSTERS = 12
N_ENDMEMBERS = 6


def _minmax_rows(a: np.ndarray) -> np.ndarray:
    lo = np.nanmin(a, axis=1, keepdims=True)
    hi = np.nanmax(a, axis=1, keepdims=True)
    return np.divide(a - lo, hi - lo, out=np.zeros_like(a), where=hi != lo)


def _farthest_point_select(shapes: np.ndarray, n: int) -> list[int]:
    """Greedy max-min selection of the most mutually distinct rows (by shape)."""
    chosen = [int(np.argmax(np.linalg.norm(shapes - shapes.mean(0), axis=1)))]
    while len(chosen) < n:
        d = np.min([np.linalg.norm(shapes - shapes[c], axis=1) for c in chosen], axis=0)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def main() -> int:
    hdr = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HDR
    scan = read_stxm_scan(hdr)
    energies = np.asarray(scan.energies_eV, dtype=float)
    od = compute_optical_density(scan.transmission, scan.i0).optical_density
    ne = od.shape[0]
    flat = od.reshape(ne, -1).T                      # (pixel, energy)
    finite = np.isfinite(flat).all(axis=1)
    spectra = flat[finite]

    rng = np.random.default_rng(0)
    sample = spectra[rng.choice(len(spectra), size=min(20000, len(spectra)), replace=False)]
    shape_sample = _minmax_rows(sample)              # cluster by SHAPE, not thickness
    centroids, labels = kmeans2(whiten(shape_sample), N_CLUSTERS, seed=0, minit="++", missing="raise")

    # Mean RAW-OD spectrum of each cluster (keeps realistic amplitude/baseline).
    means, shapes = [], []
    for k in range(N_CLUSTERS):
        rows = shape_sample[labels == k]
        if len(rows) < 20:
            continue
        idx = labels == k
        means.append(sample[idx].mean(0))
        shapes.append(rows.mean(0))
    means = np.asarray(means)
    shapes = np.asarray(shapes)

    pick = _farthest_point_select(shapes, min(N_ENDMEMBERS, len(shapes)))
    endmembers = means[pick]                          # (n_end, energy) raw OD
    names = ["matrix"] + [f"minor{i}" for i in range(1, len(endmembers))]

    out = Path(__file__).with_name("data") / "endmembers.npz"
    np.savez(out, energies=energies, endmembers=endmembers, names=np.array(names),
             source=str(hdr))
    print(f"saved {len(endmembers)} endmembers x {ne} energies -> {out}")
    for n, s in zip(names, endmembers):
        print(f"  {n:8s}  OD@peak {s.max():.2f} at {energies[int(np.argmax(s))]:.1f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
