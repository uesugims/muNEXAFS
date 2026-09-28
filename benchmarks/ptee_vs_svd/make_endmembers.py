"""Synthesize carbon K-edge endmember spectra for the benchmark phantom.

The matrix is the bare absorption-edge baseline (a sigmoid step with a small
slope) with no characteristic peak.  Each minor phase is that baseline plus one
π* resonance of an organic functional group:

  minor1 285.1 eV aromatic C=C      minor3 287.5 eV aliphatic C-H
  minor2 286.6 eV ketone/phenol     minor4 288.6 eV carboxyl C=O

so every minor phase carries its own diagnostic peak (the favourable case for
the peak map).  The spectra are sampled on the measured energy grid of the
paper (109 points: 0.5 eV steps to 283.5 eV, 0.1 eV to 292 eV, 0.5 eV to 300 eV).
Run ``extract_endmembers.py`` instead to use spectra taken straight from data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# (name, center eV, width eV, height OD) of the π* peak added to the baseline.
_MINORS = [
    ("minor1", 285.1, 0.5, 1.0),   # aromatic C=C
    ("minor2", 286.6, 0.5, 1.0),   # ketone / phenol
    ("minor3", 287.5, 0.6, 1.0),   # aliphatic C-H
    ("minor4", 288.6, 0.5, 1.0),   # carboxyl C=O
]


def _energies() -> np.ndarray:
    # The measured energy grid of the paper (Section 2.1; 109 points).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phantom import MEASURED_ENERGIES
    return np.asarray(MEASURED_ENERGIES, float)


def main() -> int:
    E = _energies()
    g = lambda c, w, h: h * np.exp(-((E - c) / w) ** 2)
    # Baseline (= matrix): sigmoid edge step + small slope.
    base = 0.35 + 0.55 / (1.0 + np.exp(-(E - 288.0) / 1.6)) + 0.02 * (E - E[0]) / (E[-1] - E[0])
    names = ["matrix"] + [m[0] for m in _MINORS]
    spectra = np.asarray([base] + [base + g(c, w, h) for _, c, w, h in _MINORS])

    out = Path(__file__).with_name("data") / "endmembers.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, energies=E, endmembers=spectra, names=np.array(names),
             source="synthetic functional-group pi* peaks on the measured energy grid")
    print(f"saved {len(spectra)} synthetic endmembers x {len(E)} energies -> {out}")
    for n, s in zip(names, spectra):
        print(f"  {n:8s} OD max {s.max():.2f} at {E[s.argmax()]:.1f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
