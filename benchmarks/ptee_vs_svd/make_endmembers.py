"""Synthesize distinct, realistic carbon K-edge endmember spectra.

For a controlled method comparison the endmembers must be spectrally *distinct*
(otherwise spectral confusion, not phase area, drives the result).  This writes
``data/endmembers.npz`` with one matrix phase plus several minor phases that
share a common absorption-edge continuum but each carry a sharp pi* resonance at
a unique energy — the standard, unambiguous phantom design.

The energy grid (and, if reachable, its exact values) comes from the real UVSOR
scan so the axis and sampling are realistic; only the peak positions are chosen.
Run ``extract_endmembers.py`` instead to use spectra taken straight from data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REAL_HDR = "/Volumes/Extreme Pro/stxm/UVSOR/211123/UV_211123007/UV_211123007.hdr"
# (peak center eV, width eV, peak OD height); index 0 is the matrix.
_PHASES = [
    ("matrix", 285.1, 0.9, 0.6),
    ("minor1", 286.6, 0.6, 1.0),
    ("minor2", 287.6, 0.5, 1.1),
    ("minor3", 288.6, 0.5, 1.1),
    ("minor4", 289.6, 0.6, 1.0),
    ("minor5", 290.8, 0.7, 0.9),
]


def _energies() -> np.ndarray:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from muaxis.io.stxm import read_header
        return np.asarray(read_header(REAL_HDR).stack_axis_energy_eV, float)
    except Exception:
        return np.linspace(280.0, 299.8, 114)


def main() -> int:
    E = _energies()
    # Shared absorption-edge continuum (a smooth sigmoid step) + a small slope.
    cont = 0.35 + 0.55 / (1.0 + np.exp(-(E - 288.0) / 1.6)) + 0.02 * (E - E[0]) / (E[-1] - E[0])
    names, spectra = [], []
    for name, c, w, h in _PHASES:
        names.append(name)
        spectra.append(cont + h * np.exp(-((E - c) / w) ** 2))
    spectra = np.asarray(spectra)

    out = Path(__file__).with_name("data") / "endmembers.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, energies=E, endmembers=spectra, names=np.array(names),
             source="synthetic distinct pi* peaks on a real energy grid")
    print(f"saved {len(spectra)} synthetic endmembers x {len(E)} energies -> {out}")
    for n, c, w, h in _PHASES:
        print(f"  {n:8s} pi* peak {h:.1f} OD at {c:.1f} eV (width {w} eV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
