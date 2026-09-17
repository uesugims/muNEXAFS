"""Non-circular validation of the PTEE *extraction* step.

The classification benchmark (benchmark.py) hands PTEE the known endmember
spectra and scores only the R² mapping.  This script instead tests the
*extraction* stage: it runs the peak-map + histogram-threshold extraction on the
noisy phantom image ALONE — without ever seeing the spectra used to build the
phantom — and then

  1. compares each extracted endmember with the (withheld) ground-truth spectrum
     that generated it (spectral error, peak-energy error), and
  2. feeds the *extracted* spectra into the R² classifier and compares the final
     detection F1 with the F1 obtained from the true spectra.

Conditions are swept over signal-to-noise (photon budget), mixing level (how
much a minor phase deviates from the matrix) and — on a controlled Gaussian-peak
phantom — peak separation and peak width.

    python extraction_validation.py

Writes outputs/extraction_metrics.json / .csv and outputs/fig_extraction.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phantom import build_phantom
from benchmark import ptee_r2_maps, best_threshold

OUT = HERE / "outputs"


# --------------------------------------------------------------------------- #
# Unsupervised PTEE extraction (image only: no reference spectra, no masks)    #
# --------------------------------------------------------------------------- #
def _pre_edge_subtract(od, n_pre=3):
    pre = np.nanmean(od[:n_pre], axis=0)
    return np.clip(od - pre[None], 0.0, None)


def _peak_map(odp, energy_smooth=1.0):
    """Energy index of maximum OD per pixel (peak map), lightly smoothed."""
    s = gaussian_filter1d(odp, energy_smooth, axis=0)
    return np.argmax(s, axis=0)


def _hist_threshold(img):
    """Paper threshold: midpoint between the most- and least-frequent grey
    values of a 256-bin histogram (background/zero bin excluded)."""
    v = img[np.isfinite(img)]
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        return hi
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist[0] = 0                                   # drop the background (near-zero) bin
    nz = np.flatnonzero(hist > 0)
    if nz.size < 2:
        return hi
    most = nz[np.argmax(hist[nz])]
    least = nz[np.argmin(hist[nz])]
    return float(0.5 * (centers[most] + centers[least]))


def ptee_extract(od, energies, n_phases, *, min_mode_frac=5e-4, peak_window=(284.0, 291.5)):
    """Extract (matrix + minor) endmember spectra from the OD cube alone.

    Peak map -> the populated peak-map energies are the candidate phase peaks.
    The most-populated energy is the matrix; each remaining candidate peak is
    refined by thresholding its energy image and averaging the selected cluster.
    Candidate peaks are restricted to the functional-group (π*) window, exactly
    as an analyst reads the peak map over the C K-edge features and not the
    post-edge σ* continuum, where photon noise makes the per-pixel maximum
    wander.  Returns (refs (n, E), peak_energies (n,)) with the matrix first.
    """
    odp = _pre_edge_subtract(od)
    E = od.shape[0]
    pk = _peak_map(odp)
    counts = np.bincount(pk.ravel(), minlength=E).astype(float)
    npix = pk.size
    in_win = (energies >= peak_window[0]) & (energies <= peak_window[1])
    # Candidate peak energies = local maxima of the peak-map histogram within the
    # π* window that hold at least ``min_mode_frac`` of the pixels.
    cand = [i for i in range(E)
            if in_win[i] and counts[i] >= max(3.0, min_mode_frac * npix)
            and counts[i] >= counts[max(0, i - 1)] and counts[i] >= counts[min(E - 1, i + 1)]]
    cand.sort(key=lambda i: counts[i], reverse=True)
    if not cand:
        cand = [int(np.argmax(counts))]
    matrix_e = cand[0]
    minor_es = cand[1:n_phases]                    # take the next strongest modes

    flat = od.reshape(E, -1)
    pk_flat = pk.ravel()
    refs, peak_es, minor_masks = [], [], []
    for te in minor_es:
        # Cluster of pixels whose maximum OD falls at this peak energy: where the
        # phase dominates it is close to pure, so its mean is a clean endmember.
        # A narrow ±1-index energy tolerance absorbs noise jitter of the argmax.
        m = np.isfinite(flat).all(axis=0) & (np.abs(pk_flat - te) <= 1)
        if m.sum() < 5:
            continue
        refs.append(np.nanmean(flat[:, m], axis=1))
        peak_es.append(energies[te])
        minor_masks.append(m)
    # Matrix = pixels not claimed by any extracted minor cluster.
    claimed = np.zeros(npix, bool)
    for m in minor_masks:
        claimed |= m
    mat_pixels = ~claimed
    matrix_ref = np.nanmean(flat[:, mat_pixels], axis=1) if mat_pixels.any() else np.nanmean(flat, axis=1)
    refs = [matrix_ref] + refs
    peak_es = [energies[matrix_e]] + peak_es
    return np.asarray(refs), np.asarray(peak_es)


# --------------------------------------------------------------------------- #
# Matching extracted <-> true spectra and error metrics                        #
# --------------------------------------------------------------------------- #
def _minmax(a):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a, -1, keepdims=True), np.nanmax(a, -1, keepdims=True)
    return np.divide(a - lo, hi - lo, out=np.zeros_like(a), where=hi != lo)


def match_to_truth(extracted, true, extracted_peaks, true_peaks):
    """Match each extracted endmember to the true phase whose peak energy is
    closest (extraction is peak-targeted, so the peak identifies the phase).
    Returns (assignment true-index per extracted, min–max RMSE per extracted).
    The RMSE is then a *quality* score of the extracted spectrum, not the
    matching criterion."""
    te = _minmax(extracted); tt = _minmax(true)
    used, assign, rmse = set(), [], []
    for i, pe in enumerate(extracted_peaks):
        d = np.array([abs(pe - tp) if j not in used else np.inf for j, tp in enumerate(true_peaks)])
        j = int(np.argmin(d)); used.add(j)
        assign.append(j)
        rmse.append(float(np.sqrt(np.mean((te[i] - tt[j]) ** 2))))
    return assign, rmse


def classify_f1(od, energies, refs, masks, assign):
    """R² map with the given references; per-phase best-F1 vs ground truth.
    ``assign[i]`` is the true phase index that extracted ref i represents."""
    scores = ptee_r2_maps(od, refs)                # (n_ref, y, x)
    f1 = {}
    for i, tphase in enumerate(assign):
        f1[tphase] = best_threshold(scores[i], masks[tphase])[0]
    return f1


# --------------------------------------------------------------------------- #
def run_condition(**kw):
    ph = build_phantom(size=kw.get("size", 256), i0_counts=kw["i0"], seed=kw["seed"],
                       minor_level=kw["minor_level"])
    n = len(ph.names)
    refs, peak_es = ptee_extract(ph.od, ph.energies, n)
    true_peaks = ph.energies[np.argmax(ph.endmembers, axis=1)]
    assign, rmse = match_to_truth(refs, ph.endmembers, peak_es, true_peaks)
    # peak-energy error vs the matched true spectrum
    peak_err = [abs(peak_es[i] - true_peaks[assign[i]]) for i in range(len(refs))]
    f1_ext = classify_f1(ph.od, ph.energies, refs, ph.masks, assign)
    f1_true = classify_f1(ph.od, ph.energies, ph.endmembers, ph.masks, list(range(n)))
    recovered = sorted(set(assign))
    return dict(i0=kw["i0"], minor_level=kw["minor_level"], seed=kw["seed"],
                n_true=n, n_extracted=len(refs), recovered_phases=recovered,
                names=ph.names, area_pct=[100 * float(a) for a in ph.area_fraction],
                assign=assign, rmse=rmse, peak_err=peak_err,
                f1_extracted={ph.names[k]: f1_ext.get(k) for k in range(n)},
                f1_true={ph.names[k]: f1_true[k] for k in range(n)})


def _phase_arrays(cell):
    """Mean-over-seeds per-phase area, extraction RMSE and F1 (extracted/true)."""
    names = cell["runs"][0]["names"]
    P = len(names)
    area = np.mean([[r["area_pct"][k] for k in range(P)] for r in cell["runs"]], axis=0)
    rmse = np.full(P, np.nan); f1e = np.full(P, np.nan); f1t = np.full(P, np.nan)
    for k in range(P):
        rs = [r["rmse"][r["assign"].index(k)] for r in cell["runs"] if k in r["assign"]]
        fe = [r["f1_extracted"][names[k]] for r in cell["runs"] if r["f1_extracted"][names[k]] is not None]
        ft = [r["f1_true"][names[k]] for r in cell["runs"]]
        if rs: rmse[k] = np.mean(rs)
        if fe: f1e[k] = np.mean(fe)
        f1t[k] = np.mean(ft)
    return names, area, rmse, f1e, f1t


def _figure(results):
    def find(i0, lvl):
        return next(c for c in results if c["i0"] == i0 and c["minor_level"] == lvl)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    # (A) extraction RMSE vs area, across SNR (mix = 1.0)
    for i0, col in zip((150.0, 350.0, 1200.0), ("#d62728", "#1f77b4", "#2ca02c")):
        _, area, rmse, _, _ = _phase_arrays(find(i0, 1.0))
        ax[0].plot(area, rmse, "o-", color=col, label=f"i0={i0:.0f} counts")
    ax[0].set_xscale("log"); ax[0].invert_xaxis()
    ax[0].set_xlabel("true phase area (%)"); ax[0].set_ylabel("extraction error (min–max RMSE)")
    ax[0].set_title("(A) Endmember extraction error vs phase area"); ax[0].legend(fontsize=8)
    # (B) F1 with extracted vs true spectra (i0=350, mix=1.0)
    names, area, rmse, f1e, f1t = _phase_arrays(find(350.0, 1.0))
    x = np.arange(len(names))
    ax[1].plot(x, f1t, "s--", color="#333", label="F1 with true spectra")
    ax[1].plot(x, f1e, "o-", color="#1f77b4", label="F1 with extracted spectra")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{n}\n{a:.2g}%" for n, a in zip(names, area)], fontsize=7)
    ax[1].set_ylabel("detection F1"); ax[1].set_ylim(0, 1.03)
    ax[1].set_title("(B) Extract-then-classify vs known spectra (i0=350)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_extraction.png", dpi=150); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    results = []
    # Sweep signal-to-noise and mixing level (3 seeds each, report mean).
    for i0 in (150.0, 350.0, 1200.0):
        for lvl in (1.0, 0.5, 0.25):
            per = [run_condition(i0=i0, minor_level=lvl, seed=s) for s in range(3)]
            results.append(dict(i0=i0, minor_level=lvl, runs=per))
    (OUT / "extraction_metrics.json").write_text(json.dumps(results, indent=2, default=float))
    _figure(results)
    # console summary
    print(f"{'i0':>6} {'mix':>4} | recovered / true | matrix..minor5 : F1(extracted spectra) [F1(true)]")
    names = results[0]["runs"][0]["names"]
    for cell in results:
        r0 = cell["runs"][0]
        rec = np.mean([len(r["recovered_phases"]) for r in cell["runs"]])
        line = f"{cell['i0']:6.0f} {cell['minor_level']:4.2f} | {rec:3.1f}/{r0['n_true']:d}       | "
        parts = []
        for k, nm in enumerate(names):
            fe = np.mean([r["f1_extracted"][nm] for r in cell["runs"] if r["f1_extracted"][nm] is not None]) \
                if any(r["f1_extracted"][nm] is not None for r in cell["runs"]) else float("nan")
            ft = np.mean([r["f1_true"][nm] for r in cell["runs"]])
            parts.append(f"{nm}:{fe:.2f}[{ft:.2f}]")
        print(line + "  ".join(parts))
    print("saved:", OUT / "extraction_metrics.json")
    return results


if __name__ == "__main__":
    main()
