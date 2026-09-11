# PTEE vs SVD/PCA — minor-phase phantom benchmark

Reusable benchmark comparing **PTEE** (Peak-Targeted Endmember Extraction, the
`spectral_r2_map` used by muNEXAFS's Spectral-fitting window) against both other
**supervised** target methods given the *same* known spectra — **SAM** (spectral
angle) and **LCF** (non-negative linear-combination fit) — and **unsupervised
SVD/PCA** (`decompose_stack`), on a controlled STXM/NEXAFS phantom, for a paper
on PTEE. Including SAM/LCF makes the test fair: PTEE has no information advantage
over them, so the comparison isolates the matching rule and the cost rather than
"PTEE knew the answer". It quantifies two claims:

1. **Computation speed** — PTEE vs a full SVD/PCA decomposition, and how each
   scales with pixel count.
2. **Minor-phase detection** — how detection quality degrades as a phase's area
   fraction shrinks.

The generated results live in [`report.md`](report.md) and `outputs/`.

## Quick start

```bash
# from the repo root (uses the project venv)
.venv/bin/python benchmarks/ptee_vs_svd/make_endmembers.py     # 1. endmember spectra
.venv/bin/python benchmarks/ptee_vs_svd/benchmark.py           # 2. run + write report
```

`benchmark.py` writes `outputs/fig1_phantom.png`, `fig2_detection_maps.png`,
`fig3_summary.png`, `metrics.json`, `metrics.csv`, and regenerates `report.md`.

Options:

```
--size N     phantom edge length in px (default 320; larger = clearer speed gap)
--i0 C       photon budget (counts); lower = noisier (default 300)
--ncomp K    SVD/PCA components an analyst keeps (default 3, the GUI default)
--seeds N    phantom realizations to average over (default 3, mean ± std)
--seed S     base RNG seed (default 0)
--seedref    estimate PTEE references from seed regions instead of the library
--threads N  BLAS/OpenMP threads for timing (default 1 = single-thread, for
             reproducible, hardware-comparable speeds; the script pins the
             thread count before numpy loads by re-exec'ing itself once)
```

Timings are recorded together with the hardware/software they were measured on
(CPU, cores, RAM, OS, Python/numpy/scipy, BLAS backend, thread count) in
`metrics.json` (`environment`) and the report's Speed section, since wall-clock
times are only meaningful with that context.

## What it does

- **Phantom** (`phantom.py`): a large OD cube built from endmember spectra with
  known ground-truth phase locations — one dominant matrix phase plus several
  **minor phases of shrinking area** (≈5 % → 0.1 %). A smooth thickness/density
  field multiplies every pixel (so total absorption ≠ composition), and Poisson
  photon noise is applied at the transmission level.
- **Supervised methods (same known spectra)** — the fair comparison:
  - **PTEE**: per-pixel min–max R² (`spectral_r2_map`); the benchmark's
    `ptee_r2_maps` reproduces that math for any number of references and is
    checked against the GUI function.
  - **SAM**: spectral-angle mapper (`sam_corr_maps`) — correlation of each pixel
    with each reference (scale/offset-invariant); the classic hyperspectral
    target-detection analog.
  - **LCF**: non-negative linear-combination fit (`lcf_frac_maps`) — per-pixel
    least-squares abundance of the known endmembers; the standard supervised
    unmixing baseline, and the one that yields real mixing fractions.
- **Unsupervised SVD/PCA (raw component)**: `decompose_stack` (centered = PCA,
  uncentered = SVD). For per-phase detection each is given its **oracle best
  component** — the single component that best separates that phase — the most
  generous unsupervised reading (in reality the phase→component mapping is
  unknown, and, as fig 2 shows, components mix all phases).
- **PCA+cluster (`kmeans_detection`)**: the realistic standard variance workflow
  (Lerotic 2004 / MANTiS) — k-means on the PCA scores, then **each cluster is
  labelled by its dominant (majority) phase** (the standard way clusters become a
  phase map), scanning a couple of `k`. This is the fair unsupervised *mapping*
  baseline (not a raw-component straw man); it keeps up on moderate minor phases
  but breaks down on the smallest ones, which k-means never gives a cluster of
  their own (so they are absorbed into the matrix cluster and missed). Obtaining
  physically non-negative component spectra on top of this needs a further
  *iterative* MCR-ALS/NMF step (not run here) — noted in the report's Speed
  section as an extra cost PTEE avoids.
- **Metrics**: detection **F1** at the best operating threshold (headline), the
  **detected area %** (vs the phase's true area — flags over-detection) and
  **det.%** (recall), per phase, averaged over several realizations; wall-clock
  time and a pixel-count scaling curve.

## Endmember spectra — two options

- `make_endmembers.py` (default): **synthetic** distinct π* resonances on the
  real UVSOR C K-edge energy grid. Distinct-by-construction, so *phase area* —
  not spectral confusion — drives the result. Recommended for the method
  comparison.
- `extract_endmembers.py`: spectra taken **straight from a real scan**
  (`/Volumes/Extreme Pro/stxm/UVSOR/211123/UV_211123007`) by clustering. Needs
  the dataset; use it for a realism cross-check.

Both write the same `data/endmembers.npz` (`energies`, `endmembers`, `names`),
which `phantom.py` consumes, so the phantom and benchmark run **without the raw
dataset** once the npz exists.

## Files

| file | role |
|---|---|
| `make_endmembers.py`  | synthesize distinct endmember spectra → `data/endmembers.npz` |
| `extract_endmembers.py` | alternative: extract endmembers from the real scan |
| `phantom.py`          | build the multi-minor-phase OD phantom (drive-free) |
| `benchmark.py`        | run PTEE / SAM / LCF vs SVD/PCA, write figures/metrics/report |
| `report.md`           | generated report (regenerated by `benchmark.py`) |
| `data/endmembers.npz` | endmember spectra + energy grid |
| `outputs/`            | figures, `metrics.json`, `metrics.csv` |
