# muNEXAFS — STXM-NEXAFS data-analysis application

*日本語版は [README_jp.md](README_jp.md) にあります.*

A PySide6 desktop application that reads UVSOR scanning transmission X-ray
microscopy (STXM) / NEXAFS stack data (`.hdr` + `.xim`) and carries it through
optical-density (OD) conversion, drift registration, pre-map generation,
segmentation, spectral fitting, and multivariate analysis (SVD/PCA) in one
consistent workflow.

The numerical layer (`muaxis.processing`) is independent of the GUI, so the same
algorithms can be reused from batch scripts and notebooks.

The distribution and application name is `muNEXAFS`. The Python import name
remains `muaxis` to keep compatibility with existing scripts and saved analyses.

---

## Table of contents

- [Setup](#setup)
- [Running](#running)
- [Data format](#data-format)
- [Overall workflow](#overall-workflow)
- [Main-window basics](#main-window-basics)
- [Methods and step-by-step operation for each process](#methods-and-step-by-step-operation-for-each-process)
  - [1. Optical-density (OD) conversion](#1-optical-density-od-conversion)
  - [2. Registration (drift correction)](#2-registration-drift-correction)
  - [3. Pre-map generation](#3-pre-map-generation)
  - [4. Segmentation](#4-segmentation)
  - [5. PTEE(-R²) spectral fitting](#5-pteer-spectral-fitting)
  - [6. PCA + clustering / SVD (multivariate analysis)](#6-pca--clustering--svd-multivariate-analysis)
- [Saving and restoring sessions](#saving-and-restoring-sessions)
- [Project layout](#project-layout)

---

## Setup

The `Makefile` creates a project-local virtual environment (`.venv`) and installs
the GUI, analysis, and test dependencies.

```bash
make setup
```

Main dependencies installed:

| Purpose | Package |
|---|---|
| Numerics | numpy |
| GUI | PySide6, pyqtgraph |
| Analysis | scipy, scikit-image |
| Testing | pytest |

## Running

```bash
# Launch the GUI
make run

# Open the GUI on a specific scan
make run ARGS='--open /path/to/UV_210902001.hdr'

# Read a header and print a summary (validation without opening the GUI)
make run ARGS='--inspect /path/to/UV_210902001.hdr'

# Run the tests
make test
```

`--inspect` prints the number of frames, energy range, stack shape, and whether
I0 / drift are present, so you can confirm the reader opens the target data
correctly.

## Data format

One scan is stored as the following files in a single directory.

| File | Contents |
|---|---|
| `<stem>.hdr` | Header: label, scan type, energy axis (StackAxis), spatial axes (PAxis/QAxis), and per-frame metadata |
| `<stem>_aNNN.xim` | Transmission-intensity image for each energy frame (tab-separated). Frame numbers are matched through the header's image records |
| `i0.txt` (optional) | Incident intensity I0 (per energy) |
| `drift.txt` (optional) | Pre-computed drift amounts (x, y per frame) |

The transmission stack is held internally in `(energy, y, x)` order. The reader
**does not automatically apply source-current normalization or drift
correction**: these are analysis decisions and are performed explicitly in the
processing layer.

---

## Overall workflow

```
Open a scan
   └─▶ Select a spectrum ROI / move the energy cursor (main window)
        └─▶ [OD] compute OD from a direct-beam ROI
             ├─▶ [Registration] drift correction (optional)
             │        └─▶ applying it also builds the pre-map automatically
             ├─▶ [Pre-map] net-absorption / peak maps (open only to change ranges)
             │        └─▶ [Segmentation] threshold clustering + cluster mean spectra
             │                 └─▶ [PTEE(-R²)] R² mapping against reference spectra
             └─▶ [PCA + clustering / SVD] multivariate decomposition and maps
```

Each process is launched from the `Process` menu, and its result is added to the
**layer list** on the left of the main window. Analysis state is saved
per-dataset automatically and restored the next time the same scan is opened
(see [Saving and restoring sessions](#saving-and-restoring-sessions)).

Menu entries whose upstream analysis is not yet done are disabled (for example,
Segmentation / PTEE(-R²) cannot be chosen until OD has been computed).

---

## Main-window basics

- **Open a scan**: `File ▸ Open scan…` (Ctrl+O). Previously opened scans can be
  reselected from the list on the left.
- **Layer list**: switches what is shown in the central image. In addition to the
  base `Transmission (raw)` / `Optical density (OD)`, each process adds its
  result (`Registered OD`, `Net absorption map (OD)`, `Peak map (OD)`,
  `OD - pre-edge (OD)`, `PTEE RGBY map`, `PCA cluster map`, …).
- **Frame navigation**: there is no frame slider. Type a frame number in the
  **Frame** box and press Enter to jump to it, or **click / drag anywhere on the
  spectrum plot** to move the yellow energy cursor — the nearest frame is
  selected and the box and energy label update accordingly. The energy label
  next to the box shows the current frame's energy.
- **Spectrum profile**: with **no ROI selected, the whole-image mean spectrum**
  (transmission or OD) is shown as a dashed grey curve. As soon as one or more
  spectrum ROIs exist, the whole-image mean is hidden and replaced by the
  per-ROI mean spectra. This is mirrored in the report output.
- **Spectrum ROI**: `ROI ▸ Select spectrum ROI` places a rectangular ROI whose
  mean spectrum is plotted. Multiple ROIs can be placed.
- **Spectrum normalization**: the combo box below the plot selects `None` /
  `Each plot min–max` / `Two energy values` (normalize at two energies E1, E2).
- **Map display**: when a pre-map is shown, the palette (Jet / Fire / Grayscale)
  and the levels in the right-hand histogram can be adjusted.

---

## Methods and step-by-step operation for each process

### 1. Optical-density (OD) conversion

**Method**

From the transmission image I and incident intensity I0, the optical density
(absorbance) is computed per pixel,

```
OD = log(I0 / I)
```

(`muaxis.processing.absorbance.compute_optical_density`). I0 is taken as the
**mean transmission of a direct-beam (bright, sample-free) ROI** at each energy.
Non-finite pixels or pixels ≤ 0 are marked invalid (NaN).

Optional storage-ring-current correction (scaling by
`current_reference / current`) is available; it is opt-in, and zero or non-finite
current values are rejected explicitly as errors.

**Operation**

1. Open `Process ▸ OD window`.
2. Press "Select direct-beam ROI" and place the yellow rectangle over a bright,
   sample-free region (draggable and resizable).
3. I0 is derived from the ROI automatically and OD is computed. Switch between
   the "Optical density (OD)" and "Direct transmission" radio buttons.
4. Move the slider to inspect frames (energies).
5. The computed OD is reflected in the main window, `Optical density (OD)` is
   added to the layer list, and the downstream processes (Registration /
   Pre-map / Segmentation / PTEE(-R²)) are enabled.

> Clearing the ROI with "Clear ROI" discards the OD.

---

### 2. Registration (drift correction)

**Method**

Sample drift during the energy sweep is corrected as a frame-to-frame
translation (`muaxis.processing.registration.register_translation_stack`).

- **Phase correlation** (scikit-image `phase_cross_correlation`) estimates each
  frame's subpixel shift relative to a reference frame. `upsample_factor` raises
  the resolution.
- The estimated shifts are applied with `scipy.ndimage.shift` (linear
  interpolation; out-of-range filled with NaN).
- Two **reference modes**:
  - `Fixed`: always use one chosen frame as the reference.
  - `Previous`: track sequentially against the previous corrected frame.
- NaNs are filled with the frame median only during shift estimation; the
  returned corrected stack keeps NaNs.

**Operation**

1. Open `Process ▸ Registration` (the OD stack is the input if OD has been
   computed, otherwise the transmission stack).
2. Set the reference mode (Fixed / Previous), reference-frame number, and
   subpixel resolution (upsample factor, default 10).
3. Press "Run registration" (runs on a separate thread). When it finishes, the
   per-frame Shift X / Shift Y (px) appear in the table.
4. **Preview the result** in the image view below: scroll through the registered
   frames with the slider, and toggle **"Show original (compare)"** to compare
   the input and corrected frames before committing.
5. Press "Apply registered stack" when the result looks correct. It is reflected
   in the main window, `Registered OD` (or `Registered transmission`) is added to
   the layer list, **and the pre-map is (re)built automatically** from the
   aligned stack with default energy ranges (see the next section).

> Closing the window after Run without pressing Apply still keeps (and applies)
> the computed registration.

---

### 3. Pre-map generation

**Method**

Generates the 2-D maps that feed segmentation (`muaxis.processing.premap`).

- **Net-absorption map** (`net_absorption_map`): per pixel, after **subtracting
  the pre-edge OD average (the non-resonant baseline ≈ the thickness term)**,
  the post-edge frames are **summed** (identical to the original `diff_sum`; a
  per-frame sum, not a trapezoidal integral, so the contrast matches). Negative
  net absorption (noise / edge effects) is clipped to 0. Removing the baseline
  cancels the thickness dependence and leaves the chemical (resonant) contrast,
  so pixels with the same chemistry but different thickness group into the same
  cluster — the right input for segmentation.
- **Peak map** (`peak_map`): the **energy position at which each pixel reaches
  its maximum intensity** within the post-edge range (it encodes the peak
  *position*, not the maximum intensity).
- **OD − pre-edge stack** (`pre_edge_subtract`): a 3-D stack with the pre-edge
  average subtracted from every frame and negatives clipped to 0. It is added to
  the main layer list as `OD - pre-edge (…)` and can be inspected per energy.
  Segmentation uses this stored stack directly as its composition-based
  clustering input (the session stores only the pre-edge range and rebuilds the
  stack on restore).
- Optionally, **SVD/PCA low-rank reconstruction denoising** can be applied as a
  pre-processing step (drops the lower principal components to reduce noise; uses
  the same decomposition as [PCA + clustering / SVD](#6-pca--clustering--svd-multivariate-analysis)).

**When it runs.** Applying a registration builds the pre-map automatically with
default ranges (pre-edge = the first fifth of the scan, post-edge = the rest).
**Open the Pre-map window only when you want to change those ranges** (or enable
denoising); pressing its button re-creates the maps.

**Operation (to modify)**

1. Open `Process ▸ Pre-map`.
2. Enter the pre-edge range (Pre-edge start / end) and post-edge range
   (Post-edge start / end) so they bracket the absorption edge.
3. Optionally enable denoising and set the method and number of components.
4. Press "Create net absorption + peak maps".
5. `Net absorption map (…)`, `Peak map (…)`, and `OD - pre-edge (…)` are added to
   the layer list. The 2-D maps allow palette / level adjustment, and
   `OD - pre-edge` can be shown per energy.

---

### 4. Segmentation

**Method**

Thresholds the input map and computes an OD mean spectrum for each connected
region (cluster) (`muaxis.processing.segmentation.segment_clusters`).

- **Threshold mask**: selects finite pixels with `low ≤ value ≤ high`.
- **Clustering**: 4-connectivity labelling (`scipy.ndimage.label`).
- **Minimum size**: clusters below a pixel count are dropped and the rest
  renumbered 1..N.
- **Mean spectrum**: `nanmean` of the OD over each cluster's pixels per energy.

The GUI previews the selected region with a histogram and overlay, tracks each
cluster's area and a "Good" flag, and plots the mean spectrum of the selected
clusters (and their average when several are selected). Profile normalization is
`Min–max` / `None` / `Two energy values`.

Besides the net-absorption and peak maps, single-energy images can be chosen as
the input layer:

- **OD**: the raw OD image.
- **OD − pre-edge**: each frame with the pre-edge average (the non-resonant
  baseline, e.g. thickness) subtracted and negatives clipped to 0
  (`max(OD − pre-edge mean, 0)`, identical to the original). Prefer this for
  composition-based clustering (the pre-edge range reuses the Pre-map setting, or
  defaults to the first few frames if no pre-map exists).

Choosing these shows an **OD energy slider** to pick which energy frame to
cluster on. Note that the cluster mean spectra (profiles) are always computed
from the **raw OD**, whatever input is chosen (identical to the original).

**Setting thresholds.** There are no threshold sliders. Set the lower and upper
thresholds either by **dragging on the histogram** — the threshold *nearer* the
cursor moves to it — or by typing values into the two boxes labelled **Lower /
Upper** next to the "Log Y" checkbox. Dragging updates the boxes and vice versa;
the red overlay and the selected-area read-out update live.

**Operation**

1. With the layer you want as input selected, open `Process ▸ Segmentation`.
   Switch the input with "Input layer" at the top; if you choose `OD`, pick the
   energy frame with the adjacent "OD energy" slider.
2. Set the thresholds by dragging the histogram or typing into the Lower / Upper
   boxes (confirm with the red overlay and the area read-out).
3. Set "Minimum size" for the smallest cluster, and switch the input layer if
   needed.
4. Press "Run" to cluster; the table on the right lists the clusters and the
   plot below shows their mean OD spectra.
5. Select clusters in the table (multi-select allowed) to highlight them; when
   several are selected their average spectrum (Cluster 0) is also drawn. Use the
   "Good" column and "Exclude bad data" to hide clusters judged poor.
6. Enter a label and press "Save" to store the selected profiles as a
   segmentation result in the main window; they become reference candidates for
   spectral fitting.

> Each time you press "Run" with new thresholds or a new layer, the clusters are
> recomputed from scratch (the previous "Good" filter is reset on a new run).

---

### 5. PTEE(-R²) spectral fitting

**Method**

Peak-targeted endmember extraction with R² RGBY mapping: each pixel's OD spectrum
is compared with reference spectra (e.g. the mean profiles saved in
Segmentation), and the agreement is scored with the coefficient of determination
R² and rendered as a colour map (`muaxis.processing.fitting.spectral_r2_map`).

- Up to four references are assigned to the R / G / B / Y channels.
- Per pixel, the standard coefficient of determination
  `R² = 1 − Σ(y−r)² / Σ(y−ȳ)²` is computed from the residual and total sums of
  squares (a fixed-reference R², which can be negative; it is not the squared
  Pearson correlation of a fitted line).
- **Classifier** (combo): the per-pixel matching rule against the references.
  `R²` (default, above) is the lightest and most interpretable; `SAM` scores by
  spectral-angle cosine similarity; `LCF` by non-negative linear-combination
  fractional abundance (a fast OLS + clip approximation). SAM and LCF reduce the
  over-detection of very faint phases; because LCF scores are fractional
  abundances, lower the score floor when using it. R², SAM and LCF share the
  floor, single-phase and RGBY logic (higher score = better match).
- A **score floor** clips `[floor, 1]` to `[0, 1]` as each channel's weight. The Y
  channel is added to R and G to form the displayed colour (premultiplied RGBA).
- **Single-phase assignment** (checkbox): by default a pixel can show a blend of
  channels (a mixture). Enable this to assign each pixel to the single
  highest-scoring reference (winner-take-all), so the map shows one phase per
  pixel; a pixel whose best match is below the floor stays unassigned.
- A Gaussian blur can optionally be applied to the input stack before scoring.
- **Profile normalization** has four modes. Modes 2–4 first subtract the pre-edge
  average from **both** pixel and reference (removing the thickness term; the
  pre-edge range reuses the Pre-map setting):
  1. **Min–max** (default): baseline and amplitude normalized → match on **shape
     only**.
  2. **Subtract pre-edge**: baseline removed, **amplitude (absorption amount)
     kept** → a semi-quantitative map that matches on **shape + amount**
     (distinguishes concentration even for the same material; robust to
     background noise).
  3. **Subtract pre-edge + Absolute max**: after pre-edge subtraction, scale by
     the absolute maximum.
  4. **Subtract pre-edge + max at energy**: after pre-edge subtraction, scale so
     the value at a **chosen energy** equals 1 (matching pixel and reference at
     that energy). The energy is chosen in the adjacent dropdown.

**Operation**

1. Save the reference mean spectra in Segmentation first.
2. Open `Process ▸ PTEE(-R²)`.
3. Assign a saved label and one of its profiles (clusters) to each of the R / G /
   B / Y channels (set unused channels to "None"). The selected references are
   previewed at the top.
4. Choose the classifier (R² / SAM / LCF), normalization mode, score floor,
   single-phase assignment if you want one phase per pixel, and a Gaussian blur
   radius if needed.
5. Press "Run" to map. The RGBY map appears at the bottom.
6. Press "Save map" to add `PTEE RGBY map` to the main window's layer list.

---

### 6. PCA + clustering / SVD (multivariate analysis)

**Method**

Decomposes the whole stack into principal components, score maps, and a low-rank
reconstruction (`muaxis.processing.multivariate.decompose_stack`).

- The data is reshaped to an `(energy, pixel)` matrix and centered by subtracting
  the mean (NaNs filled with the mean).
- Singular value decomposition (`numpy.linalg.svd`) extracts the top k
  components. Scores = `U·S`, reconstruction = `scores · Vᵀ + mean`, explained
  variance = the ratio of singular values².
- **Linear-combination map** (`linear_combination_map`): each PC score map is
  combined with R / G / B coefficients, clipped and normalized over the 1–99th
  percentiles into an RGBA map (`iterations` can repeat a smoothing pass).
- A **PCA + k-means cluster map** can also be produced.

**Operation**

1. Open `Process ▸ PCA + clustering` (the input is the pre-map's denoised stack
   if present, otherwise the OD / transmission stack).
2. Set the method (PCA / SVD), the number of components, and iterations.
3. Set the PC1–PC3 coefficients for each of the R / G / B channels (default
   PC1→R, PC2→G, PC3→B).
4. Press "Run": the decomposition runs on a separate thread with a progress bar.
   When done, the component spectra appear at the top and the map at the bottom
   (`SVD/PCA RGB map` or `PCA cluster map` in the layer list).

---

## Saving and restoring sessions

Analysis state (the OD ROI, registration settings, pre-map, segmentation
results, fitting results, the active layer / frame / normalization, etc.) is
saved automatically as a **sidecar `<stem>.muaxis.json`** in the same directory
as the scan.

- The next time the same scan is opened, each process is recomputed to rebuild
  the layers and the previous state is restored.
- The active layer is **saved and restored by name**, so the correct layer is
  selected even if the layer order shifts while upstream processes are rebuilt
  (the old positional index is still read for backward compatibility).
- `File ▸ Save session…` can also save explicitly to any location.

---

## Project layout

```
src/muaxis/
├── io/
│   └── stxm.py            # reader for .hdr / .xim / i0.txt / drift.txt (GUI-independent)
├── processing/            # numerical layer (GUI-independent)
│   ├── absorbance.py      # OD = log(I0/I)
│   ├── registration.py    # phase-correlation drift correction
│   ├── premap.py          # net-absorption / peak maps
│   ├── segmentation.py    # threshold clustering + cluster mean spectra
│   ├── fitting.py         # R² RGBY mapping against reference spectra (PTEE)
│   └── multivariate.py    # SVD/PCA and linear-combination maps
├── gui/                   # PySide6 / pyqtgraph windows
│   ├── main_window.py     # data / display hub, layer management, sessions
│   ├── od_window.py
│   ├── registration_window.py
│   ├── premap_window.py
│   ├── segmentation_window.py
│   ├── fitting_window.py
│   └── multivariate_window.py
└── __main__.py            # CLI entry point (--inspect / --open)
```

The numerical layer has no Qt dependency, so it can also be used from scripts:

```python
from muaxis.io.stxm import read_stxm_scan
from muaxis.processing import compute_optical_density, segment_clusters

scan = read_stxm_scan("UV_210902001.hdr")
od = compute_optical_density(scan.transmission, scan.i0).optical_density
```
