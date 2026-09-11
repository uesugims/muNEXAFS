"""SVD/PCA and nonnegative spectral factorization.

Stacks are (energy, y, x), with D = C @ S: C[pixel, component] and
S[component, energy]. NMF clips negative OD to zero; MCR-ALS and NNLS
retain signed observations and constrain the fitted factors instead.
Pixels containing any NaN are excluded from nonnegative fits and receive
NaN reconstruction and zero map scores. Infinity is an input error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


class AnalysisCancelled(RuntimeError):
    """A cooperative cancellation request; no partial fit is returned."""


def _check_cancel(cancel_check):
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelled("Analysis stopped")


def _progress(callback, fraction):
    if callback is not None:
        callback(float(fraction))


def _finite(array, name):
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values; fitting failed")


def _observations(stack):
    data = np.asarray(stack, dtype=np.float64)
    if data.ndim != 3 or not all(data.shape):
        raise ValueError("stack must be a nonempty (energy, y, x) array")
    if np.isinf(data).any():
        raise ValueError("OD contains infinity; correct invalid input before fitting")
    e, y, x = data.shape
    matrix = data.reshape(e, -1).T
    valid = np.isfinite(matrix).all(axis=1)
    if not valid.any():
        raise ValueError("No complete finite OD spectra are available for fitting")
    return data.shape, matrix[valid].copy(), valid


def _parameters(n_components, max_iter, tolerance, shape):
    if int(n_components) < 1 or int(max_iter) < 1:
        raise ValueError("Components and maximum iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Convergence tolerance must be finite and positive")
    return min(int(n_components), *shape), int(max_iter)


def _restore(shape, valid, c, s):
    e, y, x = shape
    _finite(c, "Component scores")
    _finite(s, "Component spectra")
    with np.errstate(over="raise", invalid="raise"):
        fitted = c @ s
    _finite(fitted, "Reconstruction")
    reconstructed = np.full((y * x, e), np.nan)
    reconstructed[valid] = fitted
    scores = np.zeros((y * x, c.shape[1]))
    scores[valid] = c
    return reconstructed.T.reshape(shape), scores


def _normalize_factors(c, s):
    """Bound spectra in working units, preserving C @ S."""
    scale = np.max(np.abs(s), axis=1)
    scale[scale == 0] = 1.0
    s /= scale[:, None]
    c *= scale[None, :]
    _finite(c, "Component scores")
    _finite(s, "Component spectra")


@dataclass(frozen=True, slots=True)
class MultivariateResult:
    method: str
    components: NDArray[np.float64]
    scores: NDArray[np.float64]
    reconstructed: NDArray[np.float64]
    mean_spectrum: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    rgb_map: NDArray[np.float32] | None = None
    analysis_parameters: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """PCA + k-means clustering of a stack (the MANTiS-style phase-mapping
    workflow).  ``labels`` is a per-pixel cluster index map (``-1`` = excluded
    pixel); ``mean_spectra`` holds one mean OD spectrum per cluster; ``scores``
    are the PCA scores used for the PC-space scatter."""
    method: str
    labels: NDArray[np.int64]              # (y, x), -1 for excluded pixels
    n_clusters: int
    scores: NDArray[np.float64]            # (npix, ncomp) PCA scores
    shape: tuple                           # (y, x)
    mean_spectra: NDArray[np.float64]      # (n_clusters, energy)
    cluster_sizes: NDArray[np.int64]       # (n_clusters,)
    colors: NDArray[np.float64]            # (n_clusters, 3) RGB in [0, 1]
    explained_variance: NDArray[np.float64]
    rgb_map: NDArray[np.float32] | None = None   # (y, x, 4) coloured cluster map
    analysis_parameters: dict = field(default_factory=dict)


def linear_combination_map(scores, shape, coefficients, iterations=1, *,
                           cancel_check=None, progress_callback=None):
    """Combine score maps into RGBA; non-finite score pixels are transparent."""
    s = np.asarray(scores, float)
    c = np.asarray(coefficients, float)
    if s.ndim != 2 or c.ndim != 2 or c.shape[0] != 3:
        raise ValueError("Expected scores[pixel, component] and coefficients[3, component]")
    if s.shape[0] != int(np.prod(shape)):
        raise ValueError("Score map shape does not match image dimensions")
    _finite(c, "RGB coefficients")
    valid = np.isfinite(s).all(axis=1).reshape(shape)
    channels = []
    for channel in range(3):
        _check_cancel(cancel_check)
        n = min(s.shape[1], c.shape[1])
        with np.errstate(over="raise", invalid="raise"):
            value = (np.where(np.isfinite(s[:, :n]), s[:, :n], 0) @ c[channel, :n]).reshape(shape)
        for _ in range(max(1, int(iterations)) - 1):
            _check_cancel(cancel_check)
            mean = float(np.mean(value[valid])) if valid.any() else 0.0
            value = 0.5 * value + 0.5 * mean
        finite = value[valid]
        if finite.size:
            lo, hi = np.percentile(finite, (1, 99))
            mapped = np.clip((value - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros(shape)
            mapped[~valid] = 0
        else:
            mapped = np.zeros(shape)
        channels.append(mapped)
        _progress(progress_callback, (channel + 1) / 3)
    rgb = np.stack(channels, axis=-1).astype(np.float32)
    return np.concatenate([rgb, np.max(rgb, axis=-1, keepdims=True)], axis=-1)


def nmf_reconstruct(stack, n_components=3, max_iter=300, cancel_check=None, *,
                    tolerance=1e-6, progress_callback=None):
    """Factor max(OD, 0) by scaled nonnegative multiplicative updates.

    Returns reconstruction, scores C and spectra S in original OD units.
    C @ S equals reconstruction at valid pixels. Clipping negative OD adds
    no artificial baseline. Non-finite iteration results raise an error.
    """
    _check_cancel(cancel_check)
    shape, d, valid = _observations(stack)
    k, max_iter = _parameters(n_components, max_iter, tolerance, d.shape)
    np.maximum(d, 0, out=d)
    scale = float(d.max())
    if scale == 0:
        raise ValueError("NMF requires positive OD; no signal remains after clipping negative OD")
    d /= scale
    rng = np.random.default_rng(0)
    amplitude = np.sqrt(float(d.mean()) / k)
    c = (rng.random((len(d), k)) + 0.1) * amplitude
    s = (rng.random((k, d.shape[1])) + 0.1) * amplitude
    previous = np.inf
    norm = np.linalg.norm(d)
    _progress(progress_callback, 0)
    for iteration in range(max_iter):
        _check_cancel(cancel_check)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            c *= (d @ s.T) / np.maximum(c @ (s @ s.T), 1e-30)
            _check_cancel(cancel_check)
            s *= (c.T @ d) / np.maximum((c.T @ c) @ s, 1e-30)
            _normalize_factors(c, s)
        _progress(progress_callback, (iteration + 1) / max_iter)
        if iteration % 5 == 4 or iteration + 1 == max_iter:
            error = float(np.linalg.norm(d - c @ s) / norm)
            if np.isfinite(previous) and 0 <= previous - error <= tolerance * max(previous, 1e-15):
                break
            previous = error
    _check_cancel(cancel_check)
    with np.errstate(over="raise", invalid="raise"):
        s *= scale
    reconstructed, scores = _restore(shape, valid, c, s)
    _progress(progress_callback, 1)
    return reconstructed, scores, s


def _nnls_batch(basis, targets, cancel_check=None, progress_callback=None):
    """Lawson-Hanson active sets grouped across bounded batches of spectra.

    The least-squares subproblems use only their active variables. This is
    NNLS, not clipping an unconstrained solution. Small Gram systems share
    factorizations across rows with the same active set.
    """
    _check_cancel(cancel_check)
    k = basis.shape[1]
    gram = basis.T @ basis
    out = np.zeros((len(targets), k))
    for start in range(0, len(targets), 1024):
        _check_cancel(cancel_check)
        stop = min(start + 1024, len(targets))
        rows = targets[start:stop]
        cross = rows @ basis
        x = np.zeros_like(cross)
        passive = np.zeros_like(cross, dtype=bool)
        tol = 1e-11 * np.maximum(np.max(np.abs(cross), axis=1), 1e-15)
        for _ in range(max(3 * k, 10)):
            _check_cancel(cancel_check)
            dual = cross - x @ gram
            eligible = (~passive) & (dual > tol[:, None])
            unresolved = eligible.any(axis=1)
            if not unresolved.any():
                break
            row_ids = np.flatnonzero(unresolved)
            best = np.argmax(np.where(eligible[row_ids], dual[row_ids], -np.inf), axis=1)
            passive[row_ids, best] = True
            for _ in range(max(3 * k, 10)):
                _check_cancel(cancel_check)
                z = np.zeros_like(x)
                patterns, membership = np.unique(passive, axis=0, return_inverse=True)
                for group, pattern in enumerate(patterns):
                    _check_cancel(cancel_check)
                    cols = np.flatnonzero(pattern)
                    if not len(cols):
                        continue
                    selected = np.flatnonzero(membership == group)
                    subgram = gram[np.ix_(cols, cols)]
                    if np.linalg.cond(subgram) > 1e8:
                        # Normal equations square the condition number.
                        # Solve the original rectangular system for nearly
                        # collinear spectra instead of losing weak features.
                        solved = np.linalg.lstsq(basis[:, cols], rows[selected].T, rcond=None)[0].T
                    else:
                        solved = np.linalg.lstsq(
                            subgram, cross[np.ix_(selected, cols)].T, rcond=None,
                        )[0].T
                    z[np.ix_(selected, cols)] = solved
                bad = passive & (z <= 0)
                needs_step = bad.any(axis=1)
                if not needs_step.any():
                    x = z
                    break
                alpha = np.ones(len(rows))
                ratios = np.full_like(x, np.inf)
                np.divide(x, x - z, out=ratios, where=bad & (x != z))
                ratios[bad & (x == z)] = 0
                alpha[needs_step] = np.min(ratios[needs_step], axis=1)
                x += alpha[:, None] * (z - x)
                passive[(x <= 1e-14) & passive] = False
                x[~passive] = 0
            else:
                break
        dual = cross - x @ gram
        unresolved = (
            np.any((~passive) & (dual > tol[:, None]), axis=1)
            | np.any(passive & (np.abs(dual) > 10 * tol[:, None]), axis=1)
            | np.any(x < 0, axis=1)
        )
        # Rare singular active sets use bounded scalar NNLS. Check cancel
        # between rows, and never return an uninitialized/partial array.
        for row in np.flatnonzero(unresolved):
            _check_cancel(cancel_check)
            x[row] = nnls(basis, rows[row], maxiter=max(30, 5 * k))[0]
        _finite(x, "NNLS scores")
        out[start:stop] = x
        _progress(progress_callback, stop / len(targets))
    _check_cancel(cancel_check)
    return out


def nnls_reconstruct(stack, basis, cancel_check=None, *, return_scores=False,
                     progress_callback=None):
    """Fit basis[energy, component] with nonnegative coefficients.

    Observations and basis may be signed; only scores are constrained.
    return_scores=True returns (reconstruction, fitted pixel scores).
    """
    _check_cancel(cancel_check)
    shape, d, valid = _observations(stack)
    b = np.asarray(basis, float)
    if b.ndim != 2 or b.shape[0] != shape[0] or b.shape[1] < 1:
        raise ValueError("NNLS basis must have shape (energy, component)")
    _finite(b, "NNLS basis")
    basis_scale = float(np.max(np.abs(b)))
    if basis_scale == 0:
        raise ValueError("NNLS basis contains no nonzero spectra")
    data_scale = float(np.max(np.abs(d)))
    if data_scale == 0:
        c = np.zeros((len(d), b.shape[1]))
    else:
        c = _nnls_batch(b / basis_scale, d / data_scale, cancel_check, progress_callback)
        with np.errstate(over="raise", invalid="raise"):
            c *= data_scale / basis_scale
    reconstructed, scores = _restore(shape, valid, c, b.T)
    _check_cancel(cancel_check)
    _progress(progress_callback, 1)
    return (reconstructed, scores) if return_scores else reconstructed


def mcr_als_reconstruct(stack, n_components=3, max_iter=50, tolerance=1e-5,
                        nonnegative=True, closure=False, cancel_check=None,
                        progress_callback=None):
    """Alternate actual least-squares fits of concentrations and spectra.

    Nonnegative mode uses NNLS for both factors and retains signed OD.
    Closure normalizes concentration rows to one before refitting spectra.
    A short deterministic NMF fit initializes the nonnegative local fit.
    """
    _check_cancel(cancel_check)
    shape, d, valid = _observations(stack)
    k, max_iter = _parameters(n_components, max_iter, tolerance, d.shape)
    scale = float(np.max(np.abs(d)))
    if scale == 0 or (nonnegative and not np.any(d > 0)):
        raise ValueError("MCR-ALS requires nonzero signal with positive OD for nonnegative fitting")
    d /= scale
    _progress(progress_callback, 0)
    if nonnegative:
        _, c, s = nmf_reconstruct(
            d.T.reshape((shape[0], 1, len(d))), k, 30, cancel_check,
            progress_callback=lambda f: _progress(progress_callback, 0.1 * f),
        )
    else:
        u, singular, vt = np.linalg.svd(d, full_matrices=False)
        _check_cancel(cancel_check)
        c, s = u[:, :k] * singular[:k], vt[:k].copy()
    previous = np.inf
    norm = np.linalg.norm(d)
    for iteration in range(1, max_iter + 1):
        _check_cancel(cancel_check)
        if nonnegative:
            c = _nnls_batch(s.T, d, cancel_check, lambda f: _progress(
                progress_callback, 0.1 + 0.9 * ((iteration - 1 + 0.5 * f) / max_iter)))
        else:
            c = np.linalg.lstsq(s.T, d.T, rcond=None)[0].T
        if closure:
            total = c.sum(axis=1, keepdims=True)
            if not nonnegative and np.any(np.abs(total) < 1e-12):
                raise ValueError("Closure is undefined for a concentration row with zero signed sum")
            np.divide(c, total, out=c, where=total != 0)
        _check_cancel(cancel_check)
        if nonnegative:
            s = _nnls_batch(c, d.T, cancel_check, lambda f: _progress(
                progress_callback, 0.1 + 0.9 * ((iteration - 0.5 + 0.5 * f) / max_iter))).T
        else:
            s = np.linalg.lstsq(c, d, rcond=None)[0]
        if not closure:
            _normalize_factors(c, s)
        error = float(np.linalg.norm(d - c @ s) / norm)
        _finite(np.asarray(error), "MCR-ALS residual")
        _progress(progress_callback, 0.1 + 0.9 * iteration / max_iter)
        if np.isfinite(previous) and abs(previous - error) <= tolerance * max(previous, 1e-15):
            break
        previous = error
    _check_cancel(cancel_check)
    with np.errstate(over="raise", invalid="raise"):
        s *= scale
    reconstructed, scores = _restore(shape, valid, c, s)
    _progress(progress_callback, 1)
    return reconstructed, scores, s, iteration


def decompose_stack(stack, n_components=3, method="PCA", *, cancel_check=None,
                    progress_callback=None):
    _check_cancel(cancel_check)
    data = np.asarray(stack, float)
    if data.ndim != 3 or not all(data.shape):
        raise ValueError("stack must be (energy,y,x)")
    if np.isinf(data).any():
        raise ValueError("OD contains infinity")
    e, y, x = data.shape
    matrix = data.reshape(e, -1).T
    if not np.isfinite(matrix).any(axis=0).all():
        raise ValueError("Every energy must contain at least one finite OD value")
    mean = np.nanmean(matrix, axis=0)
    filled = np.where(np.isfinite(matrix), matrix, mean)
    use_centering = method != "SVD (uncentered)"
    work = filled - mean if use_centering else filled
    _progress(progress_callback, 0.1)
    _check_cancel(cancel_check)
    u, singular, vt = np.linalg.svd(work, full_matrices=False)
    _check_cancel(cancel_check)
    k = max(1, min(int(n_components), len(singular)))
    scores = u[:, :k] * singular[:k]
    recon = scores @ vt[:k] + (mean if use_centering else 0.0)
    variance = (singular * singular) / max(1, len(matrix) - 1)
    explained = variance / variance.sum() if variance.sum() else variance
    _progress(progress_callback, 1)
    return MultivariateResult(method, vt[:k], scores, recon.T.reshape(data.shape),
                              mean if use_centering else np.zeros_like(mean), explained[:k])


def cluster_palette(k):
    """``k`` distinct RGB colours in [0, 1] (evenly spaced hues)."""
    import colorsys
    k = max(1, int(k))
    return np.array([colorsys.hsv_to_rgb((0.58 + i / k) % 1.0, 0.62, 0.95) for i in range(k)], float)


def cluster_labels_to_rgb(labels, colors):
    """(y, x) int cluster labels (``-1`` excluded) + (k, 3) colours -> (y, x, 4)
    RGBA float32; excluded pixels are transparent."""
    labels = np.asarray(labels)
    rgb = np.zeros((*labels.shape, 4), np.float32)
    for c in range(len(colors)):
        sel = labels == c
        rgb[sel, :3] = colors[c]
        rgb[sel, 3] = 1.0
    return rgb


def cluster_stack(stack, n_components=5, n_clusters=6, *, seed=0,
                  cancel_check=None, progress_callback=None):
    """PCA + k-means clustering of a stack — the realistic variance-based phase
    mapping workflow (Lerotic 2004 / MANTiS).

    Runs PCA (``decompose_stack``), z-scores the retained scores column-wise,
    and clusters the pixels with k-means (scipy ``kmeans2``).  Returns a
    ``ClusterResult`` with the per-pixel cluster label map, per-cluster mean OD
    spectra, the PCA scores (for a PC-space scatter) and a coloured RGBA map.
    Pixels with any non-finite OD are excluded from clustering (label ``-1``).
    """
    from scipy.cluster.vq import kmeans2
    _check_cancel(cancel_check)
    data = np.asarray(stack, float)
    if data.ndim != 3 or not all(data.shape):
        raise ValueError("stack must be (energy,y,x)")
    if np.isinf(data).any():
        raise ValueError("OD contains infinity")
    e, y, x = data.shape
    k = max(2, int(n_clusters))
    pca = decompose_stack(data, n_components, "PCA", cancel_check=cancel_check,
                          progress_callback=lambda f: _progress(progress_callback, 0.6 * f))
    _check_cancel(cancel_check)
    scores = pca.scores                                       # (npix, ncomp)
    matrix = data.reshape(e, -1).T                            # (npix, energy)
    valid = np.isfinite(matrix).all(axis=1)
    if not valid.any():
        raise ValueError("No complete finite OD spectra are available for clustering")
    feats = scores[valid]
    std = feats.std(0, keepdims=True); std[std == 0] = 1.0
    fz = feats / std
    kk = int(min(k, max(1, fz.shape[0])))
    try:
        _, lab = kmeans2(fz, kk, seed=seed, minit="++", missing="warn")
    except Exception:
        _, lab = kmeans2(fz, kk, seed=seed, minit="random")
    labels_flat = np.full(scores.shape[0], -1, dtype=np.int64)
    labels_flat[valid] = lab
    labels = labels_flat.reshape(y, x)
    _progress(progress_callback, 0.9)
    means = np.full((k, e), np.nan)
    sizes = np.zeros(k, dtype=np.int64)
    for c in range(k):
        sel = labels_flat == c
        sizes[c] = int(sel.sum())
        if sizes[c]:
            means[c] = np.nanmean(matrix[sel], axis=0)
    colors = cluster_palette(k)
    rgb = cluster_labels_to_rgb(labels, colors)
    _progress(progress_callback, 1.0)
    params = dict(method="PCA + k-means", n_components=int(scores.shape[1]),
                  n_clusters=int(k), seed=int(seed))
    return ClusterResult("PCA + k-means", labels, int(k), scores, (int(y), int(x)),
                         means, sizes, colors, pca.explained_variance, rgb, params)
