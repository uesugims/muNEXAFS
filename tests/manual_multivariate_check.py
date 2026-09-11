"""Read-only multivariate check using a real scan and its saved I0.

Run with the project's venv, for example::

    .venv/bin/python tests/manual_multivariate_check.py /path/scan.hdr --method all

This does not instantiate MainWindow, recompute or save session state, or write
any output files.  The sidecar digest is checked again after processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np

from muaxis.io.stxm import read_stxm_scan
from muaxis.processing.absorbance import compute_optical_density
from muaxis.processing.multivariate import (
    linear_combination_map,
    mcr_als_reconstruct,
    nmf_reconstruct,
    nnls_reconstruct,
)


def load_saved_input(path: Path):
    """Load saved OD (or saved Pre-map reconstruction) without GUI side effects.

    MultivariateWindow receives the Pre-map reconstruction when available,
    otherwise the OD stack.  Registered OD is a separate display layer and is
    not the source selected by MainWindow.open_multivariate_window.
    """
    sidecar = path.with_suffix(".muaxis.json")
    state = json.loads(sidecar.read_text(encoding="utf-8"))
    scan = read_stxm_scan(path)
    od_state = state.get("od", {})
    if not isinstance(od_state.get("i0"), list):
        raise ValueError("The saved session has no I0; cannot check OD analysis")
    stack = compute_optical_density(
        scan.transmission, np.asarray(od_state["i0"], dtype=float)
    ).optical_density
    source = "OD from saved I0"
    saved_reconstruction = state.get("premap", {}).get("denoised_stack")
    if isinstance(saved_reconstruction, list):
        stack = np.asarray(saved_reconstruction, dtype=float)
        source = "Saved Pre-map low-rank reconstruction"
    return scan.energies_eV, stack, source


def array_stats(array):
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    return {
        "shape": list(values.shape),
        "finite": int(finite.size),
        "size": int(values.size),
        "negative": int(np.count_nonzero(finite < 0)),
        "nonzero": int(np.count_nonzero(finite)),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdr", type=Path)
    parser.add_argument("--method", choices=("nmf", "nnls", "mcr", "all", "stats"), default="stats")
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--nmf-iterations", type=int, default=300)
    parser.add_argument("--mcr-iterations", type=int, default=50)
    parser.add_argument("--map-iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    sidecar = args.hdr.with_suffix(".muaxis.json")
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    _, stack, source = load_saved_input(args.hdr)
    print(json.dumps({"source": source, "input": array_stats(stack)}), flush=True)
    methods = ("nmf", "nnls", "mcr") if args.method == "all" else (args.method,)
    try:
        for method in methods:
            if method == "stats":
                continue
            started = time.monotonic()
            cancelled = lambda: time.monotonic() - started >= args.timeout
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                if method in {"nmf", "nnls"}:
                    reconstructed, scores, components = nmf_reconstruct(
                        stack, args.components, max_iter=args.nmf_iterations,
                        cancel_check=cancelled,
                    )
                    if method == "nnls":
                        reconstructed, scores = nnls_reconstruct(
                            stack, components.T, cancel_check=cancelled,
                            return_scores=True,
                        )
                else:
                    reconstructed, scores, components, _ = mcr_als_reconstruct(
                        stack, args.components, max_iter=args.mcr_iterations,
                        cancel_check=cancelled,
                    )
                rgb = linear_combination_map(
                    scores, stack.shape[1:], np.eye(3, args.components),
                    args.map_iterations,
                )
            valid_pixels = np.isfinite(stack).all(axis=0)
            expected = np.maximum(stack[:, valid_pixels], 0.0)
            actual = reconstructed[:, valid_pixels]
            error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            output = {
                "method": method,
                "seconds": round(time.monotonic() - started, 3),
                "timeout_requested": cancelled(),
                "components": array_stats(components),
                "scores": array_stats(scores),
                "reconstruction": array_stats(reconstructed),
                "excluded_incomplete_spectra": int(np.count_nonzero(~valid_pixels)),
                "relative_error_to_clipped_OD": float(error),
                "map": array_stats(rgb),
            }
            print(json.dumps(output), flush=True)
            assert np.isfinite(actual).all(), "Non-finite reconstruction at valid pixels"
            assert np.isfinite(components).all(), "Non-finite component spectra"
            assert np.any(components > 0), "All component spectra are zero"
            assert np.any(rgb[..., :3] > 0), "RGB map is empty"
    finally:
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == digest, "Sidecar changed"
        print("Original sidecar unchanged", flush=True)


if __name__ == "__main__":
    main()
