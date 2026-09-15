"""Report snapshots and compact, restart-safe analysis provenance.

This module does not rerun any analysis or infer missing historical settings.
Pixel scores/reconstructed stacks are deliberately excluded from fit records.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

MISSING = "Not recorded"
CHANNEL_COLORS = ("#ff4040", "#40d060", "#4080ff", "#ffd020")
CLUSTER_COLORS = ("#ff5050", "#50b4ff", "#64dc64", "#ffbe3c", "#c864ff", "#32dcc8", "#ff64b4", "#a0dc50")


def result_to_record(result, kind):
    """Save maps and short spectral/condition arrays, never full pixel fits."""
    if kind != "fitting" and getattr(result, "n_clusters", None) is not None:
        return _cluster_to_record(result)
    fields = (("rgb_y", "references", "effective_references", "raw_references", "r2_floor", "normalization")
              if kind == "fitting" else
              ("rgb_map", "method", "components", "mean_spectrum", "explained_variance"))
    record = {}
    for key in (*fields, "analysis_parameters"):
        value = getattr(result, key, None)
        if value is not None:
            record[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return record


def _cluster_to_record(result):
    """Persist a PCA+k-means ClusterResult: cluster map, labels, per-cluster mean
    spectra/colours/sizes, and a subsampled PC1/PC2 scatter for the report."""
    record = {"result_type": "cluster"}
    for key in ("rgb_map", "labels", "mean_spectra", "colors", "cluster_sizes",
                "explained_variance", "n_clusters", "method"):
        value = getattr(result, key, None)
        if value is not None:
            record[key] = value.tolist() if isinstance(value, np.ndarray) else value
    ap = getattr(result, "analysis_parameters", None)
    if ap is not None:
        record["analysis_parameters"] = ap
    scores = getattr(result, "scores", None)
    labels = getattr(result, "labels", None)
    if scores is not None and labels is not None and np.asarray(scores).shape[1] >= 2:
        scores = np.asarray(scores, float); lab = np.asarray(labels).reshape(-1)
        idx = np.flatnonzero(lab >= 0)
        if idx.size > 20000:
            idx = np.random.default_rng(0).choice(idx, 20000, replace=False)
        record["scatter"] = {"x": scores[idx, 0].tolist(), "y": scores[idx, 1].tolist(),
                             "labels": lab[idx].astype(int).tolist()}
    return record


def result_from_record(record, kind):
    if kind != "fitting" and record.get("result_type") == "cluster":
        floats = {"rgb_map", "mean_spectra", "colors", "explained_variance"}
        ns = {}
        for key, value in record.items():
            if key == "labels":
                ns[key] = np.asarray(value, dtype=int)
            elif key in floats:
                ns[key] = np.asarray(value, dtype=float)
            else:
                ns[key] = value
        return SimpleNamespace(**ns)
    arrays = ({"rgb_y", "references", "effective_references", "raw_references"}
              if kind == "fitting" else {"rgb_map", "components", "mean_spectrum", "explained_variance"})
    return SimpleNamespace(**{key: np.asarray(value, dtype=float) if key in arrays else value
                              for key, value in record.items()})


def _mean(stack):
    a = np.asarray(stack, dtype=float)
    finite = np.isfinite(a)
    count = finite.reshape(len(a), -1).sum(axis=1)
    total = np.where(finite, a, 0).reshape(len(a), -1).sum(axis=1)
    return np.divide(total, count, out=np.full(len(a), np.nan), where=count > 0)


def _small(mapping):
    """Keep condition tables readable even for old dictionaries with images."""
    excluded = {"denoised_stack", "rgb_y", "rgb_map", "labels", "profiles", "saved_groups",
                "references", "effective_references", "raw_references", "components", "scores", "reconstructed"}
    out = {}
    for key, value in (mapping or {}).items():
        if key in excluded and not np.isscalar(value):
            continue
        if isinstance(value, dict):
            out[key] = _small(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=object)
            if array.size <= 60:
                out[key] = value.tolist() if isinstance(value, np.ndarray) else value
            else:
                preview = array.reshape(-1)[:12].tolist()
                out[key] = {"entries": len(value), "shape": list(array.shape),
                            "first_values": preview, "note": "Summary; spectrum/cluster identities are in CSV headers"}
        elif isinstance(value, np.generic):
            out[key] = value.item()
        elif value is None or isinstance(value, (str, bool, int, float)):
            out[key] = value
    return out


def _cluster_scatter(result):
    """(x, y, labels) of the PC1/PC2 scatter, from a live ClusterResult (scores)
    or a restored record (persisted ``scatter``).  None if unavailable."""
    scatter = getattr(result, "scatter", None)
    if isinstance(scatter, dict) and scatter.get("x"):
        return (np.asarray(scatter["x"], float), np.asarray(scatter["y"], float),
                np.asarray(scatter["labels"], int))
    scores = getattr(result, "scores", None); labels = getattr(result, "labels", None)
    if scores is not None and labels is not None and np.asarray(scores).shape[1] >= 2:
        scores = np.asarray(scores, float); lab = np.asarray(labels).reshape(-1)
        idx = np.flatnonzero(lab >= 0)
        if idx.size > 20000:
            idx = np.random.default_rng(0).choice(idx, 20000, replace=False)
        return scores[idx, 0], scores[idx, 1], lab[idx].astype(int)
    return None


def _render_cluster_scatter(x, y, labels, colors):
    """Render the PC-space scatter to a float RGBA image (matplotlib Agg)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(4.2, 4.0), dpi=150); FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    cols = np.clip(np.asarray(colors, float), 0, 1)
    pt = cols[np.clip(labels, 0, len(cols) - 1)]
    ax.scatter(x, y, s=4, c=pt, linewidths=0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PC space (coloured by cluster)")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)
    return (buf.astype(np.float32) / 255.0)


def build_report_payload(window):
    scan = window.scan
    payload = {"layers": [], "spectra": {}, "mapping_spectra": {}, "energies": np.asarray([]),
               "metadata": {}, "warnings": []}
    if scan is None:
        return payload
    energies = np.asarray(scan.energies_eV, dtype=float)
    payload["energies"] = energies
    frame = min(max(int(getattr(window, "_current_frame", 0)), 0), len(energies) - 1)
    h = scan.header
    shape = scan.shape
    def step(axis):
        return float(np.median(np.diff(axis))) if axis is not None and len(axis) > 1 else MISSING
    measurement = {
        "Header label": getattr(h, "label", None) or MISSING,
        "Scan type": getattr(h, "scan_type", None) or MISSING,
        "Acquired at": h.frames[0].acquired_at or MISSING if h.frames else MISSING,
        "Energy range (eV)": [float(np.min(energies)), float(np.max(energies))],
        "Energy frames": len(energies), "Image size (x, y pixels)": list(shape[:0:-1]) if shape else MISSING,
        "X step (um)": step(getattr(h, "x_um", None)), "Y step (um)": step(getattr(h, "y_um", None)),
        "Dwell time (ms)": getattr(h, "dwell_time_ms", None) or MISSING,
        "Report image frame (zero-based)": frame, "Report image energy (eV)": float(energies[frame]),
        "Profile values": "Raw values (no display-only normalization)",
    }
    title_edit = getattr(window, "measurement_name_edit", None)
    title = title_edit.text().strip() if title_edit is not None else ""
    metadata = {"title": title or h.path.stem, "scan": str(h.path), "shape": shape,
                "measurement_parameters": measurement, "analysis_parameters": {}}
    payload["metadata"] = metadata

    def add_spectrum(name, values, color=None):
        if values is None:
            return None
        values = np.asarray(values, dtype=float)
        if values.shape != energies.shape:
            payload["warnings"].append(f"{name}: spectrum length does not match the scan energy axis")
            return None
        payload["spectra"][name] = values.copy()
        return {"name": name, "values": values.copy(), "color": color}

    def layer(name, source, condition, image, parameters, profiles, ylabel="OD", display=None, rois=None):
        parameters = _small(parameters)
        item = {"name": name, "source": source, "condition": condition, "image": image,
                "parameters": parameters, "profiles": [p for p in profiles if p is not None],
                "profile_ylabel": ylabel, "energies": energies}
        if display:
            item["display"] = display
        if rois:
            item["rois"] = list(rois)
        payload["layers"].append(item)
        metadata["analysis_parameters"][name] = parameters

    rois = []
    if shape:
        for roi in getattr(window, "_spectrum_rois", []):
            rois.append(window._roi_bounds(roi, shape[2], shape[1]))
    def stack_profiles(name, stack):
        if stack is None:
            return []
        profiles = [add_spectrum(f"{name} — whole-image mean", _mean(stack))]
        for i, (x0, y0, x1, y1) in enumerate(rois):
            profiles.append(add_spectrum(f"{name} — ROI {i + 1}", _mean(stack[:, y0:y1, x0:x1])))
        return profiles

    transmission = scan.transmission
    common_frame = {"Frame index (zero-based)": frame, "Energy (eV)": float(energies[frame])}
    if rois:
        common_frame["ROI bounds (x0, y0, x1, y1)"] = rois
    if transmission is not None:
        layer("Transmission (raw)", "Raw XIM stack", "No transformation",
              transmission[frame], common_frame, stack_profiles("Transmission", transmission),
              "Transmission", rois=rois)
    od_result = getattr(window, "_od_result", None)
    od = getattr(od_result, "optical_density", None)
    if od is not None:
        profiles = stack_profiles("OD", od)
        add_spectrum("I0 — direct beam", getattr(od_result, "i0", None))
        layer("Optical density (OD)", "Transmission + I0", "OD = ln(I0 / I)", od[frame],
              {**common_frame, "I0 ROI": getattr(window, "_od_roi_bounds", None) or MISSING}, profiles,
              rois=rois)
    registered = getattr(window, "_registered_stack", None)
    registration = getattr(window, "_registration_config", None)
    if registration is not None:
        name = "Registered OD" if od is not None else "Registered transmission"
        layer(name, "OD" if od is not None else "Transmission", "Subpixel translation registration",
              registered[frame] if registered is not None else None,
              {**registration, **common_frame}, stack_profiles(name, registered), "OD" if od is not None else "Transmission")
    premap = getattr(window, "_premap_config", None) or {}
    denoising_settings = premap.get("multivariate", MISSING)
    if "multivariate" in premap and denoising_settings is None:
        denoising_settings = "Disabled"
    denoised = getattr(window, "_analysis_stack", None)
    source = str(premap.get("source", MISSING))
    premap_stack = denoised if denoised is not None else (od if source == "OD" else transmission)
    if denoised is not None:
        layer("Low-rank reconstructed input", source, "Noise-reduced analysis input", denoised[frame],
              {"Denoising": denoising_settings, **common_frame},
              stack_profiles("Low-rank reconstructed input", denoised))
    presub = getattr(window, "_presub_stack", None)
    if presub is not None:
        layer("OD - pre-edge", source, "Pre-edge mean subtracted; values below zero clipped", presub[frame],
              {"pre_edge_range_eV": premap.get("pre_edge_range", MISSING), **common_frame},
              stack_profiles("OD - pre-edge", presub))
    for name, image in getattr(window, "_premap_maps", {}).items():
        condition = ("Peak energy within the selected post-edge range" if "Peak" in name else
                     "Sum of post-edge OD minus pre-edge mean; not energy-weighted integration")
        parameters = {**_small(premap), "Denoising": denoising_settings}
        layer(name, source, condition, image, parameters, stack_profiles(f"{name} input", premap_stack),
              display=getattr(window, "_premap_display", {}).get(name, {"palette": "Jet"}))

    segmentation = getattr(window, "_segmentation_config", None) or {}
    for label, group in (segmentation.get("saved_groups") or {}).items():
        if not isinstance(group, dict):
            continue
        ids = group.get("cluster_ids", [])
        profiles = []
        for i, values in enumerate(group.get("profiles", [])):
            cid = ids[i] if i < len(ids) else i + 1
            color = "#303030" if cid == 0 else CLUSTER_COLORS[(int(cid) - 1) % len(CLUSTER_COLORS)]
            profiles.append(add_spectrum(f"{label} — cluster {cid}", values, color))
        labels = group.get("labels")
        image = None
        if labels is not None:
            labels = np.asarray(labels, dtype=int)
            image = np.zeros((*labels.shape, 3), dtype=np.uint8)
            for cid in np.unique(labels):
                if cid > 0:
                    color = CLUSTER_COLORS[(int(cid) - 1) % len(CLUSTER_COLORS)]
                    image[labels == cid] = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
        else:
            payload["warnings"].append(f"Segmentation {label}: saved label image was not recorded")
        parameters = dict(group.get("analysis_parameters") or {"Historical conditions": MISSING})
        parameters["Saved cluster IDs"] = ids
        parameters["Saved areas (pixels)"] = group.get("areas", [])
        layer(f"Segmentation — {label}", parameters.get("source", MISSING),
              "Saved cluster labels and raw mean OD spectra", image, parameters, profiles)

    fit = getattr(window, "_fitting_result", None)
    if fit is not None:
        params = dict(getattr(fit, "analysis_parameters", None) or {})
        if not params:
            params["Historical conditions"] = MISSING
            payload["warnings"].append("PTEE: original mapping conditions/endmembers may not have been recorded")
        effective = getattr(fit, "effective_references", None)
        raw = getattr(fit, "raw_references", None)
        profiles, used = [], {}
        for rows, suffix in ((raw, "raw reference"), (effective, "mapping endmember")):
            if rows is None:
                continue
            for i, values in enumerate(rows):
                if i >= 4 or not np.isfinite(values).any():
                    continue
                channel = "RGBY"[i]
                ch = params.get("channels", {}).get(channel, {})
                name = f"PTEE {channel} — {ch.get('label', MISSING)} / cluster {ch.get('cluster_id', MISSING)} — {suffix}"
                profile = add_spectrum(name, values, CHANNEL_COLORS[i])
                if suffix == "mapping endmember" and profile:
                    profiles.append(profile)
                    used[name] = profile["values"]
        # Old records may contain preprocessing-stage references, but these are
        # not labelled as actual normalized mapping inputs without provenance.
        if raw is None and effective is None:
            for i, values in enumerate(getattr(fit, "references", [])):
                if np.isfinite(values).any():
                    add_spectrum(f"PTEE historical reference {i+1} (processing not recorded)", values)
        if used:
            payload["mapping_spectra"]["PTEE_endmembers"] = used
        layer("PTEE RGBY map", params.get("input_source", MISSING), "R²-weighted RGBY endmember map",
              getattr(fit, "rgb_y", None), params, profiles, "Mapping-normalized spectrum")

    result = getattr(window, "_multivariate_result", None)
    if result is not None and getattr(result, "n_clusters", None) is not None:
        params = dict(getattr(result, "analysis_parameters", None) or {"Historical conditions": MISSING})
        variance = getattr(result, "explained_variance", None)
        if variance is not None and np.asarray(variance).size:
            params["Explained variance fractions"] = np.asarray(variance).tolist()
        means = getattr(result, "mean_spectra", None)
        colors = getattr(result, "colors", None)
        sizes = getattr(result, "cluster_sizes", None)
        profiles = []
        if means is not None:
            means = np.asarray(means, float)
            for c in range(len(means)):
                if sizes is not None and c < len(np.asarray(sizes)) and int(np.asarray(sizes)[c]) == 0:
                    continue
                col = None
                if colors is not None and c < len(np.asarray(colors)):
                    col = "#%02x%02x%02x" % tuple(int(255 * float(v)) for v in np.asarray(colors)[c][:3])
                profiles.append(add_spectrum(f"Cluster {c+1} mean OD", means[c], col))
        else:
            payload["warnings"].append("PCA clustering: per-cluster mean spectra were not recorded")
        layer("PCA cluster map", params.get("input_source", MISSING),
              "PCA + k-means cluster map; per-cluster mean OD spectra",
              getattr(result, "rgb_map", None), params, profiles, "OD")
        scatter = _cluster_scatter(result)
        if scatter is not None and colors is not None:
            try:
                image = _render_cluster_scatter(scatter[0], scatter[1], scatter[2], np.asarray(colors, float))
                layer("PCA cluster PC-space", params.get("input_source", MISSING),
                      "PCA score scatter (PC1 vs PC2) coloured by cluster", image,
                      {"note": "Subsampled PC-space scatter"}, [])
            except Exception:
                payload["warnings"].append("PCA cluster PC-space scatter could not be rendered")
        else:
            payload["warnings"].append("PCA cluster PC-space scatter unavailable (scores not recorded)")
    elif result is not None:
        method = getattr(result, "method", "SVD/PCA (method not recorded)")
        params = dict(getattr(result, "analysis_parameters", None) or {"Historical conditions": MISSING})
        variance = getattr(result, "explained_variance", None)
        if variance is not None and np.asarray(variance).size:
            params["Explained variance fractions"] = np.asarray(variance).tolist()
        components = getattr(result, "components", None)
        profiles, used = [], {}
        coefficients = params.get("rgb_coefficients")
        if components is not None:
            coefficients = np.asarray(coefficients, dtype=float) if coefficients is not None else None
            for i, values in enumerate(components):
                name = f"{method} — component {i+1}"
                profile = add_spectrum(name, values, CHANNEL_COLORS[i] if i < 3 else None)
                profiles.append(profile)
                if (profile and coefficients is not None and coefficients.ndim == 2 and
                        i < coefficients.shape[1] and np.any(coefficients[:, i] != 0)):
                    used[name] = profile["values"]
            add_spectrum(f"{method} — input mean", getattr(result, "mean_spectrum", None))
        else:
            payload["warnings"].append(f"{method}: component spectra were not recorded")
        if coefficients is None:
            payload["warnings"].append(f"{method}: RGB coefficients were not recorded; mapping component selection is unavailable")
        if used:
            payload["mapping_spectra"]["Multivariate_mapping_components"] = used
        layer("SVD/PCA RGB map", params.get("input_source", MISSING), f"{method}; RGB score linear combination",
              getattr(result, "rgb_map", None), params, profiles, "Component amplitude")

    # Per-type raw-image export sets: stacks export every energy frame, maps
    # export the single map.  Consumed by the report window's per-type image
    # checkboxes; each selected set is written to its own folder.
    image_sets = []
    def add_set(name, kind, data, display=None):
        if data is not None:
            image_sets.append({"name": name, "kind": kind, "data": data,
                               "display": display, "energies": energies})
    add_set("Transmission", "stack", transmission, {"palette": "Grayscale"})
    base_od = registered if (registered is not None and od is not None) else od
    add_set("OD (registered)", "stack", base_od, {"palette": "Grayscale"})
    add_set("OD (-pre edge)", "stack", presub, {"palette": "Grayscale"})
    for nm, img in getattr(window, "_premap_maps", {}).items():
        disp = getattr(window, "_premap_display", {}).get(nm, {"palette": "Jet"})
        add_set("Peakmap" if "Peak" in nm else "Net absorption", "map", img, disp)
    add_set("PTEE-R2", "map", getattr(fit, "rgb_y", None) if fit is not None else None)
    if result is not None and getattr(result, "n_clusters", None) is not None:
        add_set("PCA-Clustering", "map", getattr(result, "rgb_map", None))
    elif result is not None:
        add_set("SVD/PCA", "map", getattr(result, "rgb_map", None))
    payload["image_sets"] = image_sets
    return payload
