"""Threshold segmentation and connected-cluster profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    labels: NDArray[np.int32]
    areas: NDArray[np.int64]
    mean_profiles: NDArray[np.float64]  # (cluster, energy)
    threshold: tuple[float, float]
    minimum_area: int
    # Profiles explicitly saved from the GUI.  A cluster id of 0 denotes the
    # average of the selected clusters and is intentionally separate from the
    # pixel label image (whose labels remain 1..N).
    saved_profiles: NDArray[np.float64] | None = None
    saved_areas: NDArray[np.int64] | None = None
    saved_cluster_ids: NDArray[np.int32] | None = None
    saved_label: str | None = None
    analysis_parameters: dict = field(default_factory=dict)


def segment_clusters(
    feature_map: NDArray[np.float64],
    od_stack: NDArray[np.float64],
    *,
    threshold: tuple[float, float],
    minimum_area: int = 1,
) -> SegmentationResult:
    """Threshold a 2-D feature map and calculate mean OD per 4-connected cluster."""
    image = np.asarray(feature_map, dtype=np.float64)
    od = np.asarray(od_stack, dtype=np.float64)
    if image.ndim != 2 or od.ndim != 3 or od.shape[1:] != image.shape:
        raise ValueError("feature_map must be (y, x), od_stack must be (energy, y, x)")
    low, high = sorted(map(float, threshold))
    mask = np.isfinite(image) & (image >= low) & (image <= high)
    labels, count = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8))
    areas_all = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas_all >= max(1, int(minimum_area))
    keep[0] = False
    relabel = np.zeros(count + 1, dtype=np.int32)
    relabel[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    final_labels = relabel[labels].astype(np.int32)
    areas = np.bincount(final_labels.ravel(), minlength=int(keep.sum()) + 1)[1:].astype(np.int64)
    profiles = np.full((len(areas), od.shape[0]), np.nan, dtype=np.float64)
    for cluster_id in range(1, len(areas) + 1):
        pixels = final_labels == cluster_id
        profiles[cluster_id - 1] = np.nanmean(od[:, pixels], axis=1)
    return SegmentationResult(final_labels, areas, profiles, (low, high), max(1, int(minimum_area)))
