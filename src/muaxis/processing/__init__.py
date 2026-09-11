"""Numerical processing independent of the GUI."""

from .absorbance import OpticalDensityResult, compute_optical_density
from .registration import RegistrationResult, register_translation_stack
from .premap import net_absorption_map, peak_map, pre_edge_subtract
from .segmentation import SegmentationResult, segment_clusters
from .fitting import SpectralFitResult, spectral_r2_map
from .multivariate import (
    MultivariateResult, ClusterResult, decompose_stack, linear_combination_map,
    mcr_als_reconstruct, cluster_stack, cluster_palette, cluster_labels_to_rgb,
)

__all__ = [
    "OpticalDensityResult",
    "compute_optical_density",
    "RegistrationResult",
    "register_translation_stack",
    "net_absorption_map",
    "pre_edge_subtract",
    "peak_map",
    "SegmentationResult",
    "segment_clusters",
    "SpectralFitResult",
    "spectral_r2_map",
    "MultivariateResult",
    "ClusterResult",
    "decompose_stack",
    "linear_combination_map",
    "mcr_als_reconstruct",
    "cluster_stack",
    "cluster_palette",
    "cluster_labels_to_rgb",
]
