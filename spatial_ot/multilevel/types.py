from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class ShapeNormalizer:
    center: np.ndarray
    scale: float
    interpolator: RBFInterpolator | None

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_norm = (x - self.center) / max(self.scale, 1e-8)
        if self.interpolator is None:
            return x_norm.astype(np.float32)
        return np.asarray(self.interpolator(x_norm), dtype=np.float32)


@dataclass
class ShapeNormalizerDiagnostics:
    geometry_source: str
    used_fallback: bool
    ot_cost: float | None
    sinkhorn_converged: bool | None
    mapped_radius_p95: float | None
    mapped_radius_max: float | None
    interpolation_residual: float | None


@dataclass
class OTSolveDiagnostics:
    effective_eps: float
    used_fallback: bool


@dataclass
class RegionGeometry:
    region_id: str
    members: np.ndarray
    polygon_vertices: np.ndarray | None = None
    polygon_components: list[np.ndarray] | None = None
    mask: np.ndarray | None = None
    affine: np.ndarray | None = None


@dataclass
class SubregionMeasure:
    subregion_id: int
    center_um: np.ndarray
    members: np.ndarray
    canonical_coords: np.ndarray
    features: np.ndarray
    weights: np.ndarray
    geometry_point_count: int
    compressed_point_count: int
    normalizer: ShapeNormalizer
    normalizer_diagnostics: ShapeNormalizerDiagnostics


@dataclass
class MultilevelOTResult:
    subregion_centers_um: np.ndarray
    subregion_members: list[np.ndarray]
    subregion_argmin_labels: np.ndarray
    subregion_forced_label_mask: np.ndarray
    subregion_geometry_point_counts: np.ndarray
    subregion_geometry_sources: list[str]
    subregion_geometry_used_fallback: np.ndarray
    subregion_normalizer_radius_p95: np.ndarray
    subregion_normalizer_radius_max: np.ndarray
    subregion_normalizer_interpolation_residual: np.ndarray
    subregion_cluster_labels: np.ndarray
    subregion_cluster_probs: np.ndarray
    subregion_cluster_costs: np.ndarray
    subregion_atom_weights: np.ndarray
    subregion_assigned_effective_eps: np.ndarray
    subregion_assigned_used_ot_fallback: np.ndarray
    cluster_supports: np.ndarray
    cluster_atom_coords: np.ndarray
    cluster_atom_features: np.ndarray
    cluster_prototype_weights: np.ndarray
    cell_feature_cluster_probs: np.ndarray
    cell_context_cluster_probs: np.ndarray
    cell_cluster_probs: np.ndarray
    cell_cluster_labels: np.ndarray
    cost_scale_x: float
    cost_scale_y: float
    objective_history: list[dict[str, float]]
    selected_restart: int
    restart_summaries: list[dict[str, float | int]]
