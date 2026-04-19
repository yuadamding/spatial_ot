from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import ot
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors


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


def _standardize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32)


def _normalize_hist(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if total <= 1e-12:
        return np.full_like(x, 1.0 / max(x.size, 1), dtype=np.float64)
    return x / total


def _softmax_over_negative_costs(costs: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -np.asarray(costs, dtype=np.float32) / max(float(temperature), 1e-5)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    return probs.astype(np.float32)


def _subsample_grid_points(points: np.ndarray, target: int) -> np.ndarray:
    if target <= 0 or points.shape[0] <= target:
        return np.arange(points.shape[0], dtype=np.int32)
    x = points[:, 0]
    y = points[:, 1]
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    xr = max(x1 - x0, 1e-6)
    yr = max(y1 - y0, 1e-6)
    aspect = xr / yr
    nx = max(1, int(round(np.sqrt(target * aspect))))
    ny = max(1, int(round(target / max(nx, 1))))
    gx = np.clip(((x - x0) / xr * nx).astype(np.int32), 0, nx - 1)
    gy = np.clip(((y - y0) / yr * ny).astype(np.int32), 0, ny - 1)
    flat = gy * nx + gx
    selected: list[int] = []
    for cell in np.unique(flat):
        idx = np.flatnonzero(flat == cell)
        cell_x = x0 + ((cell % nx) + 0.5) * xr / nx
        cell_y = y0 + ((cell // nx) + 0.5) * yr / ny
        d = (x[idx] - cell_x) ** 2 + (y[idx] - cell_y) ** 2
        selected.append(int(idx[np.argmin(d)]))
        if len(selected) >= target:
            break
    selected = np.asarray(sorted(set(selected)), dtype=np.int32)
    if selected.size < target:
        remaining = np.setdiff1d(np.arange(points.shape[0], dtype=np.int32), selected, assume_unique=False)
        selected = np.concatenate([selected, remaining[: target - selected.size]])
    return np.sort(selected[:target])


def build_subregions(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    x = coords_um[:, 0]
    y = coords_um[:, 1]
    x_centers = np.arange(float(x.min()), float(x.max()) + stride_um, stride_um, dtype=np.float32)
    y_centers = np.arange(float(y.min()), float(y.max()) + stride_um, stride_um, dtype=np.float32)
    centers = np.stack(np.meshgrid(x_centers, y_centers), axis=-1).reshape(-1, 2)

    nn = NearestNeighbors(radius=radius_um, metric="euclidean")
    nn.fit(coords_um)
    memberships = nn.radius_neighbors(centers, return_distance=False)

    kept_centers: list[np.ndarray] = []
    kept_members: list[np.ndarray] = []
    for center, members in zip(centers, memberships, strict=False):
        if members.size < min_cells:
            continue
        kept_centers.append(center.astype(np.float32))
        kept_members.append(np.asarray(members, dtype=np.int32))

    if not kept_centers:
        raise RuntimeError("No valid subregions were created; lower min_cells or increase the radius.")

    centers_arr = np.vstack(kept_centers).astype(np.float32)
    if max_subregions > 0 and centers_arr.shape[0] > max_subregions:
        keep_idx = _subsample_grid_points(centers_arr, target=max_subregions)
        centers_arr = centers_arr[keep_idx]
        kept_members = [kept_members[int(i)] for i in keep_idx.tolist()]

    return centers_arr, kept_members


def _triangle_area(tri: np.ndarray) -> float:
    x1, y1 = tri[0]
    x2, y2 = tri[1]
    x3, y3 = tri[2]
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5


def _sample_points_in_triangle(tri: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[0] + u * (tri[1] - tri[0]) + v * (tri[2] - tri[0])


def _sample_uniform_points_in_convex_hull(coords: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if coords.shape[0] <= n_points:
        return coords.astype(np.float32)

    rng = np.random.default_rng(seed)
    try:
        hull = ConvexHull(coords)
        hull_pts = coords[hull.vertices]
    except Exception:
        hull_pts = coords

    if hull_pts.shape[0] < 3:
        take = rng.choice(coords.shape[0], size=min(n_points, coords.shape[0]), replace=False)
        return coords[take].astype(np.float32)

    anchor = hull_pts[0]
    triangles = []
    areas = []
    for i in range(1, hull_pts.shape[0] - 1):
        tri = np.vstack([anchor, hull_pts[i], hull_pts[i + 1]])
        area = _triangle_area(tri)
        if area > 1e-10:
            triangles.append(tri)
            areas.append(area)

    if not triangles:
        take = rng.choice(coords.shape[0], size=min(n_points, coords.shape[0]), replace=False)
        return coords[take].astype(np.float32)

    areas_arr = np.asarray(areas, dtype=np.float64)
    probs = areas_arr / areas_arr.sum()
    chosen = rng.choice(len(triangles), size=n_points, replace=True, p=probs)
    counts = np.bincount(chosen, minlength=len(triangles))
    samples = []
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        samples.append(_sample_points_in_triangle(triangles[idx], count, rng))
    return np.vstack(samples).astype(np.float32)


def _sample_uniform_points_in_polygon_components(
    polygon_components: list[np.ndarray],
    n_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    components = [np.asarray(poly, dtype=np.float64) for poly in polygon_components if np.asarray(poly).shape[0] >= 3]
    if not components:
        return np.zeros((0, 2), dtype=np.float32)

    bboxes = []
    areas = []
    paths = []
    for poly in components:
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        bboxes.append((xmin, xmax, ymin, ymax))
        areas.append(max((xmax - xmin) * (ymax - ymin), 1e-12))
        paths.append(MplPath(poly))
    probs = np.asarray(areas, dtype=np.float64)
    probs = probs / probs.sum()

    points = []
    while len(points) < n_points:
        comp_idx = int(rng.choice(len(components), p=probs))
        xmin, xmax, ymin, ymax = bboxes[comp_idx]
        batch_n = max(128, 2 * (n_points - len(points)))
        cand = np.column_stack(
            [
                rng.uniform(xmin, xmax, size=batch_n),
                rng.uniform(ymin, ymax, size=batch_n),
            ]
        )
        inside = paths[comp_idx].contains_points(cand)
        if np.any(inside):
            points.extend(cand[inside].tolist())
    return np.asarray(points[:n_points], dtype=np.float32)


def _sample_uniform_points_in_mask(
    mask: np.ndarray,
    n_points: int,
    seed: int,
    affine: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask, dtype=bool)
    ij = np.argwhere(mask)
    if ij.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pick = rng.choice(ij.shape[0], size=n_points, replace=ij.shape[0] < n_points)
    sampled = ij[pick].astype(np.float64)
    sampled += rng.random(sampled.shape)
    xy = sampled[:, ::-1]
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape == (3, 3):
            homo = np.column_stack([xy, np.ones(xy.shape[0])])
            xy = (homo @ affine.T)[:, :2]
    return xy.astype(np.float32)


def sample_geometry_points(
    region_geometry: RegionGeometry,
    observed_coords: np.ndarray,
    n_points: int,
    seed: int,
    allow_convex_hull_fallback: bool = True,
) -> tuple[np.ndarray, str, bool]:
    if region_geometry.mask is not None:
        return _sample_uniform_points_in_mask(region_geometry.mask, n_points=n_points, seed=seed, affine=region_geometry.affine), "mask", False
    if region_geometry.polygon_components:
        pts = _sample_uniform_points_in_polygon_components(region_geometry.polygon_components, n_points=n_points, seed=seed)
        if pts.shape[0] > 0:
            return pts, "polygon_components", False
    if region_geometry.polygon_vertices is not None and np.asarray(region_geometry.polygon_vertices).shape[0] >= 3:
        pts = _sample_uniform_points_in_polygon_components([np.asarray(region_geometry.polygon_vertices)], n_points=n_points, seed=seed)
        if pts.shape[0] > 0:
            return pts, "polygon", False
    if not allow_convex_hull_fallback:
        raise ValueError(
            f"Region '{region_geometry.region_id}' has no explicit geometry. "
            "Pass polygon/mask geometry or allow convex hull fallback explicitly."
        )
    warnings.warn(
        f"Region '{region_geometry.region_id}' has no explicit geometry; using convex hull of observed coordinates for shape normalization.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _sample_uniform_points_in_convex_hull(observed_coords, n_points=n_points, seed=seed), "convex_hull_fallback", True


def _region_geometries_from_members(
    subregion_members: list[np.ndarray],
) -> list[RegionGeometry]:
    return [
        RegionGeometry(
            region_id=f"region_{idx:04d}",
            members=np.asarray(members, dtype=np.int32),
        )
        for idx, members in enumerate(subregion_members)
    ]


def _ordered_hull_points(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 3:
        return coords
    try:
        hull = ConvexHull(coords)
        return coords[hull.vertices]
    except Exception:
        return coords


def _subregion_shape_descriptors(coords: np.ndarray) -> dict[str, float]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return {
            "shape_area": 0.0,
            "shape_perimeter": 0.0,
            "shape_compactness": 0.0,
            "shape_aspect_ratio": 1.0,
            "shape_eccentricity": 0.0,
            "shape_radius_mean": 0.0,
            "shape_radius_std": 0.0,
        }

    hull_pts = _ordered_hull_points(coords)
    if hull_pts.shape[0] >= 3:
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perimeter = float(np.sum(np.sqrt(np.sum((np.roll(hull_pts, -1, axis=0) - hull_pts) ** 2, axis=1))))
    else:
        area = 0.0
        perimeter = float(np.sum(np.sqrt(np.sum((coords - coords.mean(axis=0, keepdims=True)) ** 2, axis=1))))

    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T) if coords.shape[0] > 1 else np.eye(2)
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))[::-1]
    major = float(np.sqrt(eigvals[0]))
    minor = float(np.sqrt(eigvals[1]))
    aspect_ratio = major / max(minor, 1e-8)
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[1] / max(eigvals[0], 1e-12))))
    radius = np.sqrt(np.sum(centered**2, axis=1))
    compactness = float(4.0 * np.pi * area / max(perimeter**2, 1e-12)) if perimeter > 0 else 0.0
    return {
        "shape_area": float(area),
        "shape_perimeter": float(perimeter),
        "shape_compactness": compactness,
        "shape_aspect_ratio": float(aspect_ratio),
        "shape_eccentricity": eccentricity,
        "shape_radius_mean": float(radius.mean()) if radius.size else 0.0,
        "shape_radius_std": float(radius.std()) if radius.size else 0.0,
    }


def _shape_descriptor_frame(subregion_members: list[np.ndarray], coords_um: np.ndarray) -> pd.DataFrame:
    rows = []
    for rid, members in enumerate(subregion_members):
        desc = _subregion_shape_descriptors(coords_um[members])
        desc["subregion_id"] = int(rid)
        rows.append(desc)
    return pd.DataFrame(rows)


def _shape_leakage_balanced_accuracy(shape_df: pd.DataFrame, labels: np.ndarray, seed: int) -> float | None:
    if shape_df.empty:
        return None
    y = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(y)
    counts = counts[counts > 0]
    if counts.size < 2 or counts.min() < 2:
        return None
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return None
    x = shape_df.drop(columns=["subregion_id"], errors="ignore").to_numpy(dtype=np.float32)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="balanced_accuracy")
    return float(np.mean(scores))


def _shape_leakage_spatial_block_accuracy(
    shape_df: pd.DataFrame,
    labels: np.ndarray,
    centers_um: np.ndarray,
    seed: int,
    n_blocks: int = 5,
) -> float | None:
    if shape_df.empty:
        return None
    y = np.asarray(labels, dtype=np.int32)
    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        return None
    centers = np.asarray(centers_um, dtype=np.float32)
    if centers.shape[0] != y.shape[0]:
        return None
    block_count = min(int(n_blocks), centers.shape[0])
    if block_count < 2:
        return None
    block_km = KMeans(n_clusters=block_count, n_init=20, random_state=seed)
    blocks = block_km.fit_predict(centers)
    x = shape_df.drop(columns=["subregion_id"], errors="ignore").to_numpy(dtype=np.float32)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    scores: list[float] = []
    for block in np.unique(blocks):
        test_mask = blocks == block
        train_mask = ~test_mask
        if np.unique(y[train_mask]).size < 2 or np.unique(y[test_mask]).size < 2:
            continue
        clf.fit(x[train_mask], y[train_mask])
        pred = clf.predict(x[test_mask])
        denom = 0.0
        score = 0.0
        for label in unique_labels:
            mask = y[test_mask] == label
            if not np.any(mask):
                continue
            denom += 1.0
            score += float(np.mean(pred[mask] == label))
        if denom > 0:
            scores.append(score / denom)
    if not scores:
        return None
    return float(np.mean(scores))


def _shape_leakage_permutation_baseline(
    shape_df: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
    n_perm: int = 64,
) -> dict[str, float] | None:
    observed = _shape_leakage_balanced_accuracy(shape_df, labels, seed=seed)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    perm_scores = []
    for _ in range(int(n_perm)):
        score = _shape_leakage_balanced_accuracy(shape_df, rng.permutation(labels), seed=int(rng.integers(1_000_000)))
        if score is not None:
            perm_scores.append(score)
    if not perm_scores:
        return {"observed": float(observed), "perm_mean": float("nan"), "perm_p95": float("nan"), "excess": float("nan")}
    perm = np.asarray(perm_scores, dtype=np.float64)
    return {
        "observed": float(observed),
        "perm_mean": float(np.mean(perm)),
        "perm_p95": float(np.percentile(perm, 95)),
        "excess": float(observed - np.mean(perm)),
    }


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    new = np.asarray(new, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12))


def _validate_fit_inputs(
    features: np.ndarray,
    coords_um: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    max_iter: int,
    tol: float,
    n_init: int,
    min_scale: float,
    max_scale: float,
) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if coords_um.ndim != 2 or coords_um.shape[1] != 2:
        raise ValueError("coords_um must be a 2D array with shape (n, 2).")
    if features.shape[0] != coords_um.shape[0]:
        raise ValueError("features and coords_um must have the same number of rows.")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contains NaN or Inf.")
    if not np.all(np.isfinite(coords_um)):
        raise ValueError("coords_um contains NaN or Inf.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if atoms_per_cluster < 1:
        raise ValueError("atoms_per_cluster must be at least 1.")
    if radius_um <= 0 or stride_um <= 0:
        raise ValueError("radius_um and stride_um must be positive.")
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1.")
    if max_subregions != 0 and max_subregions < 1:
        raise ValueError("max_subregions must be positive or 0.")
    if lambda_x < 0 or lambda_y < 0:
        raise ValueError("lambda_x and lambda_y must be non-negative.")
    if geometry_eps <= 0 or ot_eps <= 0:
        raise ValueError("geometry_eps and ot_eps must be positive.")
    if rho <= 0:
        raise ValueError("rho must be positive.")
    if geometry_samples < 32:
        raise ValueError("geometry_samples must be at least 32.")
    if compressed_support_size < 2:
        raise ValueError("compressed_support_size must be at least 2.")
    if align_iters < 1 or max_iter < 1 or n_init < 1:
        raise ValueError("align_iters, max_iter, and n_init must be at least 1.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        raise ValueError("min_scale and max_scale must be positive and min_scale <= max_scale.")


def make_reference_points_unit_disk(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    n_points = max(int(n_points), 32)
    idx = np.arange(n_points, dtype=np.float64)
    radius = np.sqrt((idx + 0.5) / n_points)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = idx * golden
    q = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    w = np.full(n_points, 1.0 / n_points, dtype=np.float64)
    return q.astype(np.float32), w


def _normalize_coords_basic(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0, keepdims=True)
    x0 = x - center
    scale = float(np.sqrt(np.mean(np.sum(x0**2, axis=1))))
    scale = max(scale, 1e-8)
    return x0 / scale, center.astype(np.float64), scale


def fit_ot_shape_normalizer(
    geometry_points: np.ndarray,
    reference_points: np.ndarray,
    reference_weights: np.ndarray | None = None,
    eps_geom: float = 0.03,
    rbf_smoothing: float = 1e-3,
    rbf_neighbors: int = 64,
) -> tuple[ShapeNormalizer, ShapeNormalizerDiagnostics]:
    g = np.asarray(geometry_points, dtype=np.float64)
    q = np.asarray(reference_points, dtype=np.float64)

    g_norm, center, scale = _normalize_coords_basic(g)
    if reference_weights is None:
        w_ref = np.full(q.shape[0], 1.0 / max(q.shape[0], 1), dtype=np.float64)
    else:
        w_ref = _normalize_hist(reference_weights)
    w_geom = np.full(g_norm.shape[0], 1.0 / max(g_norm.shape[0], 1), dtype=np.float64)

    interpolator: RBFInterpolator | None = None
    ot_cost = None
    sinkhorn_converged = None
    mapped_radius_p95 = None
    mapped_radius_max = None
    interpolation_residual = None
    try:
        c = ot.dist(g_norm, q, metric="sqeuclidean")
        positive = c[c > 0]
        scale_cost = float(np.median(positive)) if positive.size else 1.0
        c = c / max(scale_cost, 1e-12)
        t, log = ot.sinkhorn(
            w_geom,
            w_ref,
            c,
            reg=max(float(eps_geom), 1e-5),
            method="sinkhorn_stabilized",
            numItermax=3000,
            stopThr=1e-8,
            warn=False,
            log=True,
        )
        row_mass = np.maximum(t.sum(axis=1, keepdims=True), 1e-12)
        g_mapped = (t @ q) / row_mass
        ot_cost = float(np.sum(t * c))
        err_hist = np.asarray(log.get("err", []), dtype=np.float64)
        sinkhorn_converged = bool(err_hist.size == 0 or err_hist[-1] < 1e-7)
        radius = np.sqrt(np.sum(g_mapped**2, axis=1))
        mapped_radius_p95 = float(np.percentile(radius, 95)) if radius.size else 0.0
        mapped_radius_max = float(radius.max()) if radius.size else 0.0
        interpolator = RBFInterpolator(
            g_norm,
            g_mapped,
            smoothing=rbf_smoothing,
            neighbors=min(int(rbf_neighbors), g_norm.shape[0]),
        )
        pred = np.asarray(interpolator(g_norm), dtype=np.float64)
        interpolation_residual = float(np.sqrt(np.mean(np.sum((pred - g_mapped) ** 2, axis=1))))
    except Exception:
        interpolator = None
    normalizer = ShapeNormalizer(center=center, scale=scale, interpolator=interpolator)
    diagnostics = ShapeNormalizerDiagnostics(
        geometry_source="unknown",
        used_fallback=False,
        ot_cost=ot_cost,
        sinkhorn_converged=sinkhorn_converged,
        mapped_radius_p95=mapped_radius_p95,
        mapped_radius_max=mapped_radius_max,
        interpolation_residual=interpolation_residual,
    )
    return normalizer, diagnostics


def _compress_measure(
    canonical_coords: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    m: int,
    lambda_x: float,
    lambda_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = canonical_coords.shape[0]
    if n <= m:
        return canonical_coords.astype(np.float32), features.astype(np.float32), _normalize_hist(weights).astype(np.float32)

    z = np.hstack([np.sqrt(lambda_x) * canonical_coords, np.sqrt(lambda_y) * features]).astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=m,
        random_state=random_state,
        batch_size=min(4096, n),
        n_init="auto",
    )
    labels = km.fit_predict(z, sample_weight=weights)
    u_centers = np.zeros((m, canonical_coords.shape[1]), dtype=np.float64)
    y_centers = np.zeros((m, features.shape[1]), dtype=np.float64)
    a_centers = np.zeros(m, dtype=np.float64)
    for j in range(m):
        mask = labels == j
        if not np.any(mask):
            continue
        a_j = float(weights[mask].sum())
        a_centers[j] = a_j
        norm = max(a_j, 1e-12)
        u_centers[j] = (weights[mask, None] * canonical_coords[mask]).sum(axis=0) / norm
        y_centers[j] = (weights[mask, None] * features[mask]).sum(axis=0) / norm
    keep = a_centers > 1e-12
    if not np.any(keep):
        return canonical_coords[:m].astype(np.float32), features[:m].astype(np.float32), _normalize_hist(weights[:m]).astype(np.float32)
    return (
        u_centers[keep].astype(np.float32),
        y_centers[keep].astype(np.float32),
        _normalize_hist(a_centers[keep]).astype(np.float32),
    )


def _build_subregion_measures(
    features: np.ndarray,
    coords_um: np.ndarray,
    centers_um: np.ndarray,
    region_geometries: list[RegionGeometry],
    geometry_reference_points: np.ndarray,
    geometry_reference_weights: np.ndarray,
    geometry_eps: float,
    geometry_samples: int,
    compressed_support_size: int,
    lambda_x: float,
    lambda_y: float,
    seed: int,
    allow_convex_hull_fallback: bool,
) -> list[SubregionMeasure]:
    measures: list[SubregionMeasure] = []
    for rid, region in enumerate(region_geometries):
        members = np.asarray(region.members, dtype=np.int32)
        local_coords = np.asarray(coords_um[members], dtype=np.float32)
        local_features = np.asarray(features[members], dtype=np.float32)
        weights = np.full(local_coords.shape[0], 1.0 / max(local_coords.shape[0], 1), dtype=np.float32)
        geom_points, geometry_source, used_fallback = sample_geometry_points(
            region,
            observed_coords=local_coords,
            n_points=max(int(geometry_samples), 32),
            seed=seed + rid,
            allow_convex_hull_fallback=allow_convex_hull_fallback,
        )
        normalizer, diagnostics = fit_ot_shape_normalizer(
            geometry_points=geom_points,
            reference_points=geometry_reference_points,
            reference_weights=geometry_reference_weights,
            eps_geom=geometry_eps,
        )
        diagnostics.geometry_source = geometry_source
        diagnostics.used_fallback = used_fallback
        canonical_coords = normalizer.transform(local_coords)
        uc, yc, ac = _compress_measure(
            canonical_coords=canonical_coords,
            features=local_features,
            weights=weights,
            m=max(int(compressed_support_size), 1),
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            random_state=seed + 10000 + rid,
        )
        measures.append(
            SubregionMeasure(
                subregion_id=rid,
                center_um=centers_um[rid].astype(np.float32),
                members=np.asarray(members, dtype=np.int32),
                canonical_coords=uc,
                features=yc,
                weights=ac,
                geometry_point_count=int(geom_points.shape[0]),
                compressed_point_count=int(uc.shape[0]),
                normalizer=normalizer,
                normalizer_diagnostics=diagnostics,
            )
        )
    return measures


def _measure_summary(measure: SubregionMeasure) -> np.ndarray:
    z = np.hstack([measure.canonical_coords, measure.features]).astype(np.float64)
    mean = np.average(z, axis=0, weights=measure.weights)
    var = np.average((z - mean) ** 2, axis=0, weights=measure.weights)
    return np.hstack([mean, np.sqrt(np.maximum(var, 0.0))]).astype(np.float32)


def _initialize_cluster_atoms(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    coord_dim = measures[0].canonical_coords.shape[1]
    feat_dim = measures[0].features.shape[1]
    atom_coords = np.zeros((n_clusters, atoms_per_cluster, coord_dim), dtype=np.float32)
    atom_features = np.zeros((n_clusters, atoms_per_cluster, feat_dim), dtype=np.float32)
    betas = np.zeros((n_clusters, atoms_per_cluster), dtype=np.float32)
    sx = np.sqrt(max(float(lambda_x), 1e-12))
    sy = np.sqrt(max(float(lambda_y), 1e-12))

    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            idx = np.asarray([int(rng.integers(len(measures)))], dtype=np.int32)
        u_pool = np.vstack([measures[r].canonical_coords for r in idx]).astype(np.float32)
        y_pool = np.vstack([measures[r].features for r in idx]).astype(np.float32)
        a_pool = np.concatenate([measures[r].weights for r in idx]).astype(np.float64)
        a_pool = _normalize_hist(a_pool)
        z_pool = np.hstack([np.sqrt(lambda_x) * u_pool, np.sqrt(lambda_y) * y_pool]).astype(np.float32)

        if z_pool.shape[0] >= atoms_per_cluster:
            km = KMeans(
                n_clusters=atoms_per_cluster,
                n_init=10,
                random_state=random_state + k,
            )
            local_labels = km.fit_predict(z_pool, sample_weight=a_pool)
            beta_k = np.zeros(atoms_per_cluster, dtype=np.float64)
            for ell in range(atoms_per_cluster):
                mask = local_labels == ell
                beta_k[ell] = a_pool[mask].sum()
                if beta_k[ell] <= 1e-12:
                    continue
                atom_coords[k, ell] = np.average(u_pool[mask], axis=0, weights=a_pool[mask]).astype(np.float32)
                atom_features[k, ell] = np.average(y_pool[mask], axis=0, weights=a_pool[mask]).astype(np.float32)
            dead = beta_k <= 1e-12
            if np.any(dead):
                centers = km.cluster_centers_
                atom_coords[k, dead] = (centers[dead, :coord_dim] / sx).astype(np.float32)
                atom_features[k, dead] = (centers[dead, coord_dim:] / sy).astype(np.float32)
            betas[k] = _normalize_hist(beta_k + 1e-6).astype(np.float32)
        else:
            reps = int(np.ceil(atoms_per_cluster / max(z_pool.shape[0], 1)))
            atom_coords[k] = np.tile(u_pool, (reps, 1))[:atoms_per_cluster]
            atom_features[k] = np.tile(y_pool, (reps, 1))[:atoms_per_cluster]
            betas[k] = np.full(atoms_per_cluster, 1.0 / atoms_per_cluster, dtype=np.float32)

    return atom_coords, atom_features, betas


def _estimate_cost_scales(measures: list[SubregionMeasure], max_points: int, random_state: int) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    u_all = np.vstack([m.canonical_coords for m in measures]).astype(np.float32)
    y_all = np.vstack([m.features for m in measures]).astype(np.float32)
    if u_all.shape[0] > max_points:
        idx = rng.choice(u_all.shape[0], size=max_points, replace=False)
        u_all = u_all[idx]
        y_all = y_all[idx]

    def _estimate_pairwise(x: np.ndarray, n_pairs: int = 200_000) -> float:
        n = x.shape[0]
        if n <= 1:
            return 1.0
        i = rng.integers(0, n, size=min(n_pairs, max(n * 8, 1024)))
        j = rng.integers(0, n, size=i.shape[0])
        d2 = np.sum((x[i] - x[j]) ** 2, axis=1)
        d2 = d2[d2 > 0]
        return float(np.median(d2)) if d2.size else 1.0

    sx = _estimate_pairwise(u_all)
    sy = _estimate_pairwise(y_all)
    return max(sx, 1e-6), max(sy, 1e-6)


def weighted_similarity_fit(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    allow_reflection: bool = True,
    allow_scale: bool = True,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = _normalize_hist(w)
    xbar = np.sum(w[:, None] * x, axis=0)
    ybar = np.sum(w[:, None] * y, axis=0)
    x0 = x - xbar
    y0 = y - ybar
    h = x0.T @ (w[:, None] * y0)
    u, s, vt = np.linalg.svd(h)
    r = u @ vt
    if not allow_reflection and np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    if allow_scale:
        denom = float(np.sum(w * np.sum(x0**2, axis=1)))
        scale = float(s.sum() / max(denom, 1e-12))
    else:
        scale = 1.0
    scale = float(np.clip(scale, min_scale, max_scale))
    t = ybar - scale * xbar @ r
    return {"R": r.astype(np.float64), "scale": scale, "t": t.astype(np.float64)}


def apply_similarity(x: np.ndarray, transform: dict[str, np.ndarray | float]) -> np.ndarray:
    r = np.asarray(transform["R"], dtype=np.float64)
    scale = float(transform["scale"])
    t = np.asarray(transform["t"], dtype=np.float64)
    return (scale * np.asarray(x, dtype=np.float64) @ r + t).astype(np.float64)


def _transform_penalty(
    transform: dict[str, np.ndarray | float],
    scale_penalty: float,
    shift_penalty: float,
) -> float:
    scale = max(float(transform["scale"]), 1e-12)
    t = np.asarray(transform["t"], dtype=np.float64)
    return float(scale_penalty * (np.log(scale) ** 2) + shift_penalty * float(t @ t))


def _solve_semirelaxed_unbalanced(
    a: np.ndarray,
    beta: np.ndarray,
    c: np.ndarray,
    eps: float,
    rho: float,
) -> tuple[np.ndarray, float]:
    regs = [max(float(eps), 1e-5)]
    regs.extend([regs[0] * 2.0, regs[0] * 4.0, regs[0] * 8.0])
    last_gamma = None
    last_objective = 1e12
    for reg in regs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gamma = ot.unbalanced.sinkhorn_unbalanced(
                a,
                beta,
                c,
                reg=reg,
                reg_m=(float("inf"), max(float(rho), 1e-6)),
                method="sinkhorn_stabilized",
                reg_type="kl",
                numItermax=3000,
                stopThr=1e-8,
            )
            objective = ot.unbalanced.sinkhorn_unbalanced2(
                a,
                beta,
                c,
                reg=reg,
                reg_m=(float("inf"), max(float(rho), 1e-6)),
                method="sinkhorn_stabilized",
                reg_type="kl",
                returnCost="total",
                numItermax=3000,
                stopThr=1e-8,
            )
        gamma = np.asarray(gamma, dtype=np.float64)
        objective = float(objective)
        last_gamma = gamma
        last_objective = objective
        numeric_warn = any(
            ("Numerical errors" in str(w.message)) or ("did not converge" in str(w.message))
            for w in caught
        )
        if np.all(np.isfinite(gamma)) and np.isfinite(objective) and not numeric_warn:
            return gamma, objective
    if last_gamma is None or not np.all(np.isfinite(last_gamma)):
        raise FloatingPointError("Unable to obtain a finite semi-relaxed unbalanced OT solution.")
    return last_gamma, last_objective if np.isfinite(last_objective) else 1e12


def aligned_semirelaxed_ot_to_cluster(
    u: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    beta: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    n_align_iter: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
) -> tuple[float, np.ndarray, dict[str, np.ndarray | float], np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = _normalize_hist(a)
    beta = _normalize_hist(beta)
    transform: dict[str, np.ndarray | float] = {"R": np.eye(2, dtype=np.float64), "scale": 1.0, "t": np.zeros(2, dtype=np.float64)}
    gamma = None

    for _ in range(max(int(n_align_iter), 1)):
        u_aligned = apply_similarity(u, transform)
        cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(cost_scale_x, 1e-12)
        cy = ot.dist(y, atom_features, metric="sqeuclidean") / max(cost_scale_y, 1e-12)
        c = lambda_x * cx + lambda_y * cy
        gamma, _ = _solve_semirelaxed_unbalanced(a=a, beta=beta, c=c, eps=eps, rho=rho)
        row_mass = np.maximum(gamma.sum(axis=1), 1e-12)
        target_bary = (gamma @ atom_coords) / row_mass[:, None]
        transform = weighted_similarity_fit(
            u,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    u_aligned = apply_similarity(u, transform)
    cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(cost_scale_x, 1e-12)
    cy = ot.dist(y, atom_features, metric="sqeuclidean") / max(cost_scale_y, 1e-12)
    c = lambda_x * cx + lambda_y * cy
    gamma, objective = _solve_semirelaxed_unbalanced(a=a, beta=beta, c=c, eps=eps, rho=rho)
    target_mass = np.asarray(gamma.sum(axis=0), dtype=np.float64)
    theta = _normalize_hist(target_mass)
    objective += _transform_penalty(
        transform,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
    )
    if not np.isfinite(objective):
        objective = 1e12
    return objective, gamma.astype(np.float32), transform, theta.astype(np.float32)


def _ensure_nonempty_clusters(labels: np.ndarray, costs: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = labels.copy()
    counts = pd.Series(labels).value_counts().to_dict()
    for k in range(n_clusters):
        if counts.get(k, 0) > 0:
            continue
        donor = max(counts, key=lambda x: counts[x])
        donor_idx = np.flatnonzero(labels == donor)
        worst = donor_idx[np.argmax(costs[donor_idx, donor])]
        labels[worst] = k
        counts[donor] -= 1
        counts[k] = 1
    return labels


def _compute_assignment_costs(
    measures: list[SubregionMeasure],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
) -> np.ndarray:
    n_subregions = len(measures)
    n_clusters = atom_coords.shape[0]
    costs = np.zeros((n_subregions, n_clusters), dtype=np.float64)

    for r, measure in enumerate(measures):
        for k in range(n_clusters):
            cost, _, _, _ = aligned_semirelaxed_ot_to_cluster(
                u=measure.canonical_coords,
                y=measure.features,
                a=measure.weights,
                atom_coords=atom_coords[k],
                atom_features=atom_features[k],
                beta=betas[k],
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=eps,
                rho=rho,
                n_align_iter=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
            )
            costs[r, k] = float(np.clip(cost, -1e12, 1e12))
    return costs.astype(np.float32)


def _compute_assigned_artifacts(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray | float]], list[np.ndarray], np.ndarray]:
    plans: list[np.ndarray] = []
    transforms: list[dict[str, np.ndarray | float]] = []
    thetas: list[np.ndarray] = []
    assigned_costs = np.zeros(len(measures), dtype=np.float32)
    for r, measure in enumerate(measures):
        k = int(labels[r])
        cost, gamma, transform, theta = aligned_semirelaxed_ot_to_cluster(
            u=measure.canonical_coords,
            y=measure.features,
            a=measure.weights,
            atom_coords=atom_coords[k],
            atom_features=atom_features[k],
            beta=betas[k],
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            n_align_iter=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
        )
        assigned_costs[r] = cost
        plans.append(gamma)
        transforms.append(transform)
        thetas.append(theta)
    return plans, transforms, thetas, assigned_costs


def _update_atoms(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    plans: list[np.ndarray],
    transforms: list[dict[str, np.ndarray | float]],
    thetas: list[np.ndarray],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    beta_smoothing: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_clusters, atoms_per_cluster, coord_dim = atom_coords.shape
    feat_dim = atom_features.shape[2]
    new_coords = atom_coords.copy().astype(np.float32)
    new_features = atom_features.copy().astype(np.float32)
    new_betas = np.zeros((n_clusters, atoms_per_cluster), dtype=np.float32)

    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            r = int(rng.integers(len(measures)))
            measure = measures[r]
            pick = rng.choice(
                measure.canonical_coords.shape[0],
                size=atoms_per_cluster,
                replace=measure.canonical_coords.shape[0] < atoms_per_cluster,
                p=measure.weights,
            )
            new_coords[k] = measure.canonical_coords[pick]
            new_features[k] = measure.features[pick]
            new_betas[k] = np.full(atoms_per_cluster, 1.0 / atoms_per_cluster, dtype=np.float32)
            continue

        coord_num = np.zeros((atoms_per_cluster, coord_dim), dtype=np.float64)
        feat_num = np.zeros((atoms_per_cluster, feat_dim), dtype=np.float64)
        denom = np.zeros(atoms_per_cluster, dtype=np.float64)
        beta_num = np.zeros(atoms_per_cluster, dtype=np.float64)

        for r in idx:
            measure = measures[r]
            gamma = np.asarray(plans[r], dtype=np.float64)
            theta = np.asarray(thetas[r], dtype=np.float64)
            transform = transforms[r]
            aligned = apply_similarity(measure.canonical_coords, transform)
            coord_num += gamma.T @ aligned
            feat_num += gamma.T @ measure.features
            denom += gamma.sum(axis=0)
            beta_num += theta

        active = denom > 1e-10
        if np.any(active):
            new_coords[k, active] = (coord_num[active] / denom[active, None]).astype(np.float32)
            new_features[k, active] = (feat_num[active] / denom[active, None]).astype(np.float32)

        if np.any(~active):
            u_pool = np.vstack([measures[r].canonical_coords for r in idx]).astype(np.float32)
            y_pool = np.vstack([measures[r].features for r in idx]).astype(np.float32)
            a_pool = _normalize_hist(np.concatenate([measures[r].weights for r in idx]))
            dead = np.where(~active)[0]
            chosen = rng.choice(
                u_pool.shape[0],
                size=dead.size,
                replace=u_pool.shape[0] < dead.size,
                p=a_pool,
            )
            new_coords[k, dead] = u_pool[chosen]
            new_features[k, dead] = y_pool[chosen]

        new_betas[k] = _normalize_hist(beta_num + beta_smoothing).astype(np.float32)

    return new_coords, new_features, new_betas


def _cell_cluster_feature_costs(
    features: np.ndarray,
    support_features: np.ndarray,
    prototype_weights: np.ndarray,
    temperature: float,
) -> np.ndarray:
    n_cells = features.shape[0]
    n_clusters = support_features.shape[0]
    costs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        dist = pairwise_distances(features, support_features[k], metric="sqeuclidean").astype(np.float32)
        weights = np.clip(prototype_weights[k], 1e-8, None).astype(np.float32)
        scores = np.exp(-dist / max(float(temperature), 1e-5)) * weights[None, :]
        costs[:, k] = -max(float(temperature), 1e-5) * np.log(np.maximum(scores.sum(axis=1), 1e-8))
    return costs


def _project_cells_from_subregions(
    features: np.ndarray,
    coords_um: np.ndarray,
    measures: list[SubregionMeasure],
    subregion_labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    prototype_weights: np.ndarray,
    assigned_transforms: list[dict[str, np.ndarray | float]],
    subregion_cluster_costs: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    assignment_temperature: float,
    context_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_features = atom_features
    feature_costs = _cell_cluster_feature_costs(
        features=features,
        support_features=support_features,
        prototype_weights=prototype_weights,
        temperature=assignment_temperature,
    )
    feature_probs = _softmax_over_negative_costs(feature_costs, temperature=assignment_temperature)
    subregion_probs = _softmax_over_negative_costs(subregion_cluster_costs, temperature=max(assignment_temperature, 1e-4))

    n_cells = features.shape[0]
    n_clusters = atom_coords.shape[0]
    context_probs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    membership_counts = np.zeros((n_cells, 1), dtype=np.float32)
    local_model_probs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    local_model_counts = np.zeros((n_cells, 1), dtype=np.float32)
    for r, measure in enumerate(measures):
        members = measure.members
        context_probs[members] += subregion_probs[r]
        membership_counts[members] += 1.0
        k = int(subregion_labels[r])
        canonical = measure.normalizer.transform(coords_um[members])
        aligned = apply_similarity(canonical, assigned_transforms[r]).astype(np.float32)
        cx = pairwise_distances(aligned, atom_coords[k], metric="sqeuclidean").astype(np.float32) / max(float(cost_scale_x), 1e-5)
        cy = pairwise_distances(features[members], atom_features[k], metric="sqeuclidean").astype(np.float32) / max(float(cost_scale_y), 1e-5)
        total_cost = float(lambda_x) * cx + float(lambda_y) * cy
        atom_scores = np.exp(-total_cost / max(float(assignment_temperature), 1e-5)) * np.clip(prototype_weights[k], 1e-8, None)[None, :]
        local_model_probs[members, k] += atom_scores.sum(axis=1).astype(np.float32)
        local_model_counts[members] += 1.0
    covered = membership_counts[:, 0] > 0
    if np.any(covered):
        context_probs[covered] /= membership_counts[covered]
    if np.any(~covered):
        context_probs[~covered] = feature_probs[~covered]
    local_covered = local_model_counts[:, 0] > 0
    if np.any(local_covered):
        local_model_probs[local_covered] /= local_model_counts[local_covered]
        local_model_probs[local_covered] /= np.maximum(local_model_probs[local_covered].sum(axis=1, keepdims=True), 1e-8)
    if np.any(~local_covered):
        local_model_probs[~local_covered] = feature_probs[~local_covered]

    combined = (
        feature_probs
        * np.power(np.clip(context_probs, 1e-8, None), float(context_weight))
        * np.power(np.clip(local_model_probs, 1e-8, None), 1.0 - float(context_weight))
    )
    combined = combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-8)
    labels = combined.argmax(axis=1).astype(np.int32)
    return labels, combined.astype(np.float32), local_model_probs.astype(np.float32), context_probs.astype(np.float32)


def fit_multilevel_ot(
    features: np.ndarray,
    coords_um: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
    scale_penalty: float = 0.05,
    shift_penalty: float = 0.05,
    n_init: int = 5,
    region_geometries: list[RegionGeometry] | None = None,
    allow_convex_hull_fallback: bool = True,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 1337,
) -> MultilevelOTResult:
    features = np.asarray(features, dtype=np.float32)
    coords_um = np.asarray(coords_um, dtype=np.float32)
    _validate_fit_inputs(
        features=features,
        coords_um=coords_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        geometry_eps=geometry_eps,
        ot_eps=ot_eps,
        rho=rho,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        align_iters=align_iters,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    features = _standardize_features(features)
    centers_um, subregion_members = build_subregions(
        coords_um=coords_um,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
    )
    if centers_um.shape[0] < n_clusters:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the number of constructed subregions={centers_um.shape[0]}."
        )

    if region_geometries is None:
        region_geometries = _region_geometries_from_members(subregion_members)
    if len(region_geometries) != len(subregion_members):
        raise ValueError("region_geometries must have the same length as the constructed subregions.")
    for rid, (region, members) in enumerate(zip(region_geometries, subregion_members, strict=False)):
        region.members = np.asarray(members, dtype=np.int32)
        if not region.region_id:
            region.region_id = f"region_{rid:04d}"

    reference_points, reference_weights = make_reference_points_unit_disk(geometry_samples)
    measures = _build_subregion_measures(
        features=features,
        coords_um=coords_um,
        centers_um=centers_um,
        region_geometries=region_geometries,
        geometry_reference_points=reference_points,
        geometry_reference_weights=reference_weights,
        geometry_eps=geometry_eps,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        seed=seed,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
    )

    summaries = np.vstack([_measure_summary(m) for m in measures])
    cost_scale_x, cost_scale_y = _estimate_cost_scales(measures, max_points=5000, random_state=seed)
    best_bundle: dict[str, object] | None = None
    restart_summaries: list[dict[str, float | int]] = []

    for run in range(int(n_init)):
        run_seed = seed + 1000 * run
        init = KMeans(n_clusters=n_clusters, n_init=20, random_state=run_seed)
        labels = init.fit_predict(summaries).astype(np.int32)
        atom_coords, atom_features, betas = _initialize_cluster_atoms(
            measures=measures,
            labels=labels,
            n_clusters=n_clusters,
            atoms_per_cluster=atoms_per_cluster,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            random_state=run_seed,
        )
        objective_history: list[dict[str, float]] = []

        for iteration in range(int(max_iter)):
            prev_coords = atom_coords.copy()
            prev_features = atom_features.copy()
            prev_labels = labels.copy()
            costs = _compute_assignment_costs(
                measures=measures,
                atom_coords=atom_coords,
                atom_features=atom_features,
                betas=betas,
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=ot_eps,
                rho=rho,
                align_iters=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
            )
            labels = _ensure_nonempty_clusters(costs.argmin(axis=1).astype(np.int32), costs, n_clusters)
            plans, transforms, thetas, assigned_costs = _compute_assigned_artifacts(
                measures=measures,
                labels=labels,
                atom_coords=atom_coords,
                atom_features=atom_features,
                betas=betas,
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=ot_eps,
                rho=rho,
                align_iters=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
            )
            atom_coords, atom_features, betas = _update_atoms(
                measures=measures,
                labels=labels,
                plans=plans,
                transforms=transforms,
                thetas=thetas,
                atom_coords=atom_coords,
                atom_features=atom_features,
                beta_smoothing=1e-3,
                random_state=run_seed + iteration,
            )

            label_change_rate = float(np.mean(labels != prev_labels))
            coord_shift = _relative_change(atom_coords, prev_coords)
            feat_shift = _relative_change(atom_features, prev_features)
            mean_obj = float(np.mean(assigned_costs))
            sorted_costs = np.sort(costs, axis=1)
            mean_margin = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0])) if sorted_costs.shape[1] >= 2 else float("nan")
            objective_history.append(
                {
                    "iteration": int(iteration + 1),
                    "mean_objective": mean_obj,
                    "label_change_rate": label_change_rate,
                    "coord_shift": coord_shift,
                    "feature_shift": feat_shift,
                    "mean_assignment_margin": mean_margin,
                }
            )
            if label_change_rate < 0.005 and max(coord_shift, feat_shift) < tol:
                break

        final_costs = _compute_assignment_costs(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
        )
        final_labels = _ensure_nonempty_clusters(final_costs.argmin(axis=1).astype(np.int32), final_costs, n_clusters)
        final_plans, final_transforms, final_thetas, final_assigned_costs = _compute_assigned_artifacts(
            measures=measures,
            labels=final_labels,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
        )
        final_objective = float(np.sum(final_assigned_costs))
        restart_summaries.append(
            {
                "run": int(run),
                "seed": int(run_seed),
                "objective": final_objective,
                "n_iter": int(len(objective_history)),
                "mean_assigned_cost": float(np.mean(final_assigned_costs)),
            }
        )
        if best_bundle is None or final_objective < float(best_bundle["objective"]):
            best_bundle = {
                "run": int(run),
                "objective": final_objective,
                "labels": final_labels.astype(np.int32),
                "costs": final_costs.astype(np.float32),
                "plans": final_plans,
                "transforms": final_transforms,
                "thetas": final_thetas,
                "assigned_costs": final_assigned_costs.astype(np.float32),
                "atom_coords": atom_coords.astype(np.float32),
                "atom_features": atom_features.astype(np.float32),
                "betas": betas.astype(np.float32),
                "objective_history": objective_history,
            }

    assert best_bundle is not None
    labels = np.asarray(best_bundle["labels"], dtype=np.int32)
    costs = np.asarray(best_bundle["costs"], dtype=np.float32)
    atom_coords = np.asarray(best_bundle["atom_coords"], dtype=np.float32)
    atom_features = np.asarray(best_bundle["atom_features"], dtype=np.float32)
    betas = np.asarray(best_bundle["betas"], dtype=np.float32)
    plans = list(best_bundle["plans"])
    transforms = list(best_bundle["transforms"])
    thetas = list(best_bundle["thetas"])
    objective_history = list(best_bundle["objective_history"])

    thetas_assigned = np.vstack([np.asarray(theta, dtype=np.float32) for theta in thetas]).astype(np.float32)
    subregion_cluster_probs = _softmax_over_negative_costs(costs, temperature=max(float(np.std(costs)), 1e-3))
    cluster_supports = np.concatenate([atom_coords, atom_features], axis=2).astype(np.float32)

    cell_cluster_labels, cell_cluster_probs, cell_feature_probs, cell_context_probs = _project_cells_from_subregions(
        features=features,
        coords_um=coords_um,
        measures=measures,
        subregion_labels=labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        prototype_weights=betas,
        assigned_transforms=transforms,
        subregion_cluster_costs=costs,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        assignment_temperature=max(cost_scale_y, 1e-3),
        context_weight=0.5,
    )

    return MultilevelOTResult(
        subregion_centers_um=centers_um.astype(np.float32),
        subregion_members=[m.members for m in measures],
        subregion_geometry_point_counts=np.asarray([m.geometry_point_count for m in measures], dtype=np.int32),
        subregion_geometry_sources=[m.normalizer_diagnostics.geometry_source for m in measures],
        subregion_geometry_used_fallback=np.asarray([m.normalizer_diagnostics.used_fallback for m in measures], dtype=bool),
        subregion_normalizer_radius_p95=np.asarray([m.normalizer_diagnostics.mapped_radius_p95 if m.normalizer_diagnostics.mapped_radius_p95 is not None else np.nan for m in measures], dtype=np.float32),
        subregion_normalizer_radius_max=np.asarray([m.normalizer_diagnostics.mapped_radius_max if m.normalizer_diagnostics.mapped_radius_max is not None else np.nan for m in measures], dtype=np.float32),
        subregion_normalizer_interpolation_residual=np.asarray([m.normalizer_diagnostics.interpolation_residual if m.normalizer_diagnostics.interpolation_residual is not None else np.nan for m in measures], dtype=np.float32),
        subregion_cluster_labels=labels,
        subregion_cluster_probs=subregion_cluster_probs.astype(np.float32),
        subregion_cluster_costs=costs.astype(np.float32),
        subregion_atom_weights=thetas_assigned.astype(np.float32),
        cluster_supports=cluster_supports.astype(np.float32),
        cluster_atom_coords=atom_coords.astype(np.float32),
        cluster_atom_features=atom_features.astype(np.float32),
        cluster_prototype_weights=betas.astype(np.float32),
        cell_feature_cluster_probs=cell_feature_probs.astype(np.float32),
        cell_context_cluster_probs=cell_context_probs.astype(np.float32),
        cell_cluster_probs=cell_cluster_probs.astype(np.float32),
        cell_cluster_labels=cell_cluster_labels.astype(np.int32),
        cost_scale_x=float(cost_scale_x),
        cost_scale_y=float(cost_scale_y),
        objective_history=objective_history,
        selected_restart=int(best_bundle["run"]),
        restart_summaries=restart_summaries,
    )


def _compute_subregion_embedding(weights: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    try:
        import umap.umap_ as umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(3, weights.shape[0] - 1)),
            min_dist=0.2,
            metric="euclidean",
            random_state=seed,
            transform_seed=seed,
        )
        return reducer.fit_transform(weights).astype(np.float32), "UMAP"
    except Exception:
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(weights).astype(np.float32), "PCA"


def _cluster_palette(n_clusters: int) -> np.ndarray:
    cmap_name = "tab20" if n_clusters <= 20 else "gist_ncar"
    cmap = plt.get_cmap(cmap_name, n_clusters)
    rgba = np.asarray([cmap(i) for i in range(n_clusters)], dtype=np.float32)
    return np.clip(np.rint(rgba[:, :3] * 255.0), 0, 255).astype(np.uint8)


def _save_multilevel_outputs(
    adata: ad.AnnData,
    result: MultilevelOTResult,
    output_dir: Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    radius_um: float,
    stride_um: float,
    embedding_2d: np.ndarray,
    embedding_name: str,
    shape_df: pd.DataFrame,
    summary: dict,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = output_dir / "cells_multilevel_ot.h5ad"
    subregions_path = output_dir / "subregions_multilevel_ot.parquet"
    supports_path = output_dir / "cluster_supports_multilevel_ot.npz"
    map_path = output_dir / "multilevel_ot_spatial_map.png"
    emb_path = output_dir / "multilevel_ot_subregion_embedding.png"
    atom_path = output_dir / "multilevel_ot_atom_layouts.png"
    summary_path = output_dir / "summary.json"
    outputs = {
        "h5ad": str(h5ad_path),
        "subregions": str(subregions_path),
        "supports": str(supports_path),
        "spatial_map": str(map_path),
        "subregion_embedding": str(emb_path),
        "atom_layouts": str(atom_path),
        "summary": str(summary_path),
    }
    summary["outputs"] = outputs

    palette = _cluster_palette(result.cluster_supports.shape[0])
    label_names = [f"C{int(x)}" for x in result.cell_cluster_labels]
    label_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette[result.cell_cluster_labels].tolist()]

    cells_out = adata.copy()
    cells_out.obs["mlot_cluster_id"] = pd.Categorical(label_names)
    cells_out.obs["mlot_cluster_int"] = result.cell_cluster_labels.astype(np.int32)
    cells_out.obs["mlot_cluster_hex"] = label_hex
    cells_out.obsm["mlot_cluster_probs"] = result.cell_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_feature_cluster_probs"] = result.cell_feature_cluster_probs.astype(np.float32)
    cells_out.obsm["mlot_context_cluster_probs"] = result.cell_context_cluster_probs.astype(np.float32)
    cells_out.uns["multilevel_ot"] = {
        "feature_obsm_key": feature_obsm_key,
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "summary_json": json.dumps(summary),
    }
    cells_out.write_h5ad(h5ad_path, compression="gzip")

    subregion_rows = []
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    subregion_margin = (
        sorted_costs[:, 1] - sorted_costs[:, 0]
        if sorted_costs.shape[1] >= 2
        else np.full(sorted_costs.shape[0], np.nan, dtype=np.float32)
    )
    for idx, members in enumerate(result.subregion_members):
        row = {
            "subregion_id": int(idx),
            "center_x_um": float(result.subregion_centers_um[idx, 0]),
            "center_y_um": float(result.subregion_centers_um[idx, 1]),
            "n_cells": int(len(members)),
            "geometry_point_count": int(result.subregion_geometry_point_counts[idx]),
            "geometry_source": result.subregion_geometry_sources[idx],
            "geometry_used_fallback": bool(result.subregion_geometry_used_fallback[idx]),
            "normalizer_radius_p95": float(result.subregion_normalizer_radius_p95[idx]) if np.isfinite(result.subregion_normalizer_radius_p95[idx]) else np.nan,
            "normalizer_radius_max": float(result.subregion_normalizer_radius_max[idx]) if np.isfinite(result.subregion_normalizer_radius_max[idx]) else np.nan,
            "normalizer_interpolation_residual": float(result.subregion_normalizer_interpolation_residual[idx]) if np.isfinite(result.subregion_normalizer_interpolation_residual[idx]) else np.nan,
            "cluster_id": f"C{int(result.subregion_cluster_labels[idx])}",
            "cluster_int": int(result.subregion_cluster_labels[idx]),
            "objective": float(result.subregion_cluster_costs[idx, result.subregion_cluster_labels[idx]]),
            "assignment_margin": float(subregion_margin[idx]) if np.isfinite(subregion_margin[idx]) else np.nan,
        }
        for j, prob in enumerate(result.subregion_cluster_probs[idx]):
            row[f"cluster_prob_{j:02d}"] = float(prob)
        for j, weight in enumerate(result.subregion_atom_weights[idx]):
            row[f"atom_weight_{j:02d}"] = float(weight)
        row["embed1"] = float(embedding_2d[idx, 0])
        row["embed2"] = float(embedding_2d[idx, 1])
        subregion_rows.append(row)
    subregions_df = pd.DataFrame(subregion_rows)
    if not shape_df.empty:
        subregions_df = subregions_df.merge(shape_df, on="subregion_id", how="left")
    subregions_df.to_parquet(subregions_path, index=False)

    np.savez_compressed(
        supports_path,
        cluster_supports=result.cluster_supports.astype(np.float32),
        cluster_atom_coords=result.cluster_atom_coords.astype(np.float32),
        cluster_atom_features=result.cluster_atom_features.astype(np.float32),
        cluster_prototype_weights=result.cluster_prototype_weights.astype(np.float32),
        subregion_atom_weights=result.subregion_atom_weights.astype(np.float32),
    )

    coords = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )
    point_size = 4.0 if coords.shape[0] > 100000 else 8.0
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for cid in range(result.cluster_supports.shape[0]):
        mask = result.cell_cluster_labels == cid
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            color=palette[cid] / 255.0,
            linewidths=0,
            alpha=0.85,
            rasterized=coords.shape[0] > 20000,
            label=f"C{cid} ({int(mask.sum())})",
        )
    ax.set_title("Shape-normalized multilevel OT cell labels")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.savefig(map_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for cid in range(result.cluster_supports.shape[0]):
        mask = result.subregion_cluster_labels == cid
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=16,
            color=palette[cid] / 255.0,
            linewidths=0,
            alpha=0.9,
            label=f"C{cid} ({int(mask.sum())})",
        )
    ax.set_title(f"Subregion {embedding_name} from learned mixture weights")
    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    fig.savefig(emb_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        nrows=result.cluster_supports.shape[0],
        ncols=1,
        figsize=(6.5, max(3.0, 2.4 * result.cluster_supports.shape[0])),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    feat_norm = np.linalg.norm(result.cluster_atom_features, axis=2)
    for cid, ax in enumerate(axes):
        sizes = 200.0 * np.clip(result.cluster_prototype_weights[cid], 0.05, None)
        sc = ax.scatter(
            result.cluster_atom_coords[cid, :, 0],
            result.cluster_atom_coords[cid, :, 1],
            s=sizes,
            c=feat_norm[cid],
            cmap="viridis",
            linewidths=0.5,
            edgecolors="black",
        )
        for atom_idx in range(result.cluster_atom_coords.shape[1]):
            ax.text(
                result.cluster_atom_coords[cid, atom_idx, 0],
                result.cluster_atom_coords[cid, atom_idx, 1],
                str(atom_idx),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
            )
        ax.set_title(f"Cluster C{cid} canonical atom layout")
        ax.set_xlabel("canonical x")
        ax.set_ylabel("canonical y")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="feature norm")
    fig.savefig(atom_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    summary_path.write_text(json.dumps(summary, indent=2))
    return outputs


def run_multilevel_ot_on_h5ad(
    input_h5ad: str | Path,
    output_dir: str | Path,
    feature_obsm_key: str,
    spatial_x_key: str,
    spatial_y_key: str,
    spatial_scale: float,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
    scale_penalty: float = 0.05,
    shift_penalty: float = 0.05,
    n_init: int = 5,
    allow_convex_hull_fallback: bool = True,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 1337,
) -> dict:
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    adata = ad.read_h5ad(input_h5ad)
    if feature_obsm_key not in adata.obsm:
        raise KeyError(f"Feature key '{feature_obsm_key}' not found in obsm.")
    if spatial_x_key not in adata.obs or spatial_y_key not in adata.obs:
        raise KeyError(f"Spatial keys '{spatial_x_key}' and/or '{spatial_y_key}' not found in obs.")
    if "umap" in feature_obsm_key.lower():
        warnings.warn(
            "Using UMAP coordinates as the OT feature space. UMAP is not generally metric-preserving; prefer PCA or standardized markers for validated runs.",
            RuntimeWarning,
            stacklevel=2,
        )

    features = np.asarray(adata.obsm[feature_obsm_key], dtype=np.float32)
    coords_um = np.stack(
        [
            np.asarray(adata.obs[spatial_x_key], dtype=np.float32) * spatial_scale,
            np.asarray(adata.obs[spatial_y_key], dtype=np.float32) * spatial_scale,
        ],
        axis=1,
    )

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        geometry_eps=geometry_eps,
        ot_eps=ot_eps,
        rho=rho,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        n_init=n_init,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )
    shape_df = _shape_descriptor_frame(result.subregion_members, coords_um)
    shape_leakage = _shape_leakage_balanced_accuracy(shape_df, result.subregion_cluster_labels, seed=seed)
    shape_leakage_block = _shape_leakage_spatial_block_accuracy(
        shape_df=shape_df,
        labels=result.subregion_cluster_labels,
        centers_um=result.subregion_centers_um,
        seed=seed,
    )
    shape_leakage_perm = _shape_leakage_permutation_baseline(shape_df, result.subregion_cluster_labels, seed=seed)
    embedding_2d, embedding_name = _compute_subregion_embedding(result.subregion_atom_weights, seed=seed)
    silhouette = None
    if np.unique(result.subregion_cluster_labels).size > 1:
        silhouette = float(
            silhouette_score(
                result.subregion_atom_weights,
                result.subregion_cluster_labels,
                metric="euclidean",
            )
        )
    sorted_costs = np.sort(result.subregion_cluster_costs, axis=1)
    margin = None
    if sorted_costs.shape[1] >= 2:
        margin = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0]))
    summary = {
        "input_h5ad": str(input_h5ad),
        "output_dir": str(output_dir),
        "feature_obsm_key": feature_obsm_key,
        "spatial_x_key": spatial_x_key,
        "spatial_y_key": spatial_y_key,
        "spatial_scale": float(spatial_scale),
        "n_cells": int(adata.n_obs),
        "feature_dim": int(features.shape[1]),
        "n_subregions": int(len(result.subregion_members)),
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
        "radius_um": float(radius_um),
        "stride_um": float(stride_um),
        "min_cells": int(min_cells),
        "max_subregions": int(max_subregions),
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "geometry_eps": float(geometry_eps),
        "ot_eps": float(ot_eps),
        "rho": float(rho),
        "geometry_samples": int(geometry_samples),
        "compressed_support_size": int(compressed_support_size),
        "align_iters": int(align_iters),
        "allow_reflection": bool(allow_reflection),
        "allow_scale": bool(allow_scale),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "scale_penalty": float(scale_penalty),
        "shift_penalty": float(shift_penalty),
        "n_init": int(n_init),
        "allow_convex_hull_fallback": bool(allow_convex_hull_fallback),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "seed": int(seed),
        "cost_scale_x": float(result.cost_scale_x),
        "cost_scale_y": float(result.cost_scale_y),
        "selected_restart": int(result.selected_restart),
        "restart_summaries": result.restart_summaries,
        "subregion_cluster_counts": {f"C{int(k)}": int(v) for k, v in pd.Series(result.subregion_cluster_labels).value_counts().sort_index().items()},
        "cell_cluster_counts": {f"C{int(k)}": int(v) for k, v in pd.Series(result.cell_cluster_labels).value_counts().sort_index().items()},
        "objective_history": result.objective_history,
        "subregion_embedding_method": embedding_name,
        "subregion_weight_silhouette": silhouette,
        "mean_assignment_margin": margin,
        "shape_leakage_balanced_accuracy": shape_leakage,
        "shape_leakage_spatial_block_accuracy": shape_leakage_block,
        "shape_leakage_permutation": shape_leakage_perm,
        "geometry_point_count_range": [
            int(result.subregion_geometry_point_counts.min()),
            int(result.subregion_geometry_point_counts.max()),
        ],
        "geometry_fallback_fraction": float(np.mean(result.subregion_geometry_used_fallback.astype(np.float32))),
        "geometry_source_counts": {
            key: int(value)
            for key, value in pd.Series(result.subregion_geometry_sources).value_counts().sort_index().items()
        },
        "normalizer_radius_p95_mean": float(np.nanmean(result.subregion_normalizer_radius_p95)),
        "normalizer_radius_max_mean": float(np.nanmean(result.subregion_normalizer_radius_max)),
        "normalizer_interpolation_residual_mean": float(np.nanmean(result.subregion_normalizer_interpolation_residual)),
        "method_notes": {
            "core": "shape-normalized cluster-specific semi-relaxed Wasserstein dictionary clustering",
            "geometry_normalization": "uniform geometry samples from each subregion are OT-mapped into a shared unit-disk reference domain before clustering",
            "geometry_proxy": "when explicit masks are unavailable and convex hull fallback is allowed, geometry samples are drawn from the convex hull of local cell coordinates",
            "local_measure": "compressed empirical measures over canonical coordinates and standardized cell-level features",
            "local_matching": "semi-relaxed unbalanced Sinkhorn with fixed source marginal and relaxed target marginal",
            "residual_alignment": "weighted similarity transform is optimized during subregion-to-cluster matching",
            "support_sharing": "subregions assigned to the same cluster reuse the same shared atom dictionary but keep subregion-specific mixture weights",
            "cell_boundary_projection": "cell labels are projected from canonical-coordinate plus feature fit to assigned cluster atoms, modulated by overlapping-subregion cluster evidence",
        },
    }
    _save_multilevel_outputs(
        adata=adata,
        result=result,
        output_dir=output_dir,
        feature_obsm_key=feature_obsm_key,
        spatial_x_key=spatial_x_key,
        spatial_y_key=spatial_y_key,
        spatial_scale=spatial_scale,
        radius_um=radius_um,
        stride_um=stride_um,
        embedding_2d=embedding_2d,
        embedding_name=embedding_name,
        shape_df=shape_df,
        summary=summary,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
