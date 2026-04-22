from __future__ import annotations

import warnings

from matplotlib.path import Path as MplPath
import numpy as np
import ot
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
import torch

from .types import RegionGeometry, ShapeNormalizer, ShapeNormalizerDiagnostics


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


def _grid_centers(coords_um: np.ndarray, stride_um: float) -> np.ndarray:
    x = np.asarray(coords_um[:, 0], dtype=np.float32)
    y = np.asarray(coords_um[:, 1], dtype=np.float32)
    x_centers = np.arange(float(x.min()), float(x.max()) + stride_um, stride_um, dtype=np.float32)
    y_centers = np.arange(float(y.min()), float(y.max()) + stride_um, stride_um, dtype=np.float32)
    return np.stack(np.meshgrid(x_centers, y_centers), axis=-1).reshape(-1, 2).astype(np.float32)


def build_subregions(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    centers = _grid_centers(coords_um=coords_um, stride_um=stride_um)

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


def build_basic_niches(
    coords_um: np.ndarray,
    niche_size_um: float,
    min_cells: int,
    max_subregions: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    coords = np.asarray(coords_um, dtype=np.float32)
    if coords.shape[0] == 0:
        raise RuntimeError("No cells were provided, so no basic niches can be created.")

    niche_size_um = float(niche_size_um)
    x0 = float(coords[:, 0].min())
    y0 = float(coords[:, 1].min())
    # Assign each cell to its nearest niche center on the fixed-size grid so the
    # basic niches tile space without the pi/4 coverage loss from inscribed circles.
    grid_x = np.floor((coords[:, 0] - x0) / niche_size_um + 0.5).astype(np.int32)
    grid_y = np.floor((coords[:, 1] - y0) / niche_size_um + 0.5).astype(np.int32)

    niche_members: dict[tuple[int, int], list[int]] = {}
    for cell_idx, (ix, iy) in enumerate(zip(grid_x.tolist(), grid_y.tolist(), strict=False)):
        niche_members.setdefault((int(ix), int(iy)), []).append(int(cell_idx))

    kept_centers: list[np.ndarray] = []
    kept_members: list[np.ndarray] = []
    for ix, iy in sorted(niche_members):
        members = np.asarray(niche_members[(ix, iy)], dtype=np.int32)
        if members.size < min_cells:
            continue
        kept_centers.append(
            np.asarray(
                [x0 + float(ix) * niche_size_um, y0 + float(iy) * niche_size_um],
                dtype=np.float32,
            )
        )
        kept_members.append(members)

    if not kept_centers:
        raise RuntimeError("No valid basic niches were created; lower min_cells or decrease basic_niche_size_um.")

    centers_arr = np.vstack(kept_centers).astype(np.float32)
    if max_subregions > 0 and centers_arr.shape[0] > max_subregions:
        keep_idx = _subsample_grid_points(centers_arr, target=max_subregions)
        centers_arr = centers_arr[keep_idx]
        kept_members = [kept_members[int(i)] for i in keep_idx.tolist()]

    return centers_arr, kept_members


def build_composite_subregions_from_basic_niches(
    coords_um: np.ndarray,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    basic_niche_size_um: float,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    basic_niche_size_um = float(basic_niche_size_um)
    basic_niche_half_size_um = 0.5 * basic_niche_size_um
    basic_niche_cover_radius_um = float(np.sqrt(2.0) * basic_niche_half_size_um)
    basic_centers_um, basic_members = build_basic_niches(
        coords_um=coords_um,
        niche_size_um=basic_niche_size_um,
        min_cells=1,
        max_subregions=0,
    )

    if radius_um <= basic_niche_cover_radius_um + 1e-6:
        kept_idx = [idx for idx, members in enumerate(basic_members) if np.asarray(members).size >= min_cells]
        if not kept_idx:
            raise RuntimeError(
                "No valid basic niches were created; lower min_cells, decrease basic_niche_size_um, or increase the radius."
            )
        subregion_centers_um = basic_centers_um[np.asarray(kept_idx, dtype=np.int32)].astype(np.float32, copy=True)
        subregion_members = [np.asarray(basic_members[idx], dtype=np.int32) for idx in kept_idx]
        subregion_basic_niche_ids = [np.asarray([idx], dtype=np.int32) for idx in kept_idx]
        if max_subregions > 0 and subregion_centers_um.shape[0] > max_subregions:
            keep_idx = _subsample_grid_points(subregion_centers_um, target=max_subregions)
            subregion_centers_um = subregion_centers_um[keep_idx]
            subregion_members = [subregion_members[int(i)] for i in keep_idx.tolist()]
            subregion_basic_niche_ids = [subregion_basic_niche_ids[int(i)] for i in keep_idx.tolist()]
        return (
            subregion_centers_um,
            subregion_members,
            basic_centers_um.astype(np.float32),
            [np.asarray(members, dtype=np.int32) for members in basic_members],
            subregion_basic_niche_ids,
        )

    candidate_centers_um = _grid_centers(coords_um=coords_um, stride_um=stride_um)
    composite_radius_um = max(float(radius_um), basic_niche_cover_radius_um)
    niche_selection_radius_um = composite_radius_um + basic_niche_cover_radius_um

    nn = NearestNeighbors(radius=niche_selection_radius_um, metric="euclidean")
    nn.fit(basic_centers_um)
    niche_memberships = nn.radius_neighbors(candidate_centers_um, return_distance=False)

    kept_centers: list[np.ndarray] = []
    kept_members: list[np.ndarray] = []
    kept_niche_ids: list[np.ndarray] = []
    seen_niche_sets: set[tuple[int, ...]] = set()
    for center_um, niche_ids in zip(candidate_centers_um, niche_memberships, strict=False):
        if niche_ids.size == 0:
            continue
        niche_ids = np.asarray(np.unique(niche_ids), dtype=np.int32)
        niche_key = tuple(int(x) for x in niche_ids.tolist())
        if niche_key in seen_niche_sets:
            continue
        members = np.unique(np.concatenate([basic_members[int(niche_id)] for niche_id in niche_ids.tolist()])).astype(np.int32)
        if members.size < min_cells:
            continue
        seen_niche_sets.add(niche_key)
        kept_centers.append(np.asarray(center_um, dtype=np.float32))
        kept_members.append(members)
        kept_niche_ids.append(niche_ids)

    if not kept_centers:
        raise RuntimeError(
            "No valid composite subregions were created from the basic niches; "
            "lower min_cells, decrease basic_niche_size_um, or increase the radius."
        )

    subregion_centers_um = np.vstack(kept_centers).astype(np.float32)
    if max_subregions > 0 and subregion_centers_um.shape[0] > max_subregions:
        keep_idx = _subsample_grid_points(subregion_centers_um, target=max_subregions)
        subregion_centers_um = subregion_centers_um[keep_idx]
        kept_members = [kept_members[int(i)] for i in keep_idx.tolist()]
        kept_niche_ids = [kept_niche_ids[int(i)] for i in keep_idx.tolist()]

    return (
        subregion_centers_um,
        kept_members,
        basic_centers_um.astype(np.float32),
        [np.asarray(members, dtype=np.int32) for members in basic_members],
        kept_niche_ids,
    )


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
        x = poly[:, 0]
        y = poly[:, 1]
        poly_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(max(float(poly_area), 1e-12))
        paths.append(MplPath(poly))
    probs = np.asarray(areas, dtype=np.float64)
    probs = probs / probs.sum()

    points = []
    max_attempts = max(32, n_points // 8)
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
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
    if len(points) < n_points:
        raise ValueError("Unable to sample enough points from polygon geometry.")
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
    warn_on_fallback: bool = True,
) -> tuple[np.ndarray, str, bool]:
    if region_geometry.mask is not None:
        pts = _sample_uniform_points_in_mask(region_geometry.mask, n_points=n_points, seed=seed, affine=region_geometry.affine)
        if pts.shape[0] == 0:
            raise ValueError(f"Region '{region_geometry.region_id}' provided an empty mask geometry.")
        return pts, "mask", False
    if region_geometry.polygon_components:
        pts = _sample_uniform_points_in_polygon_components(region_geometry.polygon_components, n_points=n_points, seed=seed)
        if pts.shape[0] > 0:
            return pts, "polygon_components", False
        raise ValueError(f"Region '{region_geometry.region_id}' provided polygon components but sampling produced no points.")
    if region_geometry.polygon_vertices is not None and np.asarray(region_geometry.polygon_vertices).shape[0] >= 3:
        pts = _sample_uniform_points_in_polygon_components([np.asarray(region_geometry.polygon_vertices)], n_points=n_points, seed=seed)
        if pts.shape[0] > 0:
            return pts, "polygon", False
        raise ValueError(f"Region '{region_geometry.region_id}' provided polygon geometry but sampling produced no points.")
    if not allow_convex_hull_fallback:
        raise ValueError(
            f"Region '{region_geometry.region_id}' has no explicit geometry. "
            "Pass polygon/mask geometry or allow convex hull fallback explicitly."
        )
    if warn_on_fallback:
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


def _shape_descriptor_frame(
    subregion_members: list[np.ndarray],
    coords_um: np.ndarray,
    region_geometries: list[RegionGeometry] | None = None,
) -> pd.DataFrame:
    rows = []
    for rid, members in enumerate(subregion_members):
        source = "observed_coordinate_hull"
        if region_geometries is not None:
            region = region_geometries[rid]
            if region.mask is not None:
                geom = _sample_uniform_points_in_mask(region.mask, n_points=512, seed=rid, affine=region.affine)
                if geom.shape[0] > 0:
                    desc = _subregion_shape_descriptors(geom)
                    source = "explicit_mask"
                else:
                    desc = _subregion_shape_descriptors(coords_um[members])
            elif region.polygon_components:
                pts = np.vstack([np.asarray(poly, dtype=np.float64) for poly in region.polygon_components if np.asarray(poly).shape[0] >= 3])
                if pts.shape[0] > 0:
                    desc = _subregion_shape_descriptors(pts)
                    source = "explicit_polygon"
                else:
                    desc = _subregion_shape_descriptors(coords_um[members])
            elif region.polygon_vertices is not None and np.asarray(region.polygon_vertices).shape[0] >= 3:
                desc = _subregion_shape_descriptors(np.asarray(region.polygon_vertices, dtype=np.float64))
                source = "explicit_polygon"
            else:
                desc = _subregion_shape_descriptors(coords_um[members])
        else:
            desc = _subregion_shape_descriptors(coords_um[members])
        desc["subregion_id"] = int(rid)
        desc["shape_descriptor_source"] = source
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
    numeric_cols = [c for c in shape_df.columns if c != "subregion_id" and np.issubdtype(shape_df[c].dtype, np.number)]
    x = shape_df[numeric_cols].to_numpy(dtype=np.float32)
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
    numeric_cols = [c for c in shape_df.columns if c != "subregion_id" and np.issubdtype(shape_df[c].dtype, np.number)]
    x = shape_df[numeric_cols].to_numpy(dtype=np.float32)
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


def _validate_fit_inputs(
    features: np.ndarray,
    coords_um: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    basic_niche_size_um: float | None,
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
    if basic_niche_size_um is not None and basic_niche_size_um <= 0:
        raise ValueError("basic_niche_size_um must be positive when set.")
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


def _gpu_balanced_sinkhorn_transport(
    g_norm: np.ndarray,
    q: np.ndarray,
    w_geom: np.ndarray,
    w_ref: np.ndarray,
    eps_geom: float,
    compute_device: torch.device,
) -> tuple[np.ndarray, float, bool]:
    """GPU balanced Sinkhorn for the geometry shape normalizer.

    Returns (T, ot_cost, converged).
    """
    from .gpu_ot import sinkhorn_balanced_log_torch

    dtype = torch.float32
    g_t = torch.as_tensor(g_norm, dtype=dtype, device=compute_device)
    q_t = torch.as_tensor(q, dtype=dtype, device=compute_device)
    c = torch.cdist(g_t, q_t, p=2).pow(2)
    positive = c[c > 0]
    scale_cost = float(positive.median().item()) if positive.numel() else 1.0
    c = c / max(scale_cost, 1e-12)
    a_t = torch.as_tensor(w_geom, dtype=dtype, device=compute_device)
    b_t = torch.as_tensor(w_ref, dtype=dtype, device=compute_device)
    t, transport_cost, converged, _ = sinkhorn_balanced_log_torch(
        a_t,
        b_t,
        c,
        eps=max(float(eps_geom), 1e-5),
        num_iter=600,
        tol=1e-5,
    )
    if not bool(torch.isfinite(t).all().item()):
        # Re-try with a more relaxed regulariser before falling back to numpy.
        for fallback_eps in (max(float(eps_geom), 1e-5) * 4.0, max(float(eps_geom), 1e-5) * 16.0):
            t, transport_cost, converged, _ = sinkhorn_balanced_log_torch(
                a_t,
                b_t,
                c,
                eps=fallback_eps,
                num_iter=600,
                tol=1e-5,
            )
            if bool(torch.isfinite(t).all().item()):
                break
        else:
            raise RuntimeError("GPU balanced Sinkhorn produced non-finite plan even after regulariser fallback.")
    return (
        t.detach().cpu().numpy().astype(np.float64),
        float(transport_cost.detach().item()),
        bool(converged),
    )


def _fit_degenerate_shape_normalizer(
    geometry_points: np.ndarray,
) -> tuple[ShapeNormalizer, ShapeNormalizerDiagnostics]:
    """Fallback normalizer for subregions with fewer than 3 geometry points.

    There is not enough geometry to estimate a stable OT map in this case, so we
    keep the coordinates in a centered-and-scaled local frame and skip the OT
    interpolation step entirely.
    """

    g = np.asarray(geometry_points, dtype=np.float64)
    g_norm, center, scale = _normalize_coords_basic(g)
    radius = np.sqrt(np.sum(g_norm**2, axis=1))
    normalizer = ShapeNormalizer(center=center, scale=scale, interpolator=None)
    diagnostics = ShapeNormalizerDiagnostics(
        geometry_source="unknown",
        used_fallback=False,
        ot_cost=None,
        sinkhorn_converged=None,
        mapped_radius_p95=float(np.percentile(radius, 95)) if radius.size else 0.0,
        mapped_radius_max=float(radius.max()) if radius.size else 0.0,
        interpolation_residual=0.0,
    )
    return normalizer, diagnostics


def fit_ot_shape_normalizer(
    geometry_points: np.ndarray,
    reference_points: np.ndarray,
    reference_weights: np.ndarray | None = None,
    eps_geom: float = 0.03,
    rbf_smoothing: float = 1e-3,
    rbf_neighbors: int = 64,
    compute_device: torch.device | None = None,
) -> tuple[ShapeNormalizer, ShapeNormalizerDiagnostics]:
    g = np.asarray(geometry_points, dtype=np.float64)
    q = np.asarray(reference_points, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 2:
        raise ValueError("geometry_points must have shape (n_points, 2).")
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("reference_points must have shape (n_points, 2).")
    if g.shape[0] < 1:
        raise ValueError("At least 1 geometry point is required for shape normalization.")
    if not np.all(np.isfinite(g)):
        raise ValueError("geometry_points contains NaN or Inf.")
    if not np.all(np.isfinite(q)):
        raise ValueError("reference_points contains NaN or Inf.")
    if g.shape[0] < 3:
        return _fit_degenerate_shape_normalizer(g)

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
        use_gpu = compute_device is not None and torch.device(compute_device).type == "cuda"
        if use_gpu:
            t, ot_cost, sinkhorn_converged = _gpu_balanced_sinkhorn_transport(
                g_norm=g_norm,
                q=q,
                w_geom=w_geom,
                w_ref=w_ref,
                eps_geom=eps_geom,
                compute_device=torch.device(compute_device),
            )
            row_mass = np.maximum(t.sum(axis=1, keepdims=True), 1e-12)
            g_mapped = (t @ q) / row_mass
        else:
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
    except Exception as exc:
        raise RuntimeError("Shape normalization failed.") from exc
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
