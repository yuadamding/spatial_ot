from __future__ import annotations

from collections import Counter
from itertools import combinations
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.utils.extmath import randomized_svd


def parse_candidate_n_clusters(value: str | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        if "-" in raw and "," not in raw:
            left, right = raw.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())
            if stop < start:
                raise ValueError("candidate K range must be increasing, e.g. '2-8'.")
            return tuple(range(start, stop + 1))
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(item) for item in value)


def sanitize_candidate_n_clusters(
    candidates: tuple[int, ...] | list[int] | str | None,
    *,
    fallback_n_clusters: int,
    n_subregions: int | None = None,
) -> tuple[int, ...]:
    parsed = parse_candidate_n_clusters(candidates)
    if not parsed:
        parsed = tuple(range(2, max(int(fallback_n_clusters), 2) + 1))
    cleaned = sorted({int(k) for k in parsed if int(k) >= 2})
    if n_subregions is not None:
        cleaned = [k for k in cleaned if k < int(n_subregions)]
    if not cleaned:
        upper = int(n_subregions) - 1 if n_subregions is not None else int(fallback_n_clusters)
        upper = max(2, upper)
        cleaned = [min(max(2, int(fallback_n_clusters)), upper)]
    return tuple(cleaned)


def effective_min_cluster_size(n_items: int, n_clusters: int, requested_min_cluster_size: int) -> int:
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive.")
    n_items = int(n_items)
    n_clusters = int(n_clusters)
    requested = max(int(requested_min_cluster_size), 1)
    if n_items < n_clusters:
        raise ValueError("n_items must be at least n_clusters.")
    return max(1, min(requested, n_items // n_clusters))


def repair_labels_to_minimum_size(
    labels: np.ndarray,
    costs_to_clusters: np.ndarray,
    *,
    n_clusters: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int32).copy()
    costs = np.asarray(costs_to_clusters, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("costs_to_clusters must be a 2D matrix.")
    if labels_arr.ndim != 1 or labels_arr.shape[0] != costs.shape[0]:
        raise ValueError("labels must be a 1D vector with one row per cost row.")
    if costs.shape[1] < int(n_clusters):
        raise ValueError("costs_to_clusters has fewer columns than n_clusters.")
    n_items = int(labels_arr.shape[0])
    n_clusters = int(n_clusters)
    effective_min = effective_min_cluster_size(n_items, n_clusters, int(min_cluster_size))
    forced = np.zeros(n_items, dtype=bool)
    if n_items == 0:
        return labels_arr, forced
    labels_arr = np.clip(labels_arr, 0, n_clusters - 1).astype(np.int32)
    counts = np.bincount(labels_arr, minlength=n_clusters).astype(np.int64)
    current_cost = costs[np.arange(n_items), labels_arr].astype(np.float64)

    while True:
        deficient = np.flatnonzero(counts < effective_min)
        if deficient.size == 0:
            break
        target = int(deficient[np.argmin(counts[deficient])])
        feasible = (labels_arr != target) & (counts[labels_arr] > effective_min)
        if not np.any(feasible):
            raise RuntimeError("Cannot repair cluster sizes without violating another cluster's minimum size.")
        penalty = costs[:, target].astype(np.float64) - current_cost
        penalty[~feasible] = np.inf
        row = int(np.argmin(penalty))
        if not np.isfinite(penalty[row]):
            raise RuntimeError("Cannot repair cluster sizes because no finite reassignment is available.")
        donor = int(labels_arr[row])
        counts[donor] -= 1
        labels_arr[row] = target
        counts[target] += 1
        current_cost[row] = float(costs[row, target])
        forced[row] = True
    return labels_arr.astype(np.int32), forced.astype(bool)


def _finite_float_matrix(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("OT landmark costs must be a 2D matrix.")
    if x.shape[0] < 3:
        raise ValueError("At least three subregions are required for automatic K selection.")
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError("OT landmark costs contain no finite values.")
    finite_values = x[finite]
    fill = float(np.nanmax(finite_values))
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=float(np.nanmin(finite_values)))
    return x


def _standardize_landmark_costs(costs: np.ndarray) -> np.ndarray:
    x = _finite_float_matrix(costs)
    row_centered = x - np.median(x, axis=1, keepdims=True)
    col_mean = row_centered.mean(axis=0, keepdims=True)
    col_std = row_centered.std(axis=0, keepdims=True)
    z = (row_centered - col_mean) / np.maximum(col_std, 1e-8)
    return z.astype(np.float32)


def ot_landmark_distance_matrix(costs: np.ndarray) -> np.ndarray:
    embedding = _standardize_landmark_costs(costs)
    sq = np.sum(embedding * embedding, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (embedding @ embedding.T)
    d = np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)
    positive = d[d > 0]
    if positive.size:
        d /= max(float(np.median(positive)), 1e-8)
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def standardize_latent_embeddings(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("subregion latent embeddings must be a 2D matrix.")
    if x.shape[0] < 1:
        raise ValueError("At least one subregion latent embedding is required.")
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError("subregion latent embeddings contain no finite values.")
    finite_values = x[finite]
    fill = float(np.nanmax(finite_values))
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=float(np.nanmin(finite_values)))
    if x.shape[0] > 50000:
        sample_size = min(50000, int(x.shape[0]))
        sample_idx = np.linspace(0, int(x.shape[0]) - 1, num=sample_size, dtype=np.int64)
        sample = x[sample_idx]
        center = np.median(sample, axis=0, keepdims=True)
        scale = np.std(sample - center, axis=0, keepdims=True)
    else:
        center = np.median(x, axis=0, keepdims=True)
        scale = np.std(x - center, axis=0, keepdims=True)
    positive_scale = scale[np.isfinite(scale) & (scale > 0)]
    scale_floor = 1e-8
    if positive_scale.size:
        scale_floor = max(scale_floor, 0.25 * float(np.median(positive_scale)))
    z = (x - center) / np.maximum(scale, scale_floor)
    z = np.clip(z, -10.0, 10.0)
    return z.astype(np.float32)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def prepare_latent_clustering_embedding(
    latent_embeddings: np.ndarray,
    *,
    max_components: int | None = None,
    sample_size: int | None = None,
    random_state: int = 1337,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the denoised Euclidean embedding used by pooled-latent clustering.

    The primary label step must remain feature-only, but high-dimensional
    moment/codebook summaries can be noisy. This preprocessing robustly scales
    the raw subregion summaries, then fits PCA axes on a deterministic row
    sample when the dimension is high. The full cohort is projected onto those
    axes and re-standardized before KMeans/model selection.
    """

    z = standardize_latent_embeddings(latent_embeddings)
    n_items, raw_dim = int(z.shape[0]), int(z.shape[1])
    target = _env_int("SPATIAL_OT_SUBREGION_LATENT_PCA_COMPONENTS", 64) if max_components is None else int(max_components)
    target = max(int(target), 0)
    max_sample = _env_int("SPATIAL_OT_SUBREGION_LATENT_PCA_SAMPLE_SIZE", 50000) if sample_size is None else int(sample_size)
    max_sample = max(int(max_sample), 1)
    metadata: dict[str, object] = {
        "standardization": "median_center_std_scale",
        "reduction": "none",
        "embedding_dim_raw": raw_dim,
        "embedding_dim_used": raw_dim,
        "pca_components_requested": target,
        "pca_sample_size_requested": max_sample,
        "pca_sample_size_used": 0,
        "pca_variance_explained_estimate": None,
        "uses_spatial_coordinates": False,
        "uses_ot_costs": False,
    }
    if target <= 0 or raw_dim <= target or n_items <= max(target + 1, 3):
        return z.astype(np.float32), metadata

    rng = np.random.default_rng(int(random_state))
    fit_n = min(max_sample, n_items)
    if fit_n < n_items:
        fit_idx = np.sort(rng.choice(n_items, size=fit_n, replace=False).astype(np.int64))
        sample = z[fit_idx]
    else:
        sample = z
    center = sample.mean(axis=0, keepdims=True, dtype=np.float64)
    sample_centered = sample.astype(np.float64, copy=False) - center
    n_components = max(1, min(int(target), int(sample_centered.shape[0]) - 1, raw_dim))
    _, singular_values, vt = randomized_svd(
        sample_centered,
        n_components=n_components,
        random_state=int(random_state),
    )
    projected = (z.astype(np.float64, copy=False) - center) @ vt.T
    scale = projected.std(axis=0, keepdims=True)
    projected = projected / np.maximum(scale, 1e-8)
    total_variance = float(np.sum(np.var(sample_centered, axis=0)))
    explained = float(np.sum(singular_values * singular_values) / max(float(sample_centered.shape[0] - 1), 1.0))
    explained_fraction = explained / max(total_variance, 1e-12)
    metadata.update(
        {
            "reduction": "sampled_pca_whiten",
            "embedding_dim_used": int(projected.shape[1]),
            "pca_sample_size_used": int(sample.shape[0]),
            "pca_variance_explained_estimate": float(min(max(explained_fraction, 0.0), 1.0)),
        }
    )
    return projected.astype(np.float32), metadata


def _sample_rows(n_rows: int, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if int(max_rows) <= 0 or int(max_rows) >= int(n_rows):
        return np.arange(int(n_rows), dtype=np.int32)
    return np.sort(rng.choice(int(n_rows), size=int(max_rows), replace=False).astype(np.int32))


def _classical_mds(distance_matrix: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = int(d.shape[0])
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if n < 2:
        return np.zeros((n, max(int(n_components), 1)), dtype=np.float32)
    dim = max(1, min(int(n_components), n - 1))
    d2 = d * d
    b = -0.5 * (d2 - d2.mean(axis=0, keepdims=True) - d2.mean(axis=1, keepdims=True) + float(d2.mean()))
    if n <= 512:
        eigvals, eigvecs = np.linalg.eigh(b)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order][:dim]
        eigvecs = eigvecs[:, order][:, :dim]
        positive = np.maximum(eigvals, 0.0)
        coords = eigvecs * np.sqrt(positive)[None, :]
    else:
        u, s, _ = randomized_svd(b, n_components=dim, random_state=random_state)
        coords = u * np.sqrt(np.maximum(s, 0.0))[None, :]
    if coords.shape[1] < dim:
        pad = np.zeros((n, dim - coords.shape[1]), dtype=np.float64)
        coords = np.concatenate([coords, pad], axis=1)
    coords = np.asarray(coords, dtype=np.float64)
    coords -= coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    coords = coords / np.maximum(scale, 1e-8)
    return coords.astype(np.float32)


def _squared_distances_to_centers(embedding: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = np.asarray(embedding, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError("embedding and centers must be 2D matrices with the same feature dimension.")
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    c_norm = np.sum(c * c, axis=1, keepdims=True).T
    d2 = x_norm + c_norm - 2.0 * (x @ c.T)
    return np.maximum(d2, 0.0).astype(np.float32)


def _centers_from_labels(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int,
    fallback_centers: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(embedding, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    centers = np.zeros((int(n_clusters), x.shape[1]), dtype=np.float32)
    fallback = np.asarray(fallback_centers, dtype=np.float32) if fallback_centers is not None else None
    global_center = x.mean(axis=0).astype(np.float32) if x.size else np.zeros(x.shape[1], dtype=np.float32)
    for cluster_id in range(int(n_clusters)):
        members = x[labels_arr == int(cluster_id)]
        if members.size:
            centers[cluster_id] = members.mean(axis=0).astype(np.float32)
        elif fallback is not None and cluster_id < fallback.shape[0]:
            centers[cluster_id] = fallback[cluster_id].astype(np.float32)
        else:
            centers[cluster_id] = global_center
    return centers


def _fit_kmeans_on_standardized_embedding(
    embedding: np.ndarray,
    *,
    n_clusters: int,
    n_init: int,
    min_cluster_size: int,
    random_state: int,
) -> dict[str, object]:
    x = np.asarray(embedding, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embedding must be a 2D matrix.")
    if x.shape[0] < int(n_clusters):
        raise ValueError("n_clusters exceeds the number of subregion latent embeddings.")
    if x.shape[0] > 50000:
        max_fit_rows = 20000
        if x.shape[0] > max_fit_rows:
            rng = np.random.default_rng(int(random_state))
            fit_idx = np.sort(rng.choice(int(x.shape[0]), size=max_fit_rows, replace=False).astype(np.int64))
            fit_x = x[fit_idx]
        else:
            fit_x = x
        batch_size = min(max(int(n_clusters) * 512, 4096), max(int(fit_x.shape[0]), 4096))
        model = MiniBatchKMeans(
            n_clusters=int(n_clusters),
            n_init=int(n_init),
            batch_size=int(batch_size),
            random_state=int(random_state),
            max_iter=10,
            max_no_improvement=3,
            reassignment_ratio=0.0,
        )
        model.fit(fit_x)
        labels = model.predict(x).astype(np.int32)
    else:
        model = KMeans(n_clusters=int(n_clusters), n_init=int(n_init), random_state=int(random_state))
        labels = model.fit_predict(x).astype(np.int32)
    costs = _squared_distances_to_centers(x, model.cluster_centers_)
    labels, forced = repair_labels_to_minimum_size(
        labels,
        costs,
        n_clusters=int(n_clusters),
        min_cluster_size=int(min_cluster_size),
    )
    centers = _centers_from_labels(
        x,
        labels,
        n_clusters=int(n_clusters),
        fallback_centers=model.cluster_centers_,
    )
    costs = _squared_distances_to_centers(x, centers)
    inertia = float(np.sum(costs[np.arange(labels.shape[0]), labels]))
    return {
        "labels": labels.astype(np.int32),
        "argmin_labels": costs.argmin(axis=1).astype(np.int32),
        "costs": costs.astype(np.float32),
        "forced_label_mask": forced.astype(bool),
        "centers": centers.astype(np.float32),
        "inertia": inertia,
    }


def fit_kmeans_on_latent_embeddings(
    latent_embeddings: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 20,
    min_cluster_size: int = 1,
    random_state: int = 1337,
) -> dict[str, object]:
    embedding, preprocessing = prepare_latent_clustering_embedding(
        latent_embeddings,
        random_state=int(random_state),
    )
    result = _fit_kmeans_on_standardized_embedding(
        embedding,
        n_clusters=int(n_clusters),
        n_init=int(n_init),
        min_cluster_size=int(min_cluster_size),
        random_state=int(random_state),
    )
    result["standardized_embeddings"] = embedding.astype(np.float32)
    result["latent_preprocessing"] = preprocessing
    return result


def _gap_one_standard_error_k(rows: list[dict[str, object]]) -> int | None:
    scored: list[tuple[int, float, float]] = []
    for row in rows:
        gap = row.get("gap")
        if gap is None:
            continue
        gap_f = float(gap)
        if not np.isfinite(gap_f):
            continue
        se = row.get("gap_se")
        se_f = float(se) if se is not None and np.isfinite(float(se)) else 0.0
        scored.append((int(row["n_clusters"]), gap_f, se_f))
    if not scored:
        return None
    best_k, best_gap, best_se = max(scored, key=lambda item: (item[1], -item[0]))
    threshold = float(best_gap) - float(best_se)
    eligible = [k for k, gap, _ in scored if gap >= threshold]
    return int(min(eligible)) if eligible else int(best_k)


def _metric_rank_sums(
    rows: list[dict[str, object]],
    metric_specs: list[tuple[str, bool, float]],
) -> dict[int, float]:
    rank_sums = {int(row["n_clusters"]): 0.0 for row in rows}
    for metric, higher_is_better, weight in metric_specs:
        scored: list[tuple[float, int]] = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            value_f = float(value)
            if np.isfinite(value_f):
                scored.append((value_f, int(row["n_clusters"])))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=bool(higher_is_better))
        for rank, (_, k) in enumerate(scored, start=1):
            rank_sums[int(k)] += float(weight) * float(rank)
    return rank_sums


def _pairwise_label_stability(labels_by_seed: list[np.ndarray]) -> tuple[float | None, float | None, int]:
    if len(labels_by_seed) < 2:
        return None, None, 0
    ari_values: list[float] = []
    nmi_values: list[float] = []
    for left, right in combinations(labels_by_seed, 2):
        ari_values.append(float(adjusted_rand_score(left, right)))
        nmi_values.append(float(normalized_mutual_info_score(left, right)))
    return float(np.mean(ari_values)), float(np.mean(nmi_values)), len(ari_values)


def _summarize_counts(labels: np.ndarray, n_clusters: int) -> dict[str, object]:
    counts = np.bincount(np.asarray(labels, dtype=np.int32), minlength=int(n_clusters)).astype(np.int64)
    return {
        "cluster_size_min": int(counts.min()) if counts.size else 0,
        "cluster_size_max": int(counts.max()) if counts.size else 0,
        "cluster_size_median": float(np.median(counts)) if counts.size else 0.0,
        "cluster_size_q25": float(np.quantile(counts, 0.25)) if counts.size else 0.0,
        "cluster_size_q75": float(np.quantile(counts, 0.75)) if counts.size else 0.0,
    }


def comprehensive_select_k_from_latent_embeddings(
    latent_embeddings: np.ndarray,
    *,
    candidate_n_clusters: tuple[int, ...] | list[int] | str | None,
    fallback_n_clusters: int = 15,
    seeds: tuple[int, ...] | list[int] | str | None = None,
    n_init: int = 10,
    min_cluster_size: int = 1,
    gap_references: int = 16,
    bootstrap_repeats: int = 12,
    bootstrap_fraction: float = 0.8,
    max_silhouette_subregions: int = 0,
    random_state: int = 1337,
    include_labels: bool = False,
) -> dict[str, object]:
    """Run multi-criterion K selection on pooled subregion latent embeddings only.

    This selector intentionally ignores spatial coordinates, subregion centers, overlap
    graphs, and OT transport costs. It scores K on the pooled feature-distribution
    latent space used for subregion clustering.
    """

    embedding, preprocessing = prepare_latent_clustering_embedding(
        latent_embeddings,
        random_state=int(random_state),
    )
    n_subregions = int(embedding.shape[0])
    candidates = sanitize_candidate_n_clusters(
        candidate_n_clusters,
        fallback_n_clusters=int(fallback_n_clusters),
        n_subregions=n_subregions,
    )
    parsed_seeds = parse_candidate_n_clusters(seeds) if seeds is not None else ()
    if not parsed_seeds:
        parsed_seeds = tuple(int(random_state) + offset for offset in range(10))
    parsed_seeds = tuple(int(seed) for seed in parsed_seeds)
    if not parsed_seeds:
        parsed_seeds = (int(random_state),)

    rng = np.random.default_rng(int(random_state))
    score_idx = _sample_rows(n_subregions, int(max_silhouette_subregions), rng)
    score_embedding = embedding[score_idx]

    rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    best_labels_by_k: dict[int, np.ndarray] = {}
    best_centers_by_k: dict[int, np.ndarray] = {}

    for k in candidates:
        effective_min = effective_min_cluster_size(n_subregions, int(k), int(min_cluster_size))
        seed_results: list[dict[str, object]] = []
        seed_labels: list[np.ndarray] = []
        forced_fractions: list[float] = []
        for seed in parsed_seeds:
            result = _fit_kmeans_on_standardized_embedding(
                embedding,
                n_clusters=int(k),
                n_init=int(n_init),
                min_cluster_size=int(effective_min),
                random_state=int(seed) + 104729 * int(k),
            )
            labels = np.asarray(result["labels"], dtype=np.int32)
            forced = np.asarray(result["forced_label_mask"], dtype=bool)
            counts = np.bincount(labels, minlength=int(k)).astype(np.int64)
            inertia = float(result["inertia"])
            forced_fraction = float(forced.mean()) if forced.size else 0.0
            seed_results.append(result)
            seed_labels.append(labels)
            forced_fractions.append(forced_fraction)
            run_rows.append(
                {
                    "n_clusters": int(k),
                    "seed": int(seed),
                    "inertia": inertia,
                    "forced_repair_count": int(forced.sum()),
                    "forced_repair_fraction": forced_fraction,
                    "cluster_size_min": int(counts.min()) if counts.size else 0,
                    "cluster_size_max": int(counts.max()) if counts.size else 0,
                    "cluster_size_median": float(np.median(counts)) if counts.size else 0.0,
                    "passes_min_cluster_size": bool(counts.size > 0 and int(counts.min()) >= int(effective_min)),
                }
            )

        best_result = min(seed_results, key=lambda item: float(item["inertia"]))
        best_labels = np.asarray(best_result["labels"], dtype=np.int32)
        best_centers = np.asarray(best_result["centers"], dtype=np.float32)
        best_labels_by_k[int(k)] = best_labels.copy()
        best_centers_by_k[int(k)] = best_centers.copy()
        counts_summary = _summarize_counts(best_labels, int(k))
        silhouette = None
        ch = None
        db = None
        gap = None
        gap_se = None
        unique = np.unique(best_labels)
        if 1 < unique.size < best_labels.shape[0]:
            if score_embedding.shape[0] == embedding.shape[0]:
                score_labels = best_labels
            else:
                score_labels = best_labels[score_idx]
            if 1 < np.unique(score_labels).size < score_labels.shape[0]:
                silhouette = float(silhouette_score(score_embedding, score_labels, metric="euclidean"))
                ch = float(calinski_harabasz_score(score_embedding, score_labels))
                db = float(davies_bouldin_score(score_embedding, score_labels))
                gap, gap_se = _gap_statistic(
                    score_embedding,
                    score_labels,
                    n_clusters=int(k),
                    n_references=int(gap_references),
                    random_state=int(random_state) + 13007 * int(k),
                )
        seed_ari_mean, seed_nmi_mean, seed_pair_count = _pairwise_label_stability(seed_labels)

        bootstrap_ari_values: list[float] = []
        bootstrap_nmi_values: list[float] = []
        bootstrap_forced_fractions: list[float] = []
        if int(bootstrap_repeats) > 0:
            sample_size = int(round(float(bootstrap_fraction) * float(n_subregions)))
            sample_size = max(int(k), min(n_subregions, sample_size))
            for repeat_idx in range(int(bootstrap_repeats)):
                subset = np.sort(
                    rng.choice(n_subregions, size=sample_size, replace=False).astype(np.int32)
                )
                subset_min = max(
                    1,
                    int(round(float(effective_min) * float(sample_size) / float(n_subregions))),
                )
                subset_result = _fit_kmeans_on_standardized_embedding(
                    embedding[subset],
                    n_clusters=int(k),
                    n_init=max(1, min(int(n_init), 5)),
                    min_cluster_size=int(subset_min),
                    random_state=int(random_state) + 524287 * int(k) + int(repeat_idx),
                )
                full_costs = _squared_distances_to_centers(
                    embedding,
                    np.asarray(subset_result["centers"], dtype=np.float32),
                )
                full_labels = full_costs.argmin(axis=1).astype(np.int32)
                full_labels, forced = repair_labels_to_minimum_size(
                    full_labels,
                    full_costs,
                    n_clusters=int(k),
                    min_cluster_size=int(effective_min),
                )
                bootstrap_ari_values.append(float(adjusted_rand_score(best_labels, full_labels)))
                bootstrap_nmi_values.append(float(normalized_mutual_info_score(best_labels, full_labels)))
                bootstrap_forced_fractions.append(float(forced.mean()) if forced.size else 0.0)

        row = {
            "n_clusters": int(k),
            "silhouette": silhouette,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "gap": gap,
            "gap_se": gap_se,
            "best_seed_inertia": float(best_result["inertia"]),
            "mean_seed_inertia": float(np.mean([float(item["inertia"]) for item in seed_results])),
            "std_seed_inertia": float(np.std([float(item["inertia"]) for item in seed_results])),
            "seed_ari_mean": seed_ari_mean,
            "seed_nmi_mean": seed_nmi_mean,
            "seed_pair_count": int(seed_pair_count),
            "bootstrap_ari_mean": float(np.mean(bootstrap_ari_values)) if bootstrap_ari_values else None,
            "bootstrap_ari_std": float(np.std(bootstrap_ari_values)) if bootstrap_ari_values else None,
            "bootstrap_nmi_mean": float(np.mean(bootstrap_nmi_values)) if bootstrap_nmi_values else None,
            "bootstrap_nmi_std": float(np.std(bootstrap_nmi_values)) if bootstrap_nmi_values else None,
            "bootstrap_repeats": int(bootstrap_repeats),
            "bootstrap_fraction": float(bootstrap_fraction),
            "forced_repair_fraction_mean": float(np.mean(forced_fractions)) if forced_fractions else 0.0,
            "forced_repair_fraction_max": float(np.max(forced_fractions)) if forced_fractions else 0.0,
            "forced_repair_fraction_bootstrap_mean": (
                float(np.mean(bootstrap_forced_fractions)) if bootstrap_forced_fractions else None
            ),
            "requested_min_cluster_size": int(min_cluster_size),
            "effective_min_cluster_size": int(effective_min),
            "passes_min_cluster_size": bool(int(counts_summary["cluster_size_min"]) >= int(effective_min)),
            "cluster_size_scope": "all_subregions",
            "effective_min_cluster_size_scope": "all_subregions",
        }
        row.update(counts_summary)
        rows.append(row)

    gap_one_se = _gap_one_standard_error_k(rows)
    votes: dict[str, int] = {}
    for metric, higher in [
        ("silhouette", True),
        ("gap", True),
        ("calinski_harabasz", True),
        ("davies_bouldin", False),
        ("seed_ari_mean", True),
        ("bootstrap_ari_mean", True),
        ("forced_repair_fraction_mean", False),
    ]:
        vote = _best_metric_vote(rows, metric, higher_is_better=higher)
        if vote is not None:
            votes[metric] = int(vote)
    if gap_one_se is not None:
        votes["gap_one_standard_error"] = int(gap_one_se)

    metric_specs = [
        ("silhouette", True, 1.0),
        ("gap", True, 1.0),
        ("calinski_harabasz", True, 1.0),
        ("davies_bouldin", False, 1.0),
        ("seed_ari_mean", True, 1.25),
        ("bootstrap_ari_mean", True, 1.25),
        ("forced_repair_fraction_mean", False, 0.75),
    ]
    rank_sums = _metric_rank_sums(rows, metric_specs)
    if rank_sums:
        selected = min(rank_sums, key=lambda k: (rank_sums[k], k))
        rule = "weighted_rank_sum_internal_metrics_plus_seed_and_bootstrap_stability"
    elif votes:
        vote_counts = Counter(votes.values())
        max_votes = max(vote_counts.values())
        tied = {int(k) for k, count in vote_counts.items() if count == max_votes}
        selected = _rank_sum_tie_break(rows, tied) if len(tied) > 1 else next(iter(tied))
        rule = "fallback_majority_vote_with_rank_sum_tie_break"
    else:
        selected = int(min(candidates, key=lambda k: abs(int(k) - int(fallback_n_clusters))))
        rule = "fallback_to_nearest_configured_k"

    summary: dict[str, object] = {
        "enabled": True,
        "selected_k": int(selected),
        "recommended_k": int(selected),
        "candidate_n_clusters": [int(k) for k in candidates],
        "criterion_votes": votes,
        "rank_sums": {str(int(k)): float(v) for k, v in sorted(rank_sums.items())},
        "selection_rule": rule,
        "gap_one_standard_error_k": int(gap_one_se) if gap_one_se is not None else None,
        "scores": rows,
        "seed_runs": run_rows,
        "distance_source": "pooled_raw_member_feature_distribution_subregion_latent_embeddings",
        "distance_note": (
            "Comprehensive K selection was run on pooled raw-member feature-distribution "
            "subregion latent embeddings only. It does not use subregion centers, spatial "
            "coordinates, overlap graphs, canonical coordinates, or OT transport costs."
        ),
        "uses_spatial": False,
        "n_total_subregions": n_subregions,
        "embedding_dim": int(embedding.shape[1]),
        "embedding_dim_raw": int(preprocessing["embedding_dim_raw"]),
        "embedding_dim_used": int(preprocessing["embedding_dim_used"]),
        "latent_preprocessing": preprocessing,
        "seeds": [int(seed) for seed in parsed_seeds],
        "n_init": int(n_init),
        "gap_references": int(gap_references),
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_fraction": float(bootstrap_fraction),
        "max_silhouette_subregions": int(max_silhouette_subregions),
        "n_silhouette_subregions": int(score_embedding.shape[0]),
        "metric_score_subregions": int(score_embedding.shape[0]),
        "metric_scope": "sampled_subregions_for_silhouette_ch_db_gap; full_subregions_for_seed_and_bootstrap_stability",
        "min_cluster_size": int(min_cluster_size),
        "min_cluster_size_scope": "all_subregions",
    }
    if include_labels:
        summary["best_labels_by_k"] = best_labels_by_k
        summary["best_centers_by_k"] = best_centers_by_k
    return summary


def _cluster_precomputed_average(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=int(n_clusters), metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=int(n_clusters), affinity="precomputed", linkage="average")
    return model.fit_predict(distance_matrix).astype(np.int32)


def _cluster_distance_costs(distance_matrix: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    costs = np.zeros((d.shape[0], int(n_clusters)), dtype=np.float64)
    finite = d[np.isfinite(d)]
    fill = float(finite.max()) if finite.size else 1.0
    for cluster_id in range(int(n_clusters)):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            costs[:, cluster_id] = fill
        else:
            costs[:, cluster_id] = np.mean(d[:, members], axis=1)
    return costs


def _within_dispersion(embedding: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    total = 0.0
    for label in np.unique(labels):
        members = x[labels == label]
        if members.size == 0:
            continue
        center = members.mean(axis=0, keepdims=True)
        total += float(np.sum((members - center) ** 2))
    return max(total, 1e-12)


def _gap_statistic(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int,
    n_references: int,
    random_state: int,
) -> tuple[float | None, float | None]:
    if int(n_references) <= 0:
        return None, None
    x = np.asarray(embedding, dtype=np.float32)
    if x.shape[0] <= int(n_clusters):
        return None, None
    rng = np.random.default_rng(random_state)
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)
    observed = np.log(_within_dispersion(x, labels))
    ref_logs: list[float] = []
    for ref_idx in range(int(n_references)):
        ref = rng.uniform(0.0, 1.0, size=x.shape).astype(np.float32) * span[None, :] + mins[None, :]
        ref_labels = KMeans(
            n_clusters=int(n_clusters),
            n_init=5,
            random_state=int(random_state) + 7919 * (ref_idx + 1) + int(n_clusters),
        ).fit_predict(ref)
        ref_logs.append(float(np.log(_within_dispersion(ref, ref_labels))))
    ref_arr = np.asarray(ref_logs, dtype=np.float64)
    gap = float(ref_arr.mean() - observed)
    se = float(np.sqrt(1.0 + 1.0 / max(int(n_references), 1)) * ref_arr.std(ddof=1)) if ref_arr.size > 1 else 0.0
    return gap, se


def _best_metric_vote(rows: list[dict[str, object]], metric: str, *, higher_is_better: bool) -> int | None:
    values: list[tuple[float, int]] = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        value_f = float(value)
        if not np.isfinite(value_f):
            continue
        k = int(row["n_clusters"])
        values.append((value_f, k))
    if not values:
        return None
    if higher_is_better:
        best_value = max(value for value, _ in values)
        return min(k for value, k in values if np.isclose(value, best_value))
    best_value = min(value for value, _ in values)
    return min(k for value, k in values if np.isclose(value, best_value))


def _rank_sum_tie_break(rows: list[dict[str, object]], tied_ks: set[int]) -> int:
    rank_sums = {int(k): 0.0 for k in tied_ks}
    metric_specs = [
        ("silhouette", True),
        ("calinski_harabasz", True),
        ("davies_bouldin", False),
        ("gap", True),
    ]
    for metric, higher in metric_specs:
        scored = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            value_f = float(value)
            if np.isfinite(value_f):
                scored.append((value_f, int(row["n_clusters"])))
        if not scored:
            continue
        scored.sort(reverse=higher)
        for rank, (_, k) in enumerate(scored, start=1):
            if k in rank_sums:
                rank_sums[k] += float(rank)
    return min(rank_sums, key=lambda k: (rank_sums[k], k))


def select_k_from_ot_landmark_costs(
    landmark_costs: np.ndarray,
    *,
    candidate_n_clusters: tuple[int, ...] | list[int] | str | None,
    fallback_n_clusters: int,
    max_score_subregions: int = 2500,
    mds_components: int = 8,
    gap_references: int = 8,
    min_cluster_size: int = 1,
    random_state: int = 1337,
) -> dict[str, object]:
    costs = _finite_float_matrix(landmark_costs)
    rng = np.random.default_rng(random_state)
    score_idx = _sample_rows(costs.shape[0], int(max_score_subregions), rng)
    score_costs = costs[score_idx]
    candidates = sanitize_candidate_n_clusters(
        candidate_n_clusters,
        fallback_n_clusters=fallback_n_clusters,
        n_subregions=int(score_costs.shape[0]),
    )
    distance = ot_landmark_distance_matrix(score_costs)
    embedding = _classical_mds(distance, n_components=int(mds_components), random_state=int(random_state))

    rows: list[dict[str, object]] = []
    for k in candidates:
        labels = _cluster_precomputed_average(distance, n_clusters=int(k))
        scaled_requested_min = max(
            1,
            int(round(float(min_cluster_size) * float(labels.shape[0]) / float(costs.shape[0]))),
        )
        effective_min = effective_min_cluster_size(
            int(labels.shape[0]),
            int(k),
            scaled_requested_min,
        )
        _, raw_counts = np.unique(labels, return_counts=True)
        raw_min = int(raw_counts.min()) if raw_counts.size else 0
        forced_count = 0
        if raw_min < effective_min:
            labels, forced = repair_labels_to_minimum_size(
                labels,
                _cluster_distance_costs(distance, labels, int(k)),
                n_clusters=int(k),
                min_cluster_size=effective_min,
            )
            forced_count = int(forced.sum())
        unique, counts = np.unique(labels, return_counts=True)
        silhouette = None
        ch = None
        db = None
        gap = None
        gap_se = None
        if 1 < unique.size < labels.shape[0]:
            silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
            ch = float(calinski_harabasz_score(embedding, labels))
            db = float(davies_bouldin_score(embedding, labels))
            gap, gap_se = _gap_statistic(
                embedding,
                labels,
                n_clusters=int(k),
                n_references=int(gap_references),
                random_state=int(random_state) + 104729 * int(k),
            )
        rows.append(
            {
                "n_clusters": int(k),
                "silhouette": silhouette,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "gap": gap,
                "gap_se": gap_se,
                "cluster_size_min": int(counts.min()) if counts.size else 0,
                "cluster_size_max": int(counts.max()) if counts.size else 0,
                "cluster_size_median": float(np.median(counts)) if counts.size else 0.0,
                "cluster_size_scope": "scored_subregions",
                "raw_cluster_size_min": raw_min,
                "requested_min_cluster_size": int(min_cluster_size),
                "score_sample_min_cluster_size": int(scaled_requested_min),
                "score_effective_min_cluster_size": int(effective_min),
                "effective_min_cluster_size": int(effective_min),
                "effective_min_cluster_size_scope": "scored_subregions",
                "min_cluster_size_repair_count": forced_count,
                "passes_min_cluster_size": bool(counts.size > 0 and int(counts.min()) >= int(effective_min)),
            }
        )

    votes: dict[str, int] = {}
    for metric, higher in [
        ("silhouette", True),
        ("gap", True),
        ("calinski_harabasz", True),
        ("davies_bouldin", False),
    ]:
        vote = _best_metric_vote(rows, metric, higher_is_better=higher)
        if vote is not None:
            votes[metric] = int(vote)
    if votes:
        counts = Counter(votes.values())
        max_votes = max(counts.values())
        tied = {int(k) for k, count in counts.items() if count == max_votes}
        selected = _rank_sum_tie_break(rows, tied) if len(tied) > 1 else next(iter(tied))
        rule = "majority_vote_with_rank_sum_tie_break"
    else:
        selected = int(min(candidates, key=lambda k: abs(int(k) - int(fallback_n_clusters))))
        rule = "fallback_to_nearest_configured_k"

    return {
        "enabled": True,
        "selected_k": int(selected),
        "candidate_n_clusters": [int(k) for k in candidates],
        "criterion_votes": votes,
        "selection_rule": rule,
        "scores": rows,
        "distance_source": "pilot_ot_landmark_transport_cost_profiles",
        "distance_note": (
            "Pairwise subregion distances are Euclidean distances between row-standardized fused OT costs "
            "to pilot cluster prototypes; this is a scalable OT-landmark approximation rather than all-pairs OT."
        ),
        "mds_components": int(max(1, min(int(mds_components), max(int(score_costs.shape[0]) - 1, 1)))),
        "gap_references": int(gap_references),
        "min_cluster_size": int(min_cluster_size),
        "min_cluster_size_scope": "full_dataset_requested; per-score rows use sampled/effective fields when score_subregion_sampled=true",
        "n_total_subregions": int(costs.shape[0]),
        "n_scored_subregions": int(score_costs.shape[0]),
        "score_subregion_sampled": bool(score_costs.shape[0] != costs.shape[0]),
    }


def select_k_from_latent_embeddings(
    latent_embeddings: np.ndarray,
    *,
    candidate_n_clusters: tuple[int, ...] | list[int] | str | None,
    fallback_n_clusters: int,
    max_score_subregions: int = 2500,
    gap_references: int = 8,
    min_cluster_size: int = 1,
    random_state: int = 1337,
) -> dict[str, object]:
    embedding, preprocessing = prepare_latent_clustering_embedding(
        latent_embeddings,
        random_state=int(random_state),
    )
    rng = np.random.default_rng(random_state)
    score_idx = _sample_rows(embedding.shape[0], int(max_score_subregions), rng)
    score_embedding = embedding[score_idx]
    candidates = sanitize_candidate_n_clusters(
        candidate_n_clusters,
        fallback_n_clusters=fallback_n_clusters,
        n_subregions=int(score_embedding.shape[0]),
    )

    rows: list[dict[str, object]] = []
    for k in candidates:
        model = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state) + int(k))
        labels = model.fit_predict(score_embedding).astype(np.int32)
        costs = _squared_distances_to_centers(score_embedding, model.cluster_centers_)
        scaled_requested_min = max(
            1,
            int(round(float(min_cluster_size) * float(labels.shape[0]) / float(embedding.shape[0]))),
        )
        effective_min = effective_min_cluster_size(
            int(labels.shape[0]),
            int(k),
            scaled_requested_min,
        )
        _, raw_counts = np.unique(labels, return_counts=True)
        raw_min = int(raw_counts.min()) if raw_counts.size else 0
        forced_count = 0
        if raw_min < effective_min:
            labels, forced = repair_labels_to_minimum_size(
                labels,
                costs,
                n_clusters=int(k),
                min_cluster_size=effective_min,
            )
            forced_count = int(forced.sum())
        unique, counts = np.unique(labels, return_counts=True)
        silhouette = None
        ch = None
        db = None
        gap = None
        gap_se = None
        if 1 < unique.size < labels.shape[0]:
            silhouette = float(silhouette_score(score_embedding, labels, metric="euclidean"))
            ch = float(calinski_harabasz_score(score_embedding, labels))
            db = float(davies_bouldin_score(score_embedding, labels))
            gap, gap_se = _gap_statistic(
                score_embedding,
                labels,
                n_clusters=int(k),
                n_references=int(gap_references),
                random_state=int(random_state) + 104729 * int(k),
            )
        rows.append(
            {
                "n_clusters": int(k),
                "silhouette": silhouette,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "gap": gap,
                "gap_se": gap_se,
                "cluster_size_min": int(counts.min()) if counts.size else 0,
                "cluster_size_max": int(counts.max()) if counts.size else 0,
                "cluster_size_median": float(np.median(counts)) if counts.size else 0.0,
                "cluster_size_scope": "scored_subregions",
                "raw_cluster_size_min": raw_min,
                "requested_min_cluster_size": int(min_cluster_size),
                "score_sample_min_cluster_size": int(scaled_requested_min),
                "score_effective_min_cluster_size": int(effective_min),
                "effective_min_cluster_size": int(effective_min),
                "effective_min_cluster_size_scope": "scored_subregions",
                "min_cluster_size_repair_count": forced_count,
                "passes_min_cluster_size": bool(counts.size > 0 and int(counts.min()) >= int(effective_min)),
            }
        )

    votes: dict[str, int] = {}
    for metric, higher in [
        ("silhouette", True),
        ("gap", True),
        ("calinski_harabasz", True),
        ("davies_bouldin", False),
    ]:
        vote = _best_metric_vote(rows, metric, higher_is_better=higher)
        if vote is not None:
            votes[metric] = int(vote)
    if votes:
        counts = Counter(votes.values())
        max_votes = max(counts.values())
        tied = {int(k) for k, count in counts.items() if count == max_votes}
        selected = _rank_sum_tie_break(rows, tied) if len(tied) > 1 else next(iter(tied))
        rule = "majority_vote_with_rank_sum_tie_break"
    else:
        selected = int(min(candidates, key=lambda k: abs(int(k) - int(fallback_n_clusters))))
        rule = "fallback_to_nearest_configured_k"

    return {
        "enabled": True,
        "selected_k": int(selected),
        "candidate_n_clusters": [int(k) for k in candidates],
        "criterion_votes": votes,
        "selection_rule": rule,
        "scores": rows,
        "distance_source": "pooled_subregion_latent_embeddings",
        "distance_note": (
            "Candidate K values are scored on pooled raw-member feature-distribution subregion latent embeddings. "
            "This model-selection step does not use subregion centers, canonical spatial coordinates, "
            "overlap edges, or OT transport costs."
        ),
        "gap_references": int(gap_references),
        "min_cluster_size": int(min_cluster_size),
        "min_cluster_size_scope": "full_dataset_requested; per-score rows use sampled/effective fields when score_subregion_sampled=true",
        "n_total_subregions": int(embedding.shape[0]),
        "n_scored_subregions": int(score_embedding.shape[0]),
        "score_subregion_sampled": bool(score_embedding.shape[0] != embedding.shape[0]),
        "embedding_dim_raw": int(preprocessing["embedding_dim_raw"]),
        "embedding_dim_used": int(preprocessing["embedding_dim_used"]),
        "latent_preprocessing": preprocessing,
    }
