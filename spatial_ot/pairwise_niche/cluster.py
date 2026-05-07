from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

DEFAULT_CANDIDATE_N_CLUSTERS = tuple(range(5, 31))
DEFAULT_MODEL_SELECTION_METRICS = (
    "silhouette",
    "pseudo_calinski_harabasz",
    "medoid_davies_bouldin",
    "percentile_dunn",
)
_METRIC_ALIASES = {
    "calinski_harabasz": "pseudo_calinski_harabasz",
    "pseudo_ch": "pseudo_calinski_harabasz",
    "davies_bouldin": "medoid_davies_bouldin",
    "medoid_db": "medoid_davies_bouldin",
    "dunn": "percentile_dunn",
    "percentile_dunn_05_95": "percentile_dunn",
}
_MODEL_SELECTION_METRICS = frozenset(
    (*DEFAULT_MODEL_SELECTION_METRICS, "minimum_dunn")
)


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    assignment_score: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def _cluster_assignment_score(distance: np.ndarray, labels: np.ndarray) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    out = np.ones(y.shape[0], dtype=np.float32)
    unique = np.unique(y)
    if unique.size <= 1:
        return out
    for idx in range(y.shape[0]):
        same = y == y[idx]
        same[idx] = False
        if not np.any(same):
            out[idx] = 0.0
            continue
        same_mean = float(np.mean(d[idx, same]))
        other_means = [
            float(np.mean(d[idx, y == label]))
            for label in unique
            if int(label) != int(y[idx]) and np.any(y == label)
        ]
        if not other_means:
            continue
        nearest_other = min(other_means)
        out[idx] = float(
            np.clip((nearest_other - same_mean) / max(nearest_other, 1e-8), 0.0, 1.0)
        )
    return out


def _agglomerative(distance: np.ndarray, *, n_clusters: int) -> np.ndarray:
    kwargs = {
        "n_clusters": int(n_clusters),
        "linkage": "average",
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:  # pragma: no cover - older sklearn
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return np.asarray(model.fit_predict(distance), dtype=np.int32)


def _farthest_first_medoids(distance: np.ndarray, *, k: int) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float32)
    medoids = [int(np.argmin(np.mean(d, axis=1)))]
    while len(medoids) < int(k):
        nearest = np.min(d[:, medoids], axis=1)
        nearest[np.asarray(medoids, dtype=np.int64)] = -1.0
        medoids.append(int(np.argmax(nearest)))
    return np.asarray(medoids, dtype=np.int64)


def _kmedoids(
    distance: np.ndarray,
    *,
    n_clusters: int,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(distance, dtype=np.float32)
    k = min(max(int(n_clusters), 1), int(d.shape[0]))
    medoids = _farthest_first_medoids(d, k=k)
    labels = np.argmin(d[:, medoids], axis=1).astype(np.int32)
    for _ in range(max(int(max_iter), 1)):
        updated = medoids.copy()
        for cluster_id in range(k):
            members = np.flatnonzero(labels == cluster_id)
            if members.size == 0:
                continue
            intra = d[np.ix_(members, members)]
            updated[cluster_id] = int(members[np.argmin(np.sum(intra, axis=1))])
        new_labels = np.argmin(d[:, updated], axis=1).astype(np.int32)
        if np.array_equal(updated, medoids) and np.array_equal(new_labels, labels):
            break
        medoids = updated
        labels = new_labels
    return labels.astype(np.int32), medoids.astype(np.int64)


def _sanitize_candidate_clusters(
    values: list[int] | tuple[int, ...] | None,
    *,
    n_cells: int,
) -> list[int]:
    if not values:
        return []
    out = sorted({int(value) for value in values if 1 < int(value) < int(n_cells)})
    if not out:
        raise ValueError("candidate_n_clusters must include at least one value between 2 and n_cells-1.")
    return out


def _sanitize_candidate_resolutions(values: list[float] | tuple[float, ...] | None) -> list[float]:
    if not values:
        return []
    out = sorted({float(value) for value in values if float(value) > 0.0})
    if not out:
        raise ValueError("candidate_resolutions must include at least one positive value.")
    return out


def _sanitize_model_selection_metrics(
    values: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if not values:
        return list(DEFAULT_MODEL_SELECTION_METRICS)
    out: list[str] = []
    for value in values:
        metric = str(value).strip().lower().replace("-", "_")
        metric = _METRIC_ALIASES.get(metric, metric)
        if not metric:
            continue
        if metric not in _MODEL_SELECTION_METRICS:
            valid = ", ".join(sorted(_MODEL_SELECTION_METRICS))
            raise ValueError(f"model_selection_metrics must contain only: {valid}.")
        if metric not in out:
            out.append(metric)
    if not out:
        raise ValueError("model_selection_metrics must include at least one metric.")
    return out


def _finite_score(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _distance_silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int32)
    unique = np.unique(y)
    if unique.size < 2 or unique.size >= y.size:
        return float("-inf")
    try:
        return float(silhouette_score(distance, y, metric="precomputed"))
    except ValueError:
        return float("-inf")


def _cluster_medoids_from_labels(distance: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    d = np.asarray(distance, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    out: dict[int, int] = {}
    for label in np.unique(y):
        members = np.flatnonzero(y == int(label))
        if members.size == 0:
            continue
        intra = d[np.ix_(members, members)]
        out[int(label)] = int(members[np.argmin(np.sum(intra, axis=1))])
    return out


def _distance_calinski_harabasz(distance: np.ndarray, labels: np.ndarray) -> float:
    d = np.asarray(distance, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    n = int(y.size)
    unique = np.unique(y)
    k = int(unique.size)
    if k < 2 or k >= n:
        return float("-inf")
    d2 = d * d
    total = float(np.sum(d2) / max(2 * n, 1))
    within = 0.0
    for label in unique:
        members = np.flatnonzero(y == int(label))
        if members.size:
            within += float(np.sum(d2[np.ix_(members, members)]) / max(2 * members.size, 1))
    between = max(total - within, 0.0)
    denominator = max(within / max(n - k, 1), 1e-12)
    numerator = between / max(k - 1, 1)
    return float(numerator / denominator)


def _distance_davies_bouldin(distance: np.ndarray, labels: np.ndarray) -> float:
    d = np.asarray(distance, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    unique = np.unique(y)
    k = int(unique.size)
    if k < 2 or k >= y.size:
        return float("-inf")
    medoids_by_label = _cluster_medoids_from_labels(d, y)
    medoids = np.asarray([medoids_by_label[int(label)] for label in unique], dtype=np.int64)
    scatter = np.zeros(k, dtype=np.float64)
    for idx, label in enumerate(unique):
        members = np.flatnonzero(y == int(label))
        scatter[idx] = float(np.mean(d[members, medoids[idx]])) if members.size else 0.0
    separation = d[np.ix_(medoids, medoids)].astype(np.float64)
    ratios = np.full((k, k), -np.inf, dtype=np.float64)
    for row in range(k):
        for col in range(k):
            if row == col:
                continue
            sep = float(separation[row, col])
            if sep <= 1e-12:
                ratios[row, col] = np.inf
            else:
                ratios[row, col] = (scatter[row] + scatter[col]) / sep
    db = float(np.mean(np.max(ratios, axis=1)))
    return float("-inf") if not np.isfinite(db) else float(-db)


def _distance_dunn(distance: np.ndarray, labels: np.ndarray) -> float:
    d = np.asarray(distance, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    unique = np.unique(y)
    if unique.size < 2 or unique.size >= y.size:
        return float("-inf")
    max_diameter = 0.0
    min_inter = float("inf")
    for i, left in enumerate(unique):
        left_members = np.flatnonzero(y == int(left))
        if left_members.size > 1:
            max_diameter = max(
                max_diameter,
                float(np.max(d[np.ix_(left_members, left_members)])),
            )
        for right in unique[i + 1 :]:
            right_members = np.flatnonzero(y == int(right))
            if left_members.size and right_members.size:
                min_inter = min(
                    min_inter,
                    float(np.min(d[np.ix_(left_members, right_members)])),
                )
    if not np.isfinite(min_inter):
        return float("-inf")
    return float(min_inter / max(max_diameter, 1e-12))


def _distance_percentile_dunn(
    distance: np.ndarray,
    labels: np.ndarray,
    *,
    q_inter: float = 5.0,
    q_intra: float = 95.0,
) -> float:
    d = np.asarray(distance, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    unique = np.unique(y)
    if unique.size < 2 or unique.size >= y.size:
        return float("-inf")
    within: list[np.ndarray] = []
    between: list[np.ndarray] = []
    for pos, left in enumerate(unique):
        left_members = np.flatnonzero(y == int(left))
        if left_members.size > 1:
            block = d[np.ix_(left_members, left_members)]
            upper = block[np.triu_indices(left_members.size, k=1)]
            if upper.size:
                within.append(upper)
        for right in unique[pos + 1 :]:
            right_members = np.flatnonzero(y == int(right))
            if left_members.size and right_members.size:
                between.append(d[np.ix_(left_members, right_members)].reshape(-1))
    if not within or not between:
        return float("-inf")
    within_values = np.concatenate(within)
    between_values = np.concatenate(between)
    within_values = within_values[np.isfinite(within_values)]
    between_values = between_values[np.isfinite(between_values)]
    if within_values.size == 0 or between_values.size == 0:
        return float("-inf")
    intra = float(np.percentile(within_values, float(q_intra)))
    inter = float(np.percentile(between_values, float(q_inter)))
    return float(inter / max(intra, 1e-12))


def _cluster_size_metadata(labels: np.ndarray) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int32)
    _, counts = np.unique(y, return_counts=True)
    n = int(y.size)
    small_threshold = max(2, int(np.ceil(0.005 * max(n, 1))))
    return {
        "min_cluster_size": int(np.min(counts)) if counts.size else 0,
        "max_cluster_size": int(np.max(counts)) if counts.size else 0,
        "singleton_cluster_count": int(np.sum(counts == 1)),
        "singleton_fraction": float(np.sum(counts == 1) / max(counts.size, 1)),
        "small_cluster_threshold": int(small_threshold),
        "small_cluster_count": int(np.sum(counts < small_threshold)),
        "small_cluster_fraction": float(np.sum(counts < small_threshold) / max(counts.size, 1)),
        "cluster_sizes": [int(value) for value in counts.tolist()],
    }


def _within_between_distance_ratio(distance: np.ndarray, labels: np.ndarray) -> float | None:
    d = np.asarray(distance, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    within: list[np.ndarray] = []
    between: list[np.ndarray] = []
    unique = np.unique(y)
    for pos, left in enumerate(unique):
        left_members = np.flatnonzero(y == int(left))
        if left_members.size > 1:
            block = d[np.ix_(left_members, left_members)]
            upper = block[np.triu_indices(left_members.size, k=1)]
            if upper.size:
                within.append(upper)
        for right in unique[pos + 1 :]:
            right_members = np.flatnonzero(y == int(right))
            if left_members.size and right_members.size:
                between.append(d[np.ix_(left_members, right_members)].reshape(-1))
    if not within or not between:
        return None
    within_values = np.concatenate(within)
    between_values = np.concatenate(between)
    within_values = within_values[np.isfinite(within_values)]
    between_values = between_values[np.isfinite(between_values) & (between_values > 0)]
    if within_values.size == 0 or between_values.size == 0:
        return None
    return float(np.mean(within_values) / max(float(np.mean(between_values)), 1e-12))


def _model_selection_scores(
    distance: np.ndarray,
    labels: np.ndarray,
    *,
    metrics: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric in metrics:
        if metric == "silhouette":
            scores[metric] = _distance_silhouette(distance, labels)
        elif metric == "pseudo_calinski_harabasz":
            scores[metric] = _distance_calinski_harabasz(distance, labels)
        elif metric == "medoid_davies_bouldin":
            scores[metric] = _distance_davies_bouldin(distance, labels)
        elif metric == "percentile_dunn":
            scores[metric] = _distance_percentile_dunn(distance, labels)
        elif metric == "minimum_dunn":
            scores[metric] = _distance_dunn(distance, labels)
        else:  # pragma: no cover - protected by sanitizer.
            raise ValueError(f"Unknown model-selection metric: {metric}")
    return scores


def _rank_model_selection_results(
    results: list[dict[str, object]],
    *,
    metrics: list[str],
    tie_break_key: str,
) -> int:
    if not results:
        raise ValueError("model selection requires at least one candidate result.")
    for result in results:
        result["ranks"] = {}
    for metric in metrics:
        def score_for_rank(idx: int) -> float:
            scores = results[idx].get("scores", {})
            if not isinstance(scores, dict):
                return float("-inf")
            value = scores.get(metric)
            return float(value) if value is not None else float("-inf")

        order = sorted(
            range(len(results)),
            key=lambda idx: (
                -score_for_rank(idx),
                float(results[idx][tie_break_key]),
            ),
        )
        for rank, idx in enumerate(order, start=1):
            ranks = results[idx]["ranks"]
            assert isinstance(ranks, dict)
            ranks[metric] = int(rank)
    best_idx = 0
    best_key = (float("inf"), float("inf"))
    for idx, result in enumerate(results):
        ranks = result["ranks"]
        assert isinstance(ranks, dict)
        mean_rank = float(np.mean([float(ranks[metric]) for metric in metrics]))
        result["mean_rank"] = mean_rank
        key = (mean_rank, float(result[tie_break_key]))
        if key < best_key:
            best_key = key
            best_idx = idx
    return int(best_idx)


def _fit_fixed_k(
    distance: np.ndarray,
    *,
    method: str,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if method == "agglomerative":
        return _agglomerative(distance, n_clusters=int(n_clusters)), None
    if method in {"kmedoids", "pam"}:
        return _kmedoids(distance, n_clusters=int(n_clusters))
    raise ValueError("fixed-K model selection is only available for agglomerative and kmedoids.")


def _select_fixed_k_model(
    distance: np.ndarray,
    *,
    method: str,
    candidate_n_clusters: list[int],
    model_selection_metrics: list[str],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    results: list[dict[str, object]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    medoids_by_k: dict[int, np.ndarray | None] = {}
    best_labels: np.ndarray | None = None
    best_medoids: np.ndarray | None = None
    for k in candidate_n_clusters:
        labels, medoids = _fit_fixed_k(distance, method=method, n_clusters=int(k))
        scores = _model_selection_scores(
            distance,
            labels,
            metrics=model_selection_metrics,
        )
        labels_by_k[int(k)] = labels
        medoids_by_k[int(k)] = medoids
        results.append(
            {
                "n_clusters": int(k),
                "scores": {metric: _finite_score(value) for metric, value in scores.items()},
                "n_observed_clusters": int(np.unique(labels).size),
                "cluster_size_summary": _cluster_size_metadata(labels),
                "within_between_distance_ratio": _within_between_distance_ratio(
                    distance,
                    labels,
                ),
            }
        )
    best_idx = _rank_model_selection_results(
        results,
        metrics=model_selection_metrics,
        tie_break_key="n_clusters",
    )
    best_k = int(results[best_idx]["n_clusters"])
    best_labels = labels_by_k[best_k]
    best_medoids = medoids_by_k[best_k]
    assert best_labels is not None
    return best_labels, best_medoids, {
        "enabled": True,
        "criterion": "rank_ensemble",
        "metrics": list(model_selection_metrics),
        "candidate_n_clusters": [int(k) for k in candidate_n_clusters],
        "selected_n_clusters": int(best_k),
        "selected_mean_rank": float(results[best_idx]["mean_rank"]),
        "results": results,
    }


def _ordered_neighbors(distance: np.ndarray, row: int, limit: int | None = None) -> np.ndarray:
    order = np.argsort(distance[row], kind="stable")
    order = order[order != row]
    return order if limit is None else order[:limit]


def _knn_orders(distance: np.ndarray, *, k: int) -> list[np.ndarray]:
    n = int(distance.shape[0])
    neighbors = min(max(int(k), 1), max(n - 1, 1))
    return [_ordered_neighbors(distance, row, neighbors) for row in range(n)]


def ot_knn_affinity(
    distance: np.ndarray,
    *,
    k: int = 30,
    scaling: str = "local",
) -> sparse.csr_matrix:
    d = np.asarray(distance, dtype=np.float32)
    n = int(d.shape[0])
    orders = _knn_orders(d, k=int(k))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    finite = d[np.isfinite(d) & (d > 0)]
    global_sigma = max(float(np.median(finite)) if finite.size else 1.0, 1e-8)
    requested_scaling = str(scaling or "local").strip().lower()
    if requested_scaling not in {"local", "global"}:
        raise ValueError("ot_affinity_scaling must be local or global.")
    local_sigma = np.full(n, global_sigma, dtype=np.float32)
    if requested_scaling == "local" and n > 1:
        for row in range(n):
            ordered = _ordered_neighbors(d, row)
            positive = d[row, ordered][d[row, ordered] > 0]
            if positive.size:
                local_sigma[row] = float(positive[min(positive.size - 1, orders[row].size - 1)])
    for row, order in enumerate(orders):
        rows.extend([row] * int(order.size))
        cols.extend(int(col) for col in order)
        if requested_scaling == "global":
            data.extend(float(np.exp(-float(d[row, col]) / global_sigma)) for col in order)
        else:
            data.extend(
                float(
                    np.exp(
                        -(float(d[row, col]) ** 2)
                        / max(float(local_sigma[row]) * float(local_sigma[col]), 1e-8)
                    )
                )
                for col in order
            )
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    return graph.maximum(graph.T).tocsr()


def ot_knn_distance_graph(distance: np.ndarray, *, k: int = 30) -> sparse.csr_matrix:
    d = np.asarray(distance, dtype=np.float32)
    n = int(d.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, order in enumerate(_knn_orders(d, k=int(k))):
        rows.extend([row] * int(order.size))
        cols.extend(int(col) for col in order)
        data.extend(float(d[row, col]) for col in order)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    return graph.maximum(graph.T).tocsr()


def _leiden_ot_knn(
    distance: np.ndarray,
    *,
    k: int,
    affinity_scaling: str,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    try:
        import anndata as ad
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - optional runtime
        raise ImportError("leiden_ot_knn requires scanpy and leiden/igraph extras.") from exc
    graph = ot_knn_affinity(distance, k=int(k), scaling=str(affinity_scaling))
    tmp = ad.AnnData(X=np.zeros((distance.shape[0], 1), dtype=np.float32))
    tmp.obsp["connectivities"] = graph
    tmp.obsp["distances"] = ot_knn_distance_graph(distance, k=int(k))
    try:
        sc.tl.leiden(
            tmp,
            adjacency=graph,
            resolution=float(resolution),
            key_added="ot_niche",
            random_state=int(random_state),
        )
    except Exception as exc:  # pragma: no cover - optional runtime
        raise ImportError("leiden_ot_knn requires the optional leidenalg runtime.") from exc
    raw = tmp.obs["ot_niche"].astype(str).to_numpy()
    _, labels = np.unique(raw, return_inverse=True)
    return labels.astype(np.int32)


def cluster_from_distance(
    distance: np.ndarray,
    *,
    method: str = "agglomerative",
    n_clusters: int | None = None,
    candidate_n_clusters: list[int] | tuple[int, ...] | None = None,
    model_selection_metrics: list[str] | tuple[str, ...] | None = None,
    ot_knn: int = 30,
    ot_affinity_scaling: str = "local",
    leiden_resolution: float = 1.0,
    candidate_resolutions: list[float] | tuple[float, ...] | None = None,
    random_state: int = 1337,
) -> ClusterResult:
    d = np.asarray(distance, dtype=np.float32)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance must be a square matrix.")
    requested = str(method or "agglomerative").strip().lower()
    medoids = None
    model_selection: dict[str, object] | None = None
    selection_metrics = _sanitize_model_selection_metrics(model_selection_metrics)
    requested_candidate_n_clusters = candidate_n_clusters
    if (
        requested in {"agglomerative", "kmedoids", "pam"}
        and n_clusters is None
        and candidate_n_clusters is None
    ):
        requested_candidate_n_clusters = DEFAULT_CANDIDATE_N_CLUSTERS
    cluster_candidates = _sanitize_candidate_clusters(
        requested_candidate_n_clusters,
        n_cells=int(d.shape[0]),
    )
    resolution_candidates = _sanitize_candidate_resolutions(candidate_resolutions)
    if requested == "agglomerative":
        if cluster_candidates:
            labels, medoids, model_selection = _select_fixed_k_model(
                d,
                method=requested,
                candidate_n_clusters=cluster_candidates,
                model_selection_metrics=selection_metrics,
            )
        elif n_clusters is None:
            raise ValueError("agglomerative clustering requires --n-clusters.")
        else:
            labels = _agglomerative(d, n_clusters=int(n_clusters))
    elif requested in {"kmedoids", "pam"}:
        if cluster_candidates:
            labels, medoids, model_selection = _select_fixed_k_model(
                d,
                method=requested,
                candidate_n_clusters=cluster_candidates,
                model_selection_metrics=selection_metrics,
            )
        elif n_clusters is None:
            raise ValueError("kmedoids clustering requires --n-clusters.")
        else:
            labels, medoids = _kmedoids(d, n_clusters=int(n_clusters))
    elif requested == "leiden_ot_knn":
        if resolution_candidates:
            results: list[dict[str, object]] = []
            labels_by_resolution: dict[float, np.ndarray] = {}
            for resolution in resolution_candidates:
                labels_for_resolution = _leiden_ot_knn(
                    d,
                    k=int(ot_knn),
                    affinity_scaling=str(ot_affinity_scaling),
                    resolution=float(resolution),
                    random_state=int(random_state),
                )
                scores = _model_selection_scores(
                    d,
                    labels_for_resolution,
                    metrics=selection_metrics,
                )
                observed = int(np.unique(labels_for_resolution).size)
                labels_by_resolution[float(resolution)] = labels_for_resolution
                results.append(
                    {
                        "resolution": float(resolution),
                        "scores": {
                            metric: _finite_score(value) for metric, value in scores.items()
                        },
                        "n_observed_clusters": observed,
                        "cluster_size_summary": _cluster_size_metadata(labels_for_resolution),
                        "within_between_distance_ratio": _within_between_distance_ratio(
                            d,
                            labels_for_resolution,
                        ),
                    }
                )
            best_idx = _rank_model_selection_results(
                results,
                metrics=selection_metrics,
                tie_break_key="resolution",
            )
            best_resolution = float(results[best_idx]["resolution"])
            best_labels = labels_by_resolution[best_resolution]
            assert best_labels is not None
            labels = best_labels
            leiden_resolution = best_resolution
            model_selection = {
                "enabled": True,
                "criterion": "rank_ensemble",
                "metrics": list(selection_metrics),
                "candidate_resolutions": [float(value) for value in resolution_candidates],
                "selected_resolution": float(best_resolution),
                "selected_mean_rank": float(results[best_idx]["mean_rank"]),
                "results": results,
            }
        else:
            labels = _leiden_ot_knn(
                d,
                k=int(ot_knn),
                affinity_scaling=str(ot_affinity_scaling),
                resolution=float(leiden_resolution),
                random_state=int(random_state),
            )
    else:
        raise ValueError("cluster_method must be agglomerative, kmedoids, or leiden_ot_knn.")
    metadata: dict[str, object] = {
        "method": requested,
        "n_clusters": int(np.unique(labels).size),
        "assignment_score_type": "precomputed_distance_margin",
        "cluster_size_summary": _cluster_size_metadata(labels),
        "within_between_distance_ratio": _within_between_distance_ratio(d, labels),
    }
    if n_clusters is not None:
        metadata["requested_n_clusters"] = int(n_clusters)
    if cluster_candidates:
        metadata["candidate_n_clusters"] = [int(value) for value in cluster_candidates]
    if medoids is not None:
        metadata["medoid_indices"] = medoids.astype(int).tolist()
    if requested == "leiden_ot_knn":
        metadata["ot_knn"] = int(ot_knn)
        metadata["ot_affinity_scaling"] = str(ot_affinity_scaling)
        metadata["leiden_resolution"] = float(leiden_resolution)
        if resolution_candidates:
            metadata["candidate_resolutions"] = [
                float(value) for value in resolution_candidates
            ]
    if model_selection is not None:
        metadata["model_selection"] = model_selection
    return ClusterResult(
        labels=labels.astype(np.int32),
        assignment_score=_cluster_assignment_score(d, labels),
        metadata=metadata,
    )


def connected_components_by_label(
    neighbor_indices: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int32)
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    for row in range(indices.shape[0]):
        for col in indices[row]:
            if col >= 0 and col != row:
                rows.append(row)
                cols.append(int(col))
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(indices.shape[0], indices.shape[0]),
    ).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    instances = np.full(y.shape[0], -1, dtype=np.int32)
    names = np.empty(y.shape[0], dtype=object)
    next_id = 0
    for label in np.unique(y):
        members = np.flatnonzero(y == int(label))
        subgraph = graph[members][:, members]
        _, comp = connected_components(subgraph, directed=False, return_labels=True)
        for local_comp in np.unique(comp):
            rows_for_comp = members[comp == local_comp]
            instances[rows_for_comp] = next_id
            names[rows_for_comp] = f"ON{int(label)}_{int(local_comp)}"
            next_id += 1
    return instances, names
