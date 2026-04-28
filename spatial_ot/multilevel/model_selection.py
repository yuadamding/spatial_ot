from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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
        effective_min = effective_min_cluster_size(
            int(labels.shape[0]),
            int(k),
            max(1, int(round(float(min_cluster_size) * float(labels.shape[0]) / float(costs.shape[0])))),
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
                "raw_cluster_size_min": raw_min,
                "effective_min_cluster_size": int(effective_min),
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
        "n_total_subregions": int(costs.shape[0]),
        "n_scored_subregions": int(score_costs.shape[0]),
        "score_subregion_sampled": bool(score_costs.shape[0] != costs.shape[0]),
    }
