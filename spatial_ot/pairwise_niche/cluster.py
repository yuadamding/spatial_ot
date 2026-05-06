from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering


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
        same_mean = float(np.mean(d[idx, same])) if np.any(same) else 0.0
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


def ot_knn_affinity(distance: np.ndarray, *, k: int = 30) -> sparse.csr_matrix:
    d = np.asarray(distance, dtype=np.float32)
    n = int(d.shape[0])
    neighbors = min(max(int(k), 1), max(n - 1, 1))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    finite = d[np.isfinite(d) & (d > 0)]
    sigma = float(np.median(finite)) if finite.size else 1.0
    sigma = max(sigma, 1e-8)
    for row in range(n):
        order = np.argsort(d[row], kind="stable")
        order = order[order != row][:neighbors]
        rows.extend([row] * int(order.size))
        cols.extend(int(col) for col in order)
        data.extend(float(np.exp(-float(d[row, col]) / sigma)) for col in order)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    return graph.maximum(graph.T).tocsr()


def _leiden_ot_knn(
    distance: np.ndarray,
    *,
    k: int,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    try:
        import anndata as ad
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - optional runtime
        raise ImportError("leiden_ot_knn requires scanpy and leiden/igraph extras.") from exc
    graph = ot_knn_affinity(distance, k=int(k))
    tmp = ad.AnnData(X=np.zeros((distance.shape[0], 1), dtype=np.float32))
    tmp.obsp["connectivities"] = graph
    tmp.obsp["distances"] = sparse.csr_matrix(distance)
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
    ot_knn: int = 30,
    leiden_resolution: float = 1.0,
    random_state: int = 1337,
) -> ClusterResult:
    d = np.asarray(distance, dtype=np.float32)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance must be a square matrix.")
    requested = str(method or "agglomerative").strip().lower()
    medoids = None
    if requested == "agglomerative":
        if n_clusters is None:
            raise ValueError("agglomerative clustering requires --n-clusters.")
        labels = _agglomerative(d, n_clusters=int(n_clusters))
    elif requested in {"kmedoids", "pam"}:
        if n_clusters is None:
            raise ValueError("kmedoids clustering requires --n-clusters.")
        labels, medoids = _kmedoids(d, n_clusters=int(n_clusters))
    elif requested == "leiden_ot_knn":
        labels = _leiden_ot_knn(
            d,
            k=int(ot_knn),
            resolution=float(leiden_resolution),
            random_state=int(random_state),
        )
    else:
        raise ValueError("cluster_method must be agglomerative, kmedoids, or leiden_ot_knn.")
    metadata: dict[str, object] = {
        "method": requested,
        "n_clusters": int(np.unique(labels).size),
        "assignment_score_type": "precomputed_distance_margin",
    }
    if n_clusters is not None:
        metadata["requested_n_clusters"] = int(n_clusters)
    if medoids is not None:
        metadata["medoid_indices"] = medoids.astype(int).tolist()
    if requested == "leiden_ot_knn":
        metadata["ot_knn"] = int(ot_knn)
        metadata["leiden_resolution"] = float(leiden_resolution)
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
