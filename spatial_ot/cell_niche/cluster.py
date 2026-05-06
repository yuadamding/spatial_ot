from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .graph import NeighborhoodGraph


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    confidence: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def _pairwise_sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    return np.maximum(
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1, keepdims=True).T
        - 2.0 * (a @ b.T),
        0.0,
    ).astype(np.float32)


def _kmeans_cluster(
    embedding: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> ClusterResult:
    x = np.asarray(embedding, dtype=np.float32)
    k = min(max(int(n_clusters), 1), int(x.shape[0]))
    model = MiniBatchKMeans(
        n_clusters=k,
        n_init=max(int(n_init), 1),
        batch_size=min(max(4096, k * 256), max(int(x.shape[0]), 4096)),
        random_state=int(random_state),
    )
    labels = np.asarray(model.fit_predict(x), dtype=np.int32)
    distances = _pairwise_sqdist(x, np.asarray(model.cluster_centers_, dtype=np.float32))
    if k <= 1:
        confidence = np.ones(int(x.shape[0]), dtype=np.float32)
    else:
        sorted_dist = np.sort(distances, axis=1)
        margin = (sorted_dist[:, 1] - sorted_dist[:, 0]) / np.maximum(sorted_dist[:, 1], 1e-8)
        confidence = np.clip(margin, 0.0, 1.0).astype(np.float32)
    return ClusterResult(
        labels=labels,
        confidence=confidence,
        metadata={
            "method": "kmeans",
            "n_clusters": int(k),
            "inertia": float(model.inertia_),
            "n_init": int(n_init),
        },
    )


def _leiden_once(
    embedding: np.ndarray,
    *,
    resolution: float,
    n_neighbors: int,
    random_state: int,
) -> ClusterResult:
    try:
        import anndata as ad
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - exercised only without scanpy
        raise ImportError("Leiden clustering requires scanpy and its Leiden extras.") from exc

    x = np.asarray(embedding, dtype=np.float32)
    tmp = ad.AnnData(X=x)
    neighbors = min(max(int(n_neighbors), 2), max(int(x.shape[0]) - 1, 1))
    sc.pp.neighbors(tmp, n_neighbors=neighbors, random_state=int(random_state))
    try:
        sc.tl.leiden(
            tmp,
            resolution=float(resolution),
            key_added="spatial_niche",
            random_state=int(random_state),
        )
    except Exception as exc:  # pragma: no cover - depends on optional leidenalg
        raise ImportError(
            "Leiden clustering requires the optional igraph/leidenalg runtime. "
            "Use --cluster-method kmeans if those packages are unavailable."
        ) from exc
    raw = tmp.obs["spatial_niche"].astype(str).to_numpy()
    _, labels = np.unique(raw, return_inverse=True)
    labels = labels.astype(np.int32)
    graph = tmp.obsp["connectivities"].tocsr()
    confidence = neighbor_label_agreement(graph, labels)
    return ClusterResult(
        labels=labels,
        confidence=confidence,
        metadata={
            "method": "leiden",
            "resolution": float(resolution),
            "n_neighbors": int(neighbors),
            "n_clusters": int(np.unique(labels).size),
        },
    )


def _safe_silhouette(embedding: np.ndarray, labels: np.ndarray, *, seed: int) -> float | None:
    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= labels.shape[0]:
        return None
    x = np.asarray(embedding, dtype=np.float32)
    max_cells = min(int(x.shape[0]), 5000)
    if x.shape[0] > max_cells:
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(int(x.shape[0]), size=max_cells, replace=False))
        x = x[idx]
        labels = labels[idx]
    try:
        return float(silhouette_score(x, labels))
    except Exception:
        return None


def cluster_embeddings(
    embedding: np.ndarray,
    *,
    method: str = "kmeans",
    n_clusters: int | None = None,
    resolution: float = 1.0,
    candidate_resolutions: tuple[float, ...] | list[float] | None = None,
    n_neighbors: int = 15,
    n_init: int = 10,
    random_state: int = 1337,
) -> ClusterResult:
    x = np.asarray(embedding, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embedding must be a 2D matrix.")
    requested = str(method or "kmeans").strip().lower()
    if requested in {"kmeans", "mini_batch_kmeans", "minibatch_kmeans"}:
        return _kmeans_cluster(
            x,
            n_clusters=15 if n_clusters is None else int(n_clusters),
            random_state=int(random_state),
            n_init=int(n_init),
        )
    if requested == "leiden":
        resolutions = (
            tuple(float(value) for value in candidate_resolutions)
            if candidate_resolutions
            else (float(resolution),)
        )
        candidates: list[tuple[float | None, ClusterResult]] = []
        for res in resolutions:
            result = _leiden_once(
                x,
                resolution=float(res),
                n_neighbors=int(n_neighbors),
                random_state=int(random_state),
            )
            score = _safe_silhouette(x, result.labels, seed=int(random_state))
            candidates.append((score, result))
        valid = [item for item in candidates if item[0] is not None]
        best = max(valid, key=lambda item: float(item[0])) if valid else candidates[0]
        best_result = best[1]
        metadata = dict(best_result.metadata)
        metadata["candidate_resolutions"] = [float(value) for value in resolutions]
        metadata["candidate_silhouette"] = [
            None if score is None else float(score) for score, _ in candidates
        ]
        metadata["selected_silhouette"] = None if best[0] is None else float(best[0])
        return ClusterResult(
            labels=best_result.labels,
            confidence=best_result.confidence,
            metadata=metadata,
        )
    raise ValueError("cluster method must be kmeans or leiden.")


def neighbor_label_agreement(graph: sparse.spmatrix, labels: np.ndarray) -> np.ndarray:
    csr = graph.tocsr()
    y = np.asarray(labels, dtype=np.int32)
    out = np.ones(y.shape[0], dtype=np.float32)
    for row in range(int(csr.shape[0])):
        start, stop = int(csr.indptr[row]), int(csr.indptr[row + 1])
        if start == stop:
            continue
        cols = csr.indices[start:stop]
        weights = np.asarray(csr.data[start:stop], dtype=np.float64)
        denom = float(np.sum(weights))
        if denom <= 1e-12:
            continue
        out[row] = float(np.sum(weights[y[cols] == y[row]]) / denom)
    return out.astype(np.float32)


def connected_components_by_label(
    graph: NeighborhoodGraph,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int32)
    adjacency = graph.connectivities.maximum(graph.connectivities.T).tocsr()
    instances = np.full(y.shape[0], -1, dtype=np.int32)
    names = np.empty(y.shape[0], dtype=object)
    next_id = 0
    for label in np.unique(y):
        rows = np.flatnonzero(y == int(label))
        if rows.size == 0:
            continue
        sub = adjacency[rows][:, rows]
        n_components, component_labels = connected_components(
            sub, directed=False, return_labels=True
        )
        for component in range(int(n_components)):
            mask = component_labels == component
            cell_rows = rows[mask]
            instances[cell_rows] = int(next_id)
            instance_name = f"N{int(label)}_I{int(next_id)}"
            names[cell_rows] = instance_name
            next_id += 1
    missing = instances < 0
    if np.any(missing):
        for row in np.flatnonzero(missing):
            instances[row] = int(next_id)
            names[row] = f"N{int(y[row])}_I{int(next_id)}"
            next_id += 1
    return instances.astype(np.int32), names.astype(object)


__all__ = [
    "ClusterResult",
    "cluster_embeddings",
    "connected_components_by_label",
    "neighbor_label_agreement",
]
