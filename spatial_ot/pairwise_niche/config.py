from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class PairwiseNicheConfig:
    feature_obsm_key: str
    spatial_x_key: str
    spatial_y_key: str
    sample_obs_key: str = "sample_id"
    spatial_scale: float = 1.0

    embedding_method: Literal["pca", "svd", "precomputed"] = "pca"
    embedding_dim: int = 32
    expression_batch_key: str | None = None

    radius_um: float = 50.0
    max_neighbors: int = 32
    include_anchor: bool = True
    graph_kernel: Literal["gaussian", "uniform", "inverse_distance"] = "gaussian"
    cap_mode: Literal["radial_shell", "radial_shell_state"] = "radial_shell_state"
    cap_state_clusters: int = 16
    radial_shells: int = 3

    expression_weight: float = 1.0
    spatial_weight: float = 0.25
    distance_weight: float = 0.10
    anchor_weight: float = 0.25

    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 50
    distance_mode: Literal["sinkhorn", "sinkhorn_divergence"] = "sinkhorn_divergence"
    pairwise_mode: Literal["exact", "exact_blockwise"] = "exact_blockwise"
    block_size: int = 64
    device: str = "auto"
    max_exact_cells: int = 5000
    distance_store: Literal["auto", "h5ad", "npy_memmap"] = "auto"

    cluster_method: Literal["agglomerative", "kmedoids", "leiden_ot_knn"] = "agglomerative"
    n_clusters: int | None = None
    ot_knn: int = 30
    leiden_resolution: float = 1.0
    seed: int = 1337

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
