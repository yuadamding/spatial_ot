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
    standardize_precomputed: bool = True
    allow_umap_as_feature: bool = False

    radius_um: float = 50.0
    max_neighbors: int = 32
    include_anchor: bool = True
    isolated_policy: Literal["zero_dummy", "anchor_fallback"] = "anchor_fallback"
    graph_kernel: Literal["gaussian", "uniform", "inverse_distance"] = "gaussian"
    cap_mode: Literal["radial_shell", "radial_shell_state"] = "radial_shell_state"
    cap_state_clusters: int = 16
    radial_shells: int = 3

    expression_weight: float = 1.0
    spatial_weight: float = 0.25
    distance_weight: float = 0.10
    ground_cost_normalization: Literal["none", "dimension", "sampled_median"] = (
        "sampled_median"
    )
    ground_cost_sample_pairs: int = 10000
    anchor_weight: float = 0.0

    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 50
    distance_mode: Literal[
        "sinkhorn",
        "debiased_entropic_transport",
        "sinkhorn_divergence",
        "fused_gromov_wasserstein",
    ] = "debiased_entropic_transport"
    fgw_alpha: float = 0.25
    fgw_iters: int = 5
    fgw_node_feature_mode: Literal[
        "expression_only",
        "expression_plus_radial",
        "full_token",
    ] = "expression_only"
    fgw_structure_mode: Literal[
        "complete_euclidean",
        "local_knn_shortest_path",
        "radius_graph_shortest_path",
        "binary_edge_distance",
        "adjacency",
    ] = "local_knn_shortest_path"
    fgw_structure_knn: int = 6
    fgw_structure_radius_fraction: float = 0.5
    fgw_structure_normalization: Literal["none", "sampled_median"] = "sampled_median"
    fgw_structure_sample_pairs: int = 10000
    pairwise_mode: Literal["exact", "exact_blockwise"] = "exact_blockwise"
    block_size: int = 64
    device: str = "auto"
    max_exact_cells: int = 5000
    max_ot_work_units: float = 5e11
    max_fgw_work_units: float = 1e12
    force_large_exact_ot: bool = False
    target_block_memory_gib: float | None = None
    distance_store: Literal["auto", "h5ad", "npy_memmap"] = "auto"

    cluster_method: Literal["agglomerative", "kmedoids", "leiden_ot_knn"] = "agglomerative"
    n_clusters: int | None = None
    candidate_n_clusters: tuple[int, ...] | None = None
    model_selection_metrics: tuple[
        Literal[
            "silhouette",
            "pseudo_calinski_harabasz",
            "medoid_davies_bouldin",
            "percentile_dunn",
            "minimum_dunn",
        ],
        ...,
    ] = (
        "silhouette",
        "pseudo_calinski_harabasz",
        "medoid_davies_bouldin",
        "percentile_dunn",
    )
    ot_knn: int = 30
    ot_affinity_scaling: Literal["local", "global"] = "local"
    leiden_resolution: float = 1.0
    candidate_resolutions: tuple[float, ...] | None = None
    instance_radius_um: float | None = None
    instance_max_neighbors: int = 512
    seed: int = 1337

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
