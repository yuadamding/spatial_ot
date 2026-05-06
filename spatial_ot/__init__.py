from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CellNicheConfig": ("cell_niche", "CellNicheConfig"),
    "CellNicheDataset": ("cell_niche", "CellNicheDataset"),
    "ClusterResult": ("pairwise_niche", "ClusterResult"),
    "ExpressionEmbedding": ("pairwise_niche", "ExpressionEmbedding"),
    "ExpressionEmbeddingState": ("pairwise_niche", "ExpressionEmbeddingState"),
    "LocalMeasureSet": ("pairwise_niche", "LocalMeasureSet"),
    "OTDeepSHEModel": ("cell_niche", "OTDeepSHEModel"),
    "OTPrototypeHead": ("cell_niche", "OTPrototypeHead"),
    "PairwiseNicheConfig": ("pairwise_niche", "PairwiseNicheConfig"),
    "build_knn_graphs": ("cell_niche", "build_knn_graphs"),
    "build_instance_neighbor_indices": (
        "pairwise_niche",
        "build_instance_neighbor_indices",
    ),
    "build_local_measures": ("pairwise_niche", "build_local_measures"),
    "build_radius_graphs": ("cell_niche", "build_radius_graphs"),
    "batched_fused_gromov_wasserstein_cost": (
        "pairwise_niche",
        "batched_fused_gromov_wasserstein_cost",
    ),
    "cluster_embeddings": ("cell_niche", "cluster_embeddings"),
    "cluster_from_distance": ("pairwise_niche", "cluster_from_distance"),
    "compute_cell_heterogeneity_descriptors": (
        "cell_niche",
        "compute_cell_heterogeneity_descriptors",
    ),
    "compute_pairwise_ot_distance_matrix": (
        "pairwise_niche",
        "compute_pairwise_ot_distance_matrix",
    ),
    "distribute_pooled_feature_cache_to_inputs": (
        "pooling",
        "distribute_pooled_feature_cache_to_inputs",
    ),
    "fit_deepshe_embedding": ("cell_niche", "fit_deepshe_embedding"),
    "estimate_pairwise_ot_work": ("pairwise_niche", "estimate_pairwise_ot_work"),
    "fit_expression_embedding": ("pairwise_niche", "fit_expression_embedding"),
    "fit_state_codebook": ("cell_niche", "fit_state_codebook"),
    "fused_gromov_wasserstein_block": (
        "pairwise_niche",
        "fused_gromov_wasserstein_block",
    ),
    "pool_h5ad_files": ("pooling", "pool_h5ad_files"),
    "pool_h5ads_in_directory": ("pooling", "pool_h5ads_in_directory"),
    "prepare_feature_space": ("cell_niche", "prepare_feature_space"),
    "prepare_h5ad_feature_cache": ("feature_source", "prepare_h5ad_feature_cache"),
    "reduce_descriptor_embedding": ("cell_niche", "reduce_descriptor_embedding"),
    "run_cell_niche_on_h5ad": ("cell_niche", "run_cell_niche_on_h5ad"),
    "run_pairwise_niche_on_h5ad": ("pairwise_niche", "run_pairwise_niche_on_h5ad"),
    "save_expression_embedding_state": (
        "pairwise_niche",
        "save_expression_embedding_state",
    ),
    "sinkhorn_ot_block": ("pairwise_niche", "sinkhorn_ot_block"),
    "sinkhorn_balanced_distance": ("cell_niche", "sinkhorn_balanced_distance"),
    "sinkhorn_divergence": ("cell_niche", "sinkhorn_divergence"),
}

__version__ = "3.0.7"
__all__ = ["__version__", *_LAZY_EXPORTS]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'spatial_ot' has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)
