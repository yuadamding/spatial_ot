from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ClusterResult": ("pairwise_niche", "ClusterResult"),
    "ExpressionEmbedding": ("pairwise_niche", "ExpressionEmbedding"),
    "ExpressionEmbeddingState": ("pairwise_niche", "ExpressionEmbeddingState"),
    "HIGH_CONTRAST_PALETTE": ("pairwise_niche", "HIGH_CONTRAST_PALETTE"),
    "LocalMeasureSet": ("pairwise_niche", "LocalMeasureSet"),
    "PairwiseNicheConfig": ("pairwise_niche", "PairwiseNicheConfig"),
    "build_instance_neighbor_indices": (
        "pairwise_niche",
        "build_instance_neighbor_indices",
    ),
    "build_local_measures": ("pairwise_niche", "build_local_measures"),
    "batched_fused_gromov_wasserstein_cost": (
        "pairwise_niche",
        "batched_fused_gromov_wasserstein_cost",
    ),
    "assign_high_contrast_colors": ("pairwise_niche", "assign_high_contrast_colors"),
    "cluster_from_distance": ("pairwise_niche", "cluster_from_distance"),
    "compute_pairwise_ot_distance_matrix": (
        "pairwise_niche",
        "compute_pairwise_ot_distance_matrix",
    ),
    "distribute_pooled_feature_cache_to_inputs": (
        "pooling",
        "distribute_pooled_feature_cache_to_inputs",
    ),
    "estimate_pairwise_ot_work": ("pairwise_niche", "estimate_pairwise_ot_work"),
    "estimate_pairwise_fgw_work": ("pairwise_niche", "estimate_pairwise_fgw_work"),
    "fit_expression_embedding": ("pairwise_niche", "fit_expression_embedding"),
    "load_expression_embedding_state": (
        "pairwise_niche",
        "load_expression_embedding_state",
    ),
    "fused_gromov_wasserstein_block": (
        "pairwise_niche",
        "fused_gromov_wasserstein_block",
    ),
    "pool_h5ad_files": ("pooling", "pool_h5ad_files"),
    "pool_h5ads_in_directory": ("pooling", "pool_h5ads_in_directory"),
    "prepare_h5ad_feature_cache": ("feature_source", "prepare_h5ad_feature_cache"),
    "run_pairwise_niche_on_h5ad": ("pairwise_niche", "run_pairwise_niche_on_h5ad"),
    "save_expression_embedding_state": (
        "pairwise_niche",
        "save_expression_embedding_state",
    ),
    "sinkhorn_ot_block": ("pairwise_niche", "sinkhorn_ot_block"),
}

__version__ = "3.0.9"
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
