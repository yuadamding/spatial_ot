from __future__ import annotations

from .cluster import ClusterResult, cluster_from_distance, ot_knn_distance_graph
from .colors import HIGH_CONTRAST_PALETTE, assign_high_contrast_colors
from .config import PairwiseNicheConfig
from .distance_matrix import (
    assign_by_reference_medoids,
    choose_pairwise_block_size,
    compute_cross_ot_distance_matrix,
    compute_pairwise_ot_distance_matrix,
    estimate_cross_fgw_work,
    estimate_cross_ot_work,
    estimate_pairwise_fgw_work,
    estimate_pairwise_ot_work,
)
from .expression_embedding import (
    ExpressionEmbedding,
    ExpressionEmbeddingState,
    fit_expression_embedding,
    load_expression_embedding_state,
    save_expression_embedding_state,
)
from .fgw import (
    batched_fused_gromov_wasserstein_cost,
    fused_gromov_wasserstein_block,
)
from .io import run_pairwise_niche_on_h5ad
from .local_measure import (
    LocalMeasureSet,
    build_instance_neighbor_indices,
    build_local_measures,
)
from .sinkhorn import (
    batched_sinkhorn_cost,
    pairwise_sqdist_block,
    pairwise_sqdist_self,
    sinkhorn_ot_block,
    sinkhorn_self_cost_batch,
)

__all__ = [
    "ClusterResult",
    "ExpressionEmbedding",
    "ExpressionEmbeddingState",
    "HIGH_CONTRAST_PALETTE",
    "LocalMeasureSet",
    "PairwiseNicheConfig",
    "assign_by_reference_medoids",
    "assign_high_contrast_colors",
    "batched_fused_gromov_wasserstein_cost",
    "batched_sinkhorn_cost",
    "build_instance_neighbor_indices",
    "build_local_measures",
    "choose_pairwise_block_size",
    "cluster_from_distance",
    "compute_cross_ot_distance_matrix",
    "compute_pairwise_ot_distance_matrix",
    "estimate_cross_fgw_work",
    "estimate_cross_ot_work",
    "estimate_pairwise_fgw_work",
    "estimate_pairwise_ot_work",
    "fit_expression_embedding",
    "fused_gromov_wasserstein_block",
    "load_expression_embedding_state",
    "ot_knn_distance_graph",
    "pairwise_sqdist_block",
    "pairwise_sqdist_self",
    "run_pairwise_niche_on_h5ad",
    "save_expression_embedding_state",
    "sinkhorn_ot_block",
    "sinkhorn_self_cost_batch",
]
