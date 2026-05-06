from __future__ import annotations

from .cluster import (
    ClusterResult,
    cluster_embeddings,
    connected_components_by_label,
    neighbor_label_agreement,
)
from .codebook import (
    FeatureSpace,
    StateCodebook,
    fit_state_codebook,
    prepare_feature_space,
    standardize_features,
)
from .config import CellNicheConfig
from .dataset import CellNicheBatchSpec, CellNicheDataset
from .descriptors import (
    DEFAULT_BLOCK_WEIGHTS,
    DescriptorResult,
    EmbeddingResult,
    compute_cell_heterogeneity_descriptors,
    reduce_descriptor_embedding,
)
from .graph import NeighborhoodGraph, build_knn_graphs, build_radius_graphs
from .io import run_cell_niche_on_h5ad
from .model import MultiScaleDeepSetEncoder, OTDeepSHEModel, OTPrototypeHead
from .ot import sinkhorn_balanced_distance, sinkhorn_divergence
from .train import DeepSHEResult, fit_deepshe_embedding

__all__ = [
    "CellNicheBatchSpec",
    "CellNicheConfig",
    "CellNicheDataset",
    "ClusterResult",
    "DEFAULT_BLOCK_WEIGHTS",
    "DeepSHEResult",
    "DescriptorResult",
    "EmbeddingResult",
    "FeatureSpace",
    "MultiScaleDeepSetEncoder",
    "NeighborhoodGraph",
    "OTDeepSHEModel",
    "OTPrototypeHead",
    "StateCodebook",
    "build_knn_graphs",
    "build_radius_graphs",
    "cluster_embeddings",
    "compute_cell_heterogeneity_descriptors",
    "connected_components_by_label",
    "fit_state_codebook",
    "fit_deepshe_embedding",
    "neighbor_label_agreement",
    "prepare_feature_space",
    "reduce_descriptor_embedding",
    "run_cell_niche_on_h5ad",
    "sinkhorn_balanced_distance",
    "sinkhorn_divergence",
    "standardize_features",
]
