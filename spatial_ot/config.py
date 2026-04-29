from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
import json

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class MultilevelPathConfig:
    input_h5ad: str = ""
    output_dir: str = ""
    feature_obsm_key: str = ""
    spatial_x_key: str = "cell_x"
    spatial_y_key: str = "cell_y"
    sample_obs_key: str = "sample_id"
    spatial_scale: float = 1.0
    region_obs_key: str | None = None
    region_geometry_json: str | None = None
    allow_umap_as_feature: bool = False


@dataclass
class MultilevelOTConfig:
    n_clusters: int = 8
    atoms_per_cluster: int = 8
    radius_um: float = 100.0
    stride_um: float = 100.0
    basic_niche_size_um: float | None = 50.0
    min_cells: int = 25
    max_subregions: int = 5000
    max_subregion_area_um2: float | None = None
    lambda_x: float = 0.5
    lambda_y: float = 1.0
    geometry_eps: float = 0.03
    ot_eps: float = 0.03
    rho: float = 0.5
    geometry_samples: int = 192
    compressed_support_size: int = 96
    align_iters: int = 4
    allow_reflection: bool = False
    allow_scale: bool = False
    min_scale: float = 0.75
    max_scale: float = 1.33
    scale_penalty: float = 0.05
    shift_penalty: float = 0.05
    n_init: int = 5
    overlap_consistency_weight: float = 0.05
    overlap_jaccard_min: float = 0.15
    overlap_contrast_scale: float = 1.0
    allow_convex_hull_fallback: bool = False
    subregion_construction_method: str = "data_driven"
    subregion_feature_weight: float = 0.0
    subregion_feature_dims: int = 16
    deep_segmentation_knn: int = 12
    deep_segmentation_feature_dims: int = 32
    deep_segmentation_feature_weight: float = 1.0
    deep_segmentation_spatial_weight: float = 0.05
    joint_refinement_iters: int = 2
    joint_refinement_knn: int = 12
    joint_refinement_feature_dims: int = 32
    joint_refinement_cluster_weight: float = 1.0
    joint_refinement_spatial_weight: float = 0.25
    joint_refinement_cut_weight: float = 0.5
    joint_refinement_max_move_fraction: float = 0.05
    subregion_clustering_method: str = "heterogeneity_descriptor_niche"
    subregion_latent_embedding_mode: str = "mean_std_shrunk"
    subregion_latent_shrinkage_tau: float = 25.0
    subregion_latent_heterogeneity_weight: float = 0.5
    subregion_latent_sample_prior_weight: float = 0.5
    subregion_latent_codebook_size: int = 32
    subregion_latent_codebook_sample_size: int = 50000
    heterogeneity_composition_weight: float = 0.20
    heterogeneity_diversity_weight: float = 0.15
    heterogeneity_spatial_field_weight: float = 0.35
    heterogeneity_pair_cooccurrence_weight: float = 0.30
    heterogeneity_pair_distance_bins: str = "0.25,0.5,1.0,2.0"
    heterogeneity_pair_graph_mode: str = "all_pairs"
    heterogeneity_pair_graph_k: int = 8
    heterogeneity_pair_graph_radius: float | None = None
    heterogeneity_pair_bin_normalization: str = "per_bin"
    heterogeneity_transport_max_subregions: int = 800
    heterogeneity_transport_feature_mode: str = "soft_codebook"
    heterogeneity_transport_feature_cost: str = "hellinger_codebook"
    heterogeneity_fused_ot_feature_weight: float = 0.5
    heterogeneity_fused_ot_coordinate_weight: float = 0.5
    heterogeneity_fgw_alpha: float = 0.5
    heterogeneity_fgw_solver: str = "conditional_gradient"
    heterogeneity_fgw_epsilon: float = 0.05
    heterogeneity_fgw_loss_fun: str = "square_loss"
    heterogeneity_fgw_max_iter: int = 500
    heterogeneity_fgw_tol: float = 1e-7
    heterogeneity_fgw_structure_scale: str = "global_median"
    heterogeneity_fgw_structure_clip: float | None = 3.0
    heterogeneity_fgw_partial: bool = False
    heterogeneity_fgw_partial_mass: float = 0.85
    heterogeneity_fgw_partial_reg: float = 0.05
    shape_diagnostics: bool = True
    shape_leakage_permutations: int = 64
    compute_spot_latent: bool = True
    auto_n_clusters: bool = False
    candidate_n_clusters: tuple[int, ...] = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
    auto_k_max_score_subregions: int = 2500
    auto_k_gap_references: int = 8
    auto_k_mds_components: int = 8
    auto_k_pilot_n_init: int = 1
    auto_k_pilot_max_iter: int = 3
    min_subregions_per_cluster: int = 50
    max_iter: int = 10
    tol: float = 1e-4
    seed: int = 1337
    compute_device: str = "auto"


@dataclass
class DeepFeatureConfig:
    method: str = "none"
    latent_dim: int = 16
    hidden_dim: int = 128
    layers: int = 2
    neighbor_k: int = 8
    radius_um: float | None = None
    short_radius_um: float | None = None
    mid_radius_um: float | None = None
    graph_layers: int = 2
    graph_aggr: str = "mean"
    graph_max_neighbors: int = 64
    full_batch_max_cells: int = 50000
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation: str = "none"
    validation_context_mode: str = "inductive"
    batch_key: str | None = None
    count_layer: str | None = None
    count_decoder_rank: int = 32
    count_chunk_size: int = 1024
    count_loss_weight: float = 0.5
    device: str = "auto"
    reconstruction_weight: float = 1.0
    context_weight: float = 0.5
    contrastive_weight: float = 0.1
    variance_weight: float = 0.1
    decorrelation_weight: float = 0.01
    independence_weight: float = 0.1
    output_embedding: str | None = None
    allow_joint_ot_embedding: bool = False
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    restore_best: bool = True
    save_model: bool = True
    pretrained_model: str | None = None
    output_obsm_key: str = "X_spatial_ot_deep"


@dataclass
class MultilevelExperimentConfig:
    paths: MultilevelPathConfig = field(default_factory=MultilevelPathConfig)
    ot: MultilevelOTConfig = field(default_factory=MultilevelOTConfig)
    deep: DeepFeatureConfig = field(default_factory=DeepFeatureConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    def write_resolved(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))


def _validate_payload_keys(section: str, cls, payload: dict | None) -> dict:
    payload = payload or {}
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise KeyError(f"Unknown keys in [{section}]: {', '.join(unknown)}")
    return payload


def _make_dataclass(cls, payload: dict | None):
    payload = payload or {}
    return cls(**payload)


def _expand_env_value(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _expand_env_value(v) for k, v in value.items()}
    return value


def _parse_candidate_n_clusters(value) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        if "-" in raw and "," not in raw:
            left, right = raw.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())
            if stop < start:
                raise ValueError("ot.candidate_n_clusters range must be increasing")
            return tuple(range(start, stop + 1))
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(k) for k in value)


def _validate_multilevel_experiment(
    config: MultilevelExperimentConfig,
) -> MultilevelExperimentConfig:
    for required in ["input_h5ad", "output_dir", "feature_obsm_key"]:
        if not getattr(config.paths, required):
            raise ValueError(f"paths.{required} must be set")
    if config.paths.region_geometry_json and not config.paths.region_obs_key:
        raise ValueError("paths.region_geometry_json requires paths.region_obs_key")
    if config.paths.spatial_scale <= 0:
        raise ValueError("paths.spatial_scale must be > 0")
    if config.ot.basic_niche_size_um is not None and config.ot.basic_niche_size_um <= 0:
        config.ot.basic_niche_size_um = None
    if config.ot.n_clusters < 2:
        raise ValueError("ot.n_clusters must be at least 2")
    if config.ot.atoms_per_cluster < 1:
        raise ValueError("ot.atoms_per_cluster must be at least 1")
    if config.ot.radius_um <= 0 or config.ot.stride_um <= 0:
        raise ValueError("ot.radius_um and ot.stride_um must be > 0")
    if config.ot.min_cells < 1:
        raise ValueError("ot.min_cells must be >= 1")
    if config.ot.max_subregions != 0 and config.ot.max_subregions < 1:
        raise ValueError("ot.max_subregions must be positive or 0")
    if (
        config.ot.max_subregion_area_um2 is not None
        and config.ot.max_subregion_area_um2 <= 0
    ):
        config.ot.max_subregion_area_um2 = None
    if config.ot.lambda_x < 0 or config.ot.lambda_y < 0:
        raise ValueError("ot.lambda_x and ot.lambda_y must be non-negative")
    if config.ot.lambda_x == 0 and config.ot.lambda_y == 0:
        raise ValueError("at least one of lambda_x or lambda_y must be positive")
    if config.ot.overlap_consistency_weight < 0:
        raise ValueError("ot.overlap_consistency_weight must be >= 0")
    if config.ot.overlap_jaccard_min < 0 or config.ot.overlap_jaccard_min > 1:
        raise ValueError("ot.overlap_jaccard_min must be between 0 and 1")
    if config.ot.overlap_contrast_scale <= 0:
        raise ValueError("ot.overlap_contrast_scale must be > 0")
    valid_subregion_construction = {
        "data_driven",
        "deep_segmentation",
        "joint_refinement",
    }
    config.ot.subregion_construction_method = (
        str(config.ot.subregion_construction_method).strip().lower()
    )
    if config.ot.subregion_construction_method not in valid_subregion_construction:
        raise ValueError(
            "ot.subregion_construction_method must be one of "
            f"{sorted(valid_subregion_construction)}, got '{config.ot.subregion_construction_method}'"
        )
    if config.ot.subregion_feature_weight < 0:
        raise ValueError("ot.subregion_feature_weight must be >= 0")
    if config.ot.subregion_feature_dims < 0:
        raise ValueError("ot.subregion_feature_dims must be >= 0")
    if config.ot.deep_segmentation_knn < 2:
        raise ValueError("ot.deep_segmentation_knn must be at least 2")
    if config.ot.deep_segmentation_feature_dims < 1:
        raise ValueError("ot.deep_segmentation_feature_dims must be at least 1")
    if (
        config.ot.deep_segmentation_feature_weight < 0
        or config.ot.deep_segmentation_spatial_weight < 0
    ):
        raise ValueError(
            "ot.deep_segmentation_feature_weight and ot.deep_segmentation_spatial_weight must be >= 0"
        )
    if (
        config.ot.deep_segmentation_feature_weight == 0
        and config.ot.deep_segmentation_spatial_weight == 0
    ):
        raise ValueError("at least one deep segmentation edge weight must be positive")
    if config.ot.joint_refinement_iters < 0:
        raise ValueError("ot.joint_refinement_iters must be >= 0")
    if config.ot.joint_refinement_knn < 2:
        raise ValueError("ot.joint_refinement_knn must be at least 2")
    if config.ot.joint_refinement_feature_dims < 1:
        raise ValueError("ot.joint_refinement_feature_dims must be at least 1")
    if config.ot.joint_refinement_cluster_weight < 0:
        raise ValueError("ot.joint_refinement_cluster_weight must be >= 0")
    if config.ot.joint_refinement_spatial_weight < 0:
        raise ValueError("ot.joint_refinement_spatial_weight must be >= 0")
    if config.ot.joint_refinement_cut_weight < 0:
        raise ValueError("ot.joint_refinement_cut_weight must be >= 0")
    if (
        config.ot.joint_refinement_max_move_fraction < 0
        or config.ot.joint_refinement_max_move_fraction > 1
    ):
        raise ValueError("ot.joint_refinement_max_move_fraction must be in [0, 1]")
    valid_clustering_methods = {
        "pooled_subregion_latent",
        "heterogeneity_descriptor_niche",
        "heterogeneity_ot_niche",
        "heterogeneity_fused_ot_niche",
        "heterogeneity_fgw_niche",
        "ot_dictionary",
    }
    config.ot.subregion_clustering_method = (
        str(config.ot.subregion_clustering_method).strip().lower()
    )
    if config.ot.subregion_clustering_method not in valid_clustering_methods:
        raise ValueError(
            "ot.subregion_clustering_method must be one of "
            f"{sorted(valid_clustering_methods)}, got '{config.ot.subregion_clustering_method}'"
        )
    valid_latent_modes = {
        "mean_std",
        "mean_std_shrunk",
        "mean_std_skew_count",
        "mean_std_quantile",
        "codebook_histogram",
        "mean_std_codebook",
    }
    config.ot.subregion_latent_embedding_mode = (
        str(config.ot.subregion_latent_embedding_mode).strip().lower().replace("-", "_")
    )
    if config.ot.subregion_latent_embedding_mode not in valid_latent_modes:
        raise ValueError(
            "ot.subregion_latent_embedding_mode must be one of "
            f"{sorted(valid_latent_modes)}, got '{config.ot.subregion_latent_embedding_mode}'"
        )
    if config.ot.subregion_latent_shrinkage_tau < 0:
        raise ValueError("ot.subregion_latent_shrinkage_tau must be >= 0")
    if config.ot.subregion_latent_heterogeneity_weight < 0:
        raise ValueError("ot.subregion_latent_heterogeneity_weight must be >= 0")
    if not 0 <= config.ot.subregion_latent_sample_prior_weight <= 1:
        raise ValueError("ot.subregion_latent_sample_prior_weight must be in [0, 1]")
    if config.ot.subregion_latent_codebook_size < 2:
        raise ValueError("ot.subregion_latent_codebook_size must be at least 2")
    if (
        config.ot.subregion_latent_codebook_sample_size
        < config.ot.subregion_latent_codebook_size
    ):
        raise ValueError(
            "ot.subregion_latent_codebook_sample_size must be >= ot.subregion_latent_codebook_size"
        )
    heterogeneity_weights = [
        config.ot.heterogeneity_composition_weight,
        config.ot.heterogeneity_diversity_weight,
        config.ot.heterogeneity_spatial_field_weight,
        config.ot.heterogeneity_pair_cooccurrence_weight,
    ]
    if any(float(weight) < 0 for weight in heterogeneity_weights):
        raise ValueError("ot.heterogeneity_*_weight values must be >= 0")
    if sum(float(weight) for weight in heterogeneity_weights) <= 1e-12:
        raise ValueError("at least one ot.heterogeneity_*_weight must be positive")
    config.ot.heterogeneity_pair_graph_mode = (
        str(config.ot.heterogeneity_pair_graph_mode)
        .strip()
        .lower()
        .replace("-", "_")
    )
    if config.ot.heterogeneity_pair_graph_mode not in {"all_pairs", "knn", "radius"}:
        raise ValueError(
            "ot.heterogeneity_pair_graph_mode must be one of all_pairs, knn, radius"
        )
    if config.ot.heterogeneity_pair_graph_k < 1:
        raise ValueError("ot.heterogeneity_pair_graph_k must be >= 1")
    if (
        config.ot.heterogeneity_pair_graph_radius is not None
        and config.ot.heterogeneity_pair_graph_radius <= 0
    ):
        config.ot.heterogeneity_pair_graph_radius = None
    config.ot.heterogeneity_pair_bin_normalization = (
        str(config.ot.heterogeneity_pair_bin_normalization).strip().lower()
    )
    if config.ot.heterogeneity_pair_bin_normalization not in {"per_bin", "global"}:
        raise ValueError(
            "ot.heterogeneity_pair_bin_normalization must be 'per_bin' or 'global'"
        )
    if config.ot.heterogeneity_transport_max_subregions < 0:
        raise ValueError("ot.heterogeneity_transport_max_subregions must be >= 0")
    config.ot.heterogeneity_transport_feature_mode = (
        str(config.ot.heterogeneity_transport_feature_mode).strip().lower()
    )
    if config.ot.heterogeneity_transport_feature_mode not in {
        "soft_codebook",
        "whitened_features",
        "whitened_features_plus_soft_codebook",
    }:
        raise ValueError(
            "ot.heterogeneity_transport_feature_mode must be soft_codebook, "
            "whitened_features, or whitened_features_plus_soft_codebook"
        )
    config.ot.heterogeneity_transport_feature_cost = (
        str(config.ot.heterogeneity_transport_feature_cost).strip().lower()
    )
    if config.ot.heterogeneity_transport_feature_cost not in {
        "hellinger_codebook",
        "hellinger",
        "sqeuclidean",
        "squared_euclidean",
    }:
        raise ValueError(
            "ot.heterogeneity_transport_feature_cost must be hellinger_codebook "
            "or sqeuclidean"
        )
    if (
        config.ot.heterogeneity_fused_ot_feature_weight < 0
        or config.ot.heterogeneity_fused_ot_coordinate_weight < 0
    ):
        raise ValueError("ot.heterogeneity_fused_ot_*_weight values must be >= 0")
    if (
        config.ot.heterogeneity_fused_ot_feature_weight
        + config.ot.heterogeneity_fused_ot_coordinate_weight
        <= 1e-12
    ):
        raise ValueError("at least one fused-OT heterogeneity weight must be positive")
    if not 0 <= config.ot.heterogeneity_fgw_alpha <= 1:
        raise ValueError("ot.heterogeneity_fgw_alpha must be in [0, 1]")
    config.ot.heterogeneity_fgw_solver = (
        str(config.ot.heterogeneity_fgw_solver).strip().lower()
    )
    if config.ot.heterogeneity_fgw_solver not in {
        "conditional_gradient",
        "cg",
        "entropic",
        "pgd",
        "ppa",
    }:
        raise ValueError(
            "ot.heterogeneity_fgw_solver must be conditional_gradient, entropic, pgd, or ppa"
        )
    if config.ot.heterogeneity_fgw_epsilon <= 0:
        raise ValueError("ot.heterogeneity_fgw_epsilon must be > 0")
    if config.ot.heterogeneity_fgw_max_iter < 1:
        raise ValueError("ot.heterogeneity_fgw_max_iter must be >= 1")
    if config.ot.heterogeneity_fgw_tol <= 0:
        raise ValueError("ot.heterogeneity_fgw_tol must be > 0")
    if (
        config.ot.heterogeneity_fgw_structure_clip is not None
        and config.ot.heterogeneity_fgw_structure_clip <= 0
    ):
        config.ot.heterogeneity_fgw_structure_clip = None
    if not 0 < config.ot.heterogeneity_fgw_partial_mass <= 1:
        raise ValueError("ot.heterogeneity_fgw_partial_mass must be in (0, 1]")
    if config.ot.heterogeneity_fgw_partial_reg <= 0:
        raise ValueError("ot.heterogeneity_fgw_partial_reg must be > 0")
    if config.ot.geometry_eps <= 0 or config.ot.ot_eps <= 0:
        raise ValueError("ot.geometry_eps and ot.ot_eps must be > 0")
    if config.ot.rho <= 0:
        raise ValueError("ot.rho must be > 0")
    if config.ot.geometry_samples < 32:
        raise ValueError("ot.geometry_samples must be at least 32")
    if config.ot.compressed_support_size < 2:
        raise ValueError("ot.compressed_support_size must be at least 2")
    if config.ot.align_iters < 1 or config.ot.max_iter < 1 or config.ot.n_init < 1:
        raise ValueError(
            "ot.align_iters, ot.max_iter, and ot.n_init must be at least 1"
        )
    if config.ot.tol <= 0:
        raise ValueError("ot.tol must be > 0")
    if (
        config.ot.min_scale <= 0
        or config.ot.max_scale <= 0
        or config.ot.min_scale > config.ot.max_scale
    ):
        raise ValueError(
            "ot.min_scale and ot.max_scale must be positive and min_scale <= max_scale"
        )
    if not str(config.ot.compute_device).strip():
        raise ValueError("ot.compute_device must be a non-empty string")
    if config.ot.shape_leakage_permutations < 0:
        raise ValueError("ot.shape_leakage_permutations must be >= 0")
    config.ot.candidate_n_clusters = tuple(
        sorted(
            {
                int(k)
                for k in _parse_candidate_n_clusters(config.ot.candidate_n_clusters)
                if int(k) >= 2
            }
        )
    )
    if not config.ot.candidate_n_clusters:
        raise ValueError("ot.candidate_n_clusters must contain at least one K >= 2")
    if config.ot.auto_k_max_score_subregions < 0:
        raise ValueError("ot.auto_k_max_score_subregions must be >= 0")
    if config.ot.auto_k_gap_references < 0:
        raise ValueError("ot.auto_k_gap_references must be >= 0")
    if config.ot.auto_k_mds_components < 1:
        raise ValueError("ot.auto_k_mds_components must be at least 1")
    if config.ot.auto_k_pilot_n_init < 1 or config.ot.auto_k_pilot_max_iter < 1:
        raise ValueError(
            "ot.auto_k_pilot_n_init and ot.auto_k_pilot_max_iter must be at least 1"
        )
    if config.ot.min_subregions_per_cluster < 1:
        raise ValueError("ot.min_subregions_per_cluster must be at least 1")

    valid_methods = {"none", "autoencoder", "graph_autoencoder"}
    if config.deep.method not in valid_methods:
        raise ValueError(
            f"deep.method must be one of {sorted(valid_methods)}, got '{config.deep.method}'"
        )
    valid_validation = {"none", "spatial_block", "sample_holdout"}
    if config.deep.validation not in valid_validation:
        raise ValueError(
            f"deep.validation must be one of {sorted(valid_validation)}, got '{config.deep.validation}'"
        )
    valid_context_modes = {"inductive", "transductive"}
    if config.deep.validation_context_mode not in valid_context_modes:
        raise ValueError(
            f"deep.validation_context_mode must be one of {sorted(valid_context_modes)}, got '{config.deep.validation_context_mode}'"
        )
    if config.deep.latent_dim < 2:
        raise ValueError("deep.latent_dim must be at least 2")
    if config.deep.hidden_dim < 4:
        raise ValueError("deep.hidden_dim must be at least 4")
    if config.deep.layers < 1:
        raise ValueError("deep.layers must be at least 1")
    if config.deep.graph_layers < 1:
        raise ValueError("deep.graph_layers must be at least 1")
    if config.deep.graph_max_neighbors < 1:
        raise ValueError("deep.graph_max_neighbors must be at least 1")
    if config.deep.full_batch_max_cells < 0:
        raise ValueError("deep.full_batch_max_cells must be >= 0")
    if config.deep.neighbor_k < 1:
        raise ValueError("deep.neighbor_k must be at least 1")
    if config.deep.radius_um is not None and config.deep.radius_um <= 0:
        raise ValueError("deep.radius_um must be > 0 when set")
    if config.deep.short_radius_um is not None and config.deep.short_radius_um <= 0:
        raise ValueError("deep.short_radius_um must be > 0 when set")
    if config.deep.mid_radius_um is not None and config.deep.mid_radius_um <= 0:
        raise ValueError("deep.mid_radius_um must be > 0 when set")
    if (
        config.deep.short_radius_um is not None
        and config.deep.mid_radius_um is not None
        and config.deep.mid_radius_um < config.deep.short_radius_um
    ):
        raise ValueError(
            "deep.mid_radius_um must be >= deep.short_radius_um when both are set"
        )
    valid_graph_aggr = {"mean"}
    if config.deep.graph_aggr not in valid_graph_aggr:
        raise ValueError(
            f"deep.graph_aggr must be one of {sorted(valid_graph_aggr)}, got '{config.deep.graph_aggr}'"
        )
    if config.deep.epochs < 1:
        raise ValueError("deep.epochs must be at least 1")
    if config.deep.batch_size < 1:
        raise ValueError("deep.batch_size must be at least 1")
    if config.deep.learning_rate <= 0 or config.deep.weight_decay < 0:
        raise ValueError(
            "deep.learning_rate must be > 0 and deep.weight_decay must be >= 0"
        )
    if config.deep.count_layer is not None and not str(config.deep.count_layer).strip():
        raise ValueError("deep.count_layer must be a non-empty string when set")
    if config.deep.count_decoder_rank < 1:
        raise ValueError("deep.count_decoder_rank must be at least 1")
    if config.deep.count_chunk_size < 1:
        raise ValueError("deep.count_chunk_size must be at least 1")
    if config.deep.count_loss_weight < 0:
        raise ValueError("deep.count_loss_weight must be >= 0")
    for name in [
        "reconstruction_weight",
        "context_weight",
        "contrastive_weight",
        "variance_weight",
        "decorrelation_weight",
        "independence_weight",
    ]:
        if getattr(config.deep, name) < 0:
            raise ValueError(f"deep.{name} must be >= 0")
    valid_outputs = {"intrinsic", "context", "joint"}
    if config.deep.method != "none" and config.deep.output_embedding is None:
        raise ValueError(
            "deep.output_embedding must be set explicitly when deep.method is active"
        )
    if (
        config.deep.output_embedding is not None
        and config.deep.output_embedding not in valid_outputs
    ):
        raise ValueError(
            f"deep.output_embedding must be one of {sorted(valid_outputs)}, got '{config.deep.output_embedding}'"
        )
    if config.deep.early_stopping_patience < 1:
        raise ValueError("deep.early_stopping_patience must be at least 1")
    if config.deep.min_delta < 0:
        raise ValueError("deep.min_delta must be >= 0")
    if config.deep.pretrained_model and config.deep.method == "none":
        raise ValueError(
            "deep.pretrained_model requires deep.method to be an active encoder method"
        )
    return config


def load_multilevel_config(path: str | Path) -> MultilevelExperimentConfig:
    config_path = Path(path)
    payload = _expand_env_value(tomllib.loads(config_path.read_text()))
    top_level = {"paths", "ot", "deep"}
    unknown_top = sorted(set(payload) - top_level)
    if unknown_top:
        raise KeyError(f"Unknown top-level config sections: {', '.join(unknown_top)}")
    config = MultilevelExperimentConfig(
        paths=_make_dataclass(
            MultilevelPathConfig,
            _validate_payload_keys("paths", MultilevelPathConfig, payload.get("paths")),
        ),
        ot=_make_dataclass(
            MultilevelOTConfig,
            _validate_payload_keys("ot", MultilevelOTConfig, payload.get("ot")),
        ),
        deep=_make_dataclass(
            DeepFeatureConfig,
            _validate_payload_keys("deep", DeepFeatureConfig, payload.get("deep")),
        ),
    )
    return _validate_multilevel_experiment(config)


def validate_multilevel_config(
    config: MultilevelExperimentConfig,
) -> MultilevelExperimentConfig:
    return _validate_multilevel_experiment(config)
