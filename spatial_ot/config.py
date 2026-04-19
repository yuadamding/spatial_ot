from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
import json
import tomllib


@dataclass
class PathConfig:
    cells_h5ad: str
    bins8_h5: str
    bins8_positions: str
    barcode_mappings: str | None = None
    programs_json: str | None = None
    output_dir: str = "./runs/default"


@dataclass
class DataConfig:
    seed: int = 1337
    cell_subset: int = 0
    bin_subset: int = 0
    subset_strategy: str = "spatial_grid"
    cell_type_key: str = "coarse_cell_type"
    hvg_n: int = 2000
    encoder_gene_cap: int = 4000
    teacher_knn: int = 8
    context_knn: int = 24
    ot_knn: int = 32
    shell_bounds_um: tuple[float, ...] = (12.0, 25.0, 50.0)
    top_program_features: int = 8
    negative_edge_ratio: float = 1.0
    allow_nearest_overlap_fallback: bool = True


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    z_dim: int = 32
    teacher_dim: int = 24
    de_novo_programs: int = 16
    state_atoms: int = 24
    niche_prototypes: int = 12
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    device: str = "auto"
    teacher_epochs: int = 8
    intrinsic_epochs: int = 8
    context_epochs: int = 12
    lr: float = 1e-3
    context_lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    permutation_fraction: float = 0.1


@dataclass
class LossConfig:
    teacher_self: float = 1.0
    teacher_nb: float = 0.7
    teacher_edge: float = 0.1
    intrinsic_rec: float = 1.0
    kl_z: float = 0.03
    context_self: float = 1.0
    context_nb: float = 0.8
    edge: float = 0.1
    kl_s: float = 0.03
    teacher_distill: float = 0.5
    teacher_logits: float = 0.2
    independence: float = 0.25
    marker: float = 0.05
    sparsity: float = 1e-4
    ot_hist: float = 1.0
    ot_cov: float = 0.25
    ot_shell: float = 0.5
    ot_temperature: float = 0.35
    state_temperature: float = 0.5
    niche_temperature: float = 0.35
    teacher_temperature: float = 0.5
    comm_epsilon: float = 0.8
    residual_ridge: float = 1.0


@dataclass
class ExperimentConfig:
    paths: PathConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig

    def as_dict(self) -> dict:
        return asdict(self)

    def write_resolved(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))


@dataclass
class MultilevelPathConfig:
    input_h5ad: str = ""
    output_dir: str = ""
    feature_obsm_key: str = ""
    spatial_x_key: str = "cell_x"
    spatial_y_key: str = "cell_y"
    spatial_scale: float = 1.0
    region_obs_key: str | None = None


@dataclass
class MultilevelOTConfig:
    n_clusters: int = 8
    atoms_per_cluster: int = 8
    radius_um: float = 100.0
    stride_um: float = 100.0
    min_cells: int = 25
    max_subregions: int = 1500
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
    allow_convex_hull_fallback: bool = False
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
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation: str = "none"
    validation_context_mode: str = "inductive"
    batch_key: str | None = None
    count_layer: str | None = None
    device: str = "auto"
    reconstruction_weight: float = 1.0
    context_weight: float = 0.5
    contrastive_weight: float = 0.1
    variance_weight: float = 0.1
    decorrelation_weight: float = 0.01
    output_embedding: str = "joint"
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


def _validate_experiment(config: ExperimentConfig) -> ExperimentConfig:
    valid_subset = {"spatial_grid", "stratified"}
    if config.data.subset_strategy not in valid_subset:
        raise ValueError(f"subset_strategy must be one of {sorted(valid_subset)}, got '{config.data.subset_strategy}'")
    if config.data.cell_subset < 0 or config.data.bin_subset < 0:
        raise ValueError("cell_subset and bin_subset must be >= 0")
    if config.data.hvg_n <= 0:
        raise ValueError("hvg_n must be > 0")
    if config.data.encoder_gene_cap < 0:
        raise ValueError("encoder_gene_cap must be >= 0")
    for name in ["teacher_knn", "context_knn", "ot_knn"]:
        if getattr(config.data, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if config.data.negative_edge_ratio <= 0:
        raise ValueError("negative_edge_ratio must be > 0")
    if len(config.data.shell_bounds_um) == 0:
        raise ValueError("shell_bounds_um must contain at least one radius")
    shell_bounds = tuple(float(x) for x in config.data.shell_bounds_um)
    if any(x <= 0 for x in shell_bounds):
        raise ValueError("shell_bounds_um values must be > 0")
    if any(b <= a for a, b in zip(shell_bounds, shell_bounds[1:])):
        raise ValueError("shell_bounds_um must be strictly increasing")
    for required in ["cells_h5ad", "bins8_h5", "bins8_positions", "output_dir"]:
        if not getattr(config.paths, required):
            raise ValueError(f"paths.{required} must be set")
    return config


def _validate_multilevel_experiment(config: MultilevelExperimentConfig) -> MultilevelExperimentConfig:
    for required in ["input_h5ad", "output_dir", "feature_obsm_key"]:
        if not getattr(config.paths, required):
            raise ValueError(f"paths.{required} must be set")
    if config.paths.spatial_scale <= 0:
        raise ValueError("paths.spatial_scale must be > 0")
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
    if config.ot.lambda_x < 0 or config.ot.lambda_y < 0:
        raise ValueError("ot.lambda_x and ot.lambda_y must be non-negative")
    if config.ot.geometry_eps <= 0 or config.ot.ot_eps <= 0:
        raise ValueError("ot.geometry_eps and ot.ot_eps must be > 0")
    if config.ot.rho <= 0:
        raise ValueError("ot.rho must be > 0")
    if config.ot.geometry_samples < 32:
        raise ValueError("ot.geometry_samples must be at least 32")
    if config.ot.compressed_support_size < 2:
        raise ValueError("ot.compressed_support_size must be at least 2")
    if config.ot.align_iters < 1 or config.ot.max_iter < 1 or config.ot.n_init < 1:
        raise ValueError("ot.align_iters, ot.max_iter, and ot.n_init must be at least 1")
    if config.ot.tol <= 0:
        raise ValueError("ot.tol must be > 0")
    if config.ot.min_scale <= 0 or config.ot.max_scale <= 0 or config.ot.min_scale > config.ot.max_scale:
        raise ValueError("ot.min_scale and ot.max_scale must be positive and min_scale <= max_scale")
    if not str(config.ot.compute_device).strip():
        raise ValueError("ot.compute_device must be a non-empty string")

    valid_methods = {"none", "autoencoder", "graph_autoencoder"}
    if config.deep.method not in valid_methods:
        raise ValueError(f"deep.method must be one of {sorted(valid_methods)}, got '{config.deep.method}'")
    valid_validation = {"none", "spatial_block", "sample_holdout"}
    if config.deep.validation not in valid_validation:
        raise ValueError(f"deep.validation must be one of {sorted(valid_validation)}, got '{config.deep.validation}'")
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
        raise ValueError("deep.mid_radius_um must be >= deep.short_radius_um when both are set")
    valid_graph_aggr = {"mean"}
    if config.deep.graph_aggr not in valid_graph_aggr:
        raise ValueError(f"deep.graph_aggr must be one of {sorted(valid_graph_aggr)}, got '{config.deep.graph_aggr}'")
    if config.deep.epochs < 1:
        raise ValueError("deep.epochs must be at least 1")
    if config.deep.batch_size < 1:
        raise ValueError("deep.batch_size must be at least 1")
    if config.deep.learning_rate <= 0 or config.deep.weight_decay < 0:
        raise ValueError("deep.learning_rate must be > 0 and deep.weight_decay must be >= 0")
    if config.deep.count_layer is not None:
        raise NotImplementedError("deep.count_layer is configured but count reconstruction is not implemented yet.")
    for name in ["reconstruction_weight", "context_weight", "contrastive_weight", "variance_weight", "decorrelation_weight"]:
        if getattr(config.deep, name) < 0:
            raise ValueError(f"deep.{name} must be >= 0")
    valid_outputs = {"intrinsic", "context", "joint"}
    if config.deep.output_embedding not in valid_outputs:
        raise ValueError(f"deep.output_embedding must be one of {sorted(valid_outputs)}, got '{config.deep.output_embedding}'")
    if config.deep.early_stopping_patience < 1:
        raise ValueError("deep.early_stopping_patience must be at least 1")
    if config.deep.min_delta < 0:
        raise ValueError("deep.min_delta must be >= 0")
    if config.deep.pretrained_model and config.deep.method == "none":
        raise ValueError("deep.pretrained_model requires deep.method to be an active encoder method")
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = _expand_env_value(tomllib.loads(config_path.read_text()))
    top_level = {"paths", "data", "model", "training", "loss"}
    unknown_top = sorted(set(payload) - top_level)
    if unknown_top:
        raise KeyError(f"Unknown top-level config sections: {', '.join(unknown_top)}")
    config = ExperimentConfig(
        paths=_make_dataclass(PathConfig, _validate_payload_keys("paths", PathConfig, payload.get("paths"))),
        data=_make_dataclass(DataConfig, _validate_payload_keys("data", DataConfig, payload.get("data"))),
        model=_make_dataclass(ModelConfig, _validate_payload_keys("model", ModelConfig, payload.get("model"))),
        training=_make_dataclass(TrainingConfig, _validate_payload_keys("training", TrainingConfig, payload.get("training"))),
        loss=_make_dataclass(LossConfig, _validate_payload_keys("loss", LossConfig, payload.get("loss"))),
    )
    return _validate_experiment(config)


def load_multilevel_config(path: str | Path) -> MultilevelExperimentConfig:
    config_path = Path(path)
    payload = _expand_env_value(tomllib.loads(config_path.read_text()))
    top_level = {"paths", "ot", "deep"}
    unknown_top = sorted(set(payload) - top_level)
    if unknown_top:
        raise KeyError(f"Unknown top-level config sections: {', '.join(unknown_top)}")
    config = MultilevelExperimentConfig(
        paths=_make_dataclass(MultilevelPathConfig, _validate_payload_keys("paths", MultilevelPathConfig, payload.get("paths"))),
        ot=_make_dataclass(MultilevelOTConfig, _validate_payload_keys("ot", MultilevelOTConfig, payload.get("ot"))),
        deep=_make_dataclass(DeepFeatureConfig, _validate_payload_keys("deep", DeepFeatureConfig, payload.get("deep"))),
    )
    return _validate_multilevel_experiment(config)


def validate_multilevel_config(config: MultilevelExperimentConfig) -> MultilevelExperimentConfig:
    return _validate_multilevel_experiment(config)
