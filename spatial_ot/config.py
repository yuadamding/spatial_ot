from __future__ import annotations

from dataclasses import asdict, dataclass, fields
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


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text())
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
