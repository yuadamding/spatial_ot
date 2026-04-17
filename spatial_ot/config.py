from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import tomllib


@dataclass
class PathConfig:
    cells_h5ad: str
    bins8_h5: str
    bins8_positions: str
    bins2_h5: str | None = None
    bins2_positions: str | None = None
    bins16_h5: str | None = None
    bins16_positions: str | None = None
    barcode_mappings: str | None = None
    programs_json: str | None = None
    output_dir: str = "./runs/default"


@dataclass
class DataConfig:
    seed: int = 1337
    cell_subset: int = 0
    bin_subset: int = 0
    subset_strategy: str = "spatial_knn"
    cell_type_key: str = "coarse_cell_type"
    hvg_n: int = 2000
    encoder_gene_cap: int = 4000
    teacher_knn: int = 8
    contact_knn: int = 8
    context_knn: int = 24
    ot_knn: int = 32
    shell_bounds_um: tuple[float, ...] = (12.0, 25.0, 50.0)
    top_program_features: int = 8
    negative_edge_ratio: float = 1.0


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
    ot_hist: float = 1.0
    ot_cov: float = 0.25
    ot_shell: float = 0.5
    ot_temperature: float = 0.35
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


def _make_dataclass(cls, payload: dict | None):
    payload = payload or {}
    return cls(**payload)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text())
    return ExperimentConfig(
        paths=_make_dataclass(PathConfig, payload.get("paths")),
        data=_make_dataclass(DataConfig, payload.get("data")),
        model=_make_dataclass(ModelConfig, payload.get("model")),
        training=_make_dataclass(TrainingConfig, payload.get("training")),
        loss=_make_dataclass(LossConfig, payload.get("loss")),
    )
