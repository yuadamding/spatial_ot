from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class CellNicheConfig:
    """Configuration for cell-centered SHE / OT-DeepSHE runs."""

    feature_obsm_key: str = ""
    spatial_x_key: str = ""
    spatial_y_key: str = ""
    sample_obs_key: str = "sample_id"

    radii_um: tuple[float, ...] = (20.0, 50.0, 100.0)
    spatial_scale: float = 1.0
    max_neighbors_per_radius: int = 64
    kernel: Literal["gaussian", "triangular", "uniform", "binary", "inverse"] = "gaussian"

    molecular_dim: int = 128
    state_codebook_size: int = 64
    codebook_temperature: float | None = None

    radial_shells: int = 3
    descriptor_blocks: tuple[str, ...] = (
        "composition",
        "diversity",
        "moments",
        "radial",
        "pair",
        "covariance",
        "gradient",
    )
    covariance_dims: int = 8

    encoder: Literal["descriptor", "deepsets", "attention_deepsets", "ot_deepshe"] = (
        "descriptor"
    )
    embedding_dim: int = 64
    token_dim: int = 128
    hidden_dim: int = 256

    use_ot_prototypes: bool = False
    n_ot_prototypes: int = 20
    prototype_support_size: int = 32
    ot_epsilon: float = 0.05
    ot_temperature: float = 0.25
    ot_distance_feature_weight: float = 1.0

    batch_size: int = 1024
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"

    context_reconstruction_weight: float = 1.0
    ot_prototype_weight: float = 0.5
    prototype_balance_weight: float = 0.05
    variance_weight: float = 0.02
    decorrelation_weight: float = 0.005

    cluster_method: Literal["leiden", "kmeans"] = "kmeans"
    n_clusters: int | None = None
    leiden_resolution: float = 1.0

    seed: int = 1337

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["CellNicheConfig"]
