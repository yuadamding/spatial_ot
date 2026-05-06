from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CellNicheConfig
from .dataset import CellNicheDataset
from .graph import NeighborhoodGraph
from .losses import deepshe_loss
from .model import OTDeepSHEModel


@dataclass
class DeepSHEResult:
    embedding: np.ndarray
    prototype_distances: np.ndarray | None
    prototype_posterior: np.ndarray | None
    embedding_raw: np.ndarray | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    model: OTDeepSHEModel | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _resolve_device(requested: str) -> torch.device:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def fit_deepshe_embedding(
    *,
    features: np.ndarray,
    posteriors: np.ndarray,
    coords_um: np.ndarray,
    graphs: dict[str, NeighborhoodGraph],
    descriptor_targets: np.ndarray,
    config: CellNicheConfig,
) -> DeepSHEResult:
    """Fit the mini-batch DeepSHE encoder and return cell embeddings."""

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    dataset = CellNicheDataset(
        features=features,
        posteriors=posteriors,
        coords_um=coords_um,
        graphs=graphs,
        descriptor_targets=descriptor_targets,
        radial_shells=int(config.radial_shells),
        max_neighbors_per_graph=int(config.max_neighbors_per_radius),
    )
    use_ot = bool(config.use_ot_prototypes or config.encoder == "ot_deepshe")
    model = OTDeepSHEModel(
        z_dim=int(np.asarray(features).shape[1]),
        token_input_dim=dataset.token_input_dim,
        n_radii=len(dataset.graph_keys),
        descriptor_dim=int(np.asarray(descriptor_targets).shape[1]),
        token_dim=int(config.token_dim),
        hidden_dim=int(config.hidden_dim),
        embedding_dim=int(config.embedding_dim),
        use_attention=str(config.encoder) != "deepsets",
        use_ot_prototypes=use_ot,
        n_ot_prototypes=int(config.n_ot_prototypes),
        prototype_support_size=int(config.prototype_support_size),
        ot_epsilon=float(config.ot_epsilon),
        ot_temperature=float(config.ot_temperature),
    )
    device = _resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    batch_size = min(max(int(config.batch_size), 1), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(max(int(config.epochs), 1)):
        epoch_parts: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, metrics = deepshe_loss(
                outputs,
                batch,
                context_reconstruction_weight=float(config.context_reconstruction_weight),
                ot_prototype_weight=float(config.ot_prototype_weight),
                prototype_balance_weight=float(config.prototype_balance_weight),
                variance_weight=float(config.variance_weight),
                decorrelation_weight=float(config.decorrelation_weight),
                ot_temperature=float(config.ot_temperature),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for key, value in metrics.items():
                epoch_parts[key] = float(epoch_parts.get(key, 0.0) + float(value))
            n_batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                **{key: float(value / max(n_batches, 1)) for key, value in epoch_parts.items()},
            }
        )

    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    embeddings = np.zeros((len(dataset), int(config.embedding_dim)), dtype=np.float32)
    raw_embeddings = np.zeros((len(dataset), int(config.embedding_dim)), dtype=np.float32)
    proto_dist: np.ndarray | None = None
    proto_post: np.ndarray | None = None
    if use_ot:
        proto_dist = np.zeros((len(dataset), int(config.n_ot_prototypes)), dtype=np.float32)
        proto_post = np.zeros((len(dataset), int(config.n_ot_prototypes)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            ids = batch["anchor_ids"].numpy()
            device_batch = _to_device(batch, device)
            outputs = model(device_batch)
            embeddings[ids] = outputs["embedding"].detach().cpu().numpy().astype(np.float32)
            if isinstance(outputs["embedding_raw"], torch.Tensor):
                raw_embeddings[ids] = (
                    outputs["embedding_raw"].detach().cpu().numpy().astype(np.float32)
                )
            if proto_dist is not None and isinstance(outputs["prototype_distances"], torch.Tensor):
                proto_dist[ids] = (
                    outputs["prototype_distances"].detach().cpu().numpy().astype(np.float32)
                )
            if proto_post is not None and isinstance(outputs["prototype_posterior"], torch.Tensor):
                proto_post[ids] = (
                    outputs["prototype_posterior"].detach().cpu().numpy().astype(np.float32)
                )

    return DeepSHEResult(
        embedding=embeddings,
        embedding_raw=raw_embeddings,
        prototype_distances=proto_dist,
        prototype_posterior=proto_post,
        history=history,
        model=model,
        metadata={
            "encoder": str(config.encoder),
            "n_radii": int(len(dataset.graph_keys)),
            "graph_keys": list(dataset.graph_keys),
            "max_neighbors_per_radius": int(config.max_neighbors_per_radius),
            "token_input_dim": int(dataset.token_input_dim),
            "token_dim": int(config.token_dim),
            "embedding_dim": int(config.embedding_dim),
            "use_ot_prototypes": bool(use_ot),
            "n_ot_prototypes": int(config.n_ot_prototypes) if use_ot else 0,
            "prototype_support_size": int(config.prototype_support_size) if use_ot else 0,
            "device": str(device),
            "epochs": int(config.epochs),
            "batch_size": int(batch_size),
            "training_objective": (
                "context_reconstruction"
                + ("+balanced_sinkhorn_ot_prototypes" if use_ot else "")
            ),
        },
    )


__all__ = ["DeepSHEResult", "fit_deepshe_embedding"]
