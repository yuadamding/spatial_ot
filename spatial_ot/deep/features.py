from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import DeepFeatureConfig
from .._runtime import runtime_memory_snapshot as _runtime_memory_snapshot
from ._utils import (
    apply_standardization as _apply_standardization,
    iter_batches as _iter_batches,
    resolve_device as _resolve_device,
    row_sums as _row_sums,
    sample_gene_chunk as _sample_gene_chunk,
    seed_everything as _seed_everything,
    slice_count_chunk as _slice_count_chunk,
    standardize_features as _standardize_features,
)
from .checkpoint import (
    load_encoder_bundle,
    load_training_checkpoint,
    save_encoder_bundle,
    save_training_checkpoint,
)
from .diagnostics import (
    graph_summary as _graph_summary,
    latent_diagnostics as _latent_diagnostics,
)
from .losses import (
    cross_correlation_loss,
    decorrelation_loss,
    edge_contrastive_loss,
    negative_binomial_loss,
    variance_loss,
)
from .models import (
    make_model as _make_model,
    tensor_graphs as _tensor_graphs,
)
from .validation import (
    build_split_context_targets as _build_split_context_targets,
    split_validation as _split_validation,
)


_START_TIME = time.perf_counter()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _progress_enabled() -> bool:
    raw = os.environ.get("SPATIAL_OT_PROGRESS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _jsonable_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cuda_memory_record(device: torch.device) -> dict[str, float | int | str] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = torch.cuda.current_device() if device.index is None else device.index
    allocated = int(torch.cuda.memory_allocated(device_index))
    reserved = int(torch.cuda.memory_reserved(device_index))
    max_allocated = int(torch.cuda.max_memory_allocated(device_index))
    max_reserved = int(torch.cuda.max_memory_reserved(device_index))
    return {
        "cuda_device_index": int(device_index),
        "cuda_memory_allocated_bytes": allocated,
        "cuda_memory_reserved_bytes": reserved,
        "cuda_max_memory_allocated_bytes": max_allocated,
        "cuda_max_memory_reserved_bytes": max_reserved,
        "cuda_memory_allocated_gb": allocated / (1024.0**3),
        "cuda_memory_reserved_gb": reserved / (1024.0**3),
        "cuda_max_memory_allocated_gb": max_allocated / (1024.0**3),
        "cuda_max_memory_reserved_gb": max_reserved / (1024.0**3),
    }


def _log_epoch_record(
    epoch_row: dict[str, object],
    *,
    total_epochs: int,
    method: str,
    device: torch.device,
    context: dict[str, object] | None = None,
) -> None:
    if not _progress_enabled():
        return
    elapsed = time.perf_counter() - _START_TIME
    record = {
        "method": str(method),
        "device": str(device),
        "epoch": int(epoch_row.get("epoch", 0)),
        "epochs": int(total_epochs),
        "elapsed_sec": float(elapsed),
    }
    if context:
        record.update({key: _jsonable_scalar(value) for key, value in context.items()})
    record.update(
        {
            key: _jsonable_scalar(value)
            for key, value in epoch_row.items()
            if key != "epoch"
        }
    )
    cuda_record = _cuda_memory_record(device)
    if cuda_record is not None:
        record.update(cuda_record)
    print(
        f"[spatial_ot deep {elapsed:8.1f}s] epoch_record "
        f"{json.dumps(record, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


@dataclass
class DeepFeatureResult:
    embedding: np.ndarray
    history: list[dict[str, float]]
    config: dict
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_schema: dict
    validation_report: dict
    latent_diagnostics: dict
    model_path: str | None = None


class SpatialOTFeatureEncoder:
    def __init__(self, config: DeepFeatureConfig) -> None:
        self.config = config
        self.model: nn.Module | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.input_dim: int | None = None
        self.context_dim: int | None = None
        self.count_dim: int | None = None
        self.history: list[dict[str, float]] = []
        self.feature_schema: dict = {}
        self.validation_report: dict = {}
        self.latent_diagnostics: dict = {}
        self.device = _resolve_device(config.device)

    @property
    def uses_coordinate_input(self) -> bool:
        return False

    def _check_fitted(self) -> None:
        if (
            self.model is None
            or self.feature_mean is None
            or self.feature_std is None
            or self.input_dim is None
            or self.context_dim is None
        ):
            raise RuntimeError("SpatialOTFeatureEncoder is not fitted.")

    def _prepare_count_targets(
        self,
        count_matrix,
    ) -> tuple[object | None, np.ndarray | None]:
        self.count_dim = None
        if self.config.count_layer is None:
            return None, None
        if count_matrix is None:
            raise ValueError(
                f"deep.count_layer='{self.config.count_layer}' requires a count matrix target, but none was provided."
            )
        counts = (
            count_matrix.tocsr(copy=True)
            if sparse.issparse(count_matrix)
            else np.asarray(count_matrix)
        )
        library = np.log(np.maximum(_row_sums(counts), 1.0)).astype(
            np.float32, copy=False
        )
        self.count_dim = int(counts.shape[1])
        return counts, library

    def _checkpoint_dir(self) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        raw = str(self.config.checkpoint_dir).strip()
        if not raw:
            return None
        return Path(raw)

    def _checkpoint_path_for_epoch(self, epoch: int) -> Path | None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None:
            return None
        return checkpoint_dir / f"deep_feature_epoch_{int(epoch):04d}.pt"

    def _latest_checkpoint_path(self) -> Path | None:
        checkpoint_dir = self._checkpoint_dir()
        if checkpoint_dir is None:
            return None
        return checkpoint_dir / "latest.pt"

    def _resolve_resume_checkpoint_path(self) -> Path | None:
        raw = self.config.resume_checkpoint
        if raw is None or not str(raw).strip():
            return None
        token = str(raw).strip()
        if token.lower() == "auto":
            latest = self._latest_checkpoint_path()
            if latest is not None and latest.exists():
                return latest
            return None
        return Path(token)

    def _save_training_checkpoint(
        self,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        best_state: dict[str, torch.Tensor] | None,
        best_val: float,
        patience_left: int,
    ) -> None:
        interval = int(self.config.checkpoint_every_epochs)
        if interval <= 0 or int(epoch) % interval != 0:
            return
        checkpoint_path = self._checkpoint_path_for_epoch(epoch)
        latest_path = self._latest_checkpoint_path()
        if checkpoint_path is None or latest_path is None:
            return
        assert self.model is not None
        state = {
            "format": "spatial_ot_deep_training_checkpoint_v1",
            "epoch": int(epoch),
            "config": asdict(self.config),
            "input_dim": int(self.input_dim or 0),
            "context_dim": int(self.context_dim or 0),
            "count_dim": int(self.count_dim) if self.count_dim is not None else None,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": list(self.history),
            "feature_mean": torch.as_tensor(
                np.asarray(self.feature_mean, dtype=np.float32)
            ),
            "feature_std": torch.as_tensor(
                np.asarray(self.feature_std, dtype=np.float32)
            ),
            "feature_schema": dict(self.feature_schema),
            "validation_report": dict(self.validation_report),
            "best_state_dict": best_state,
            "best_val": None if best_val == float("inf") else float(best_val),
            "patience_left": int(patience_left),
            "saved_at_unix": float(time.time()),
        }
        save_training_checkpoint(checkpoint_path, state)
        save_training_checkpoint(latest_path, state)
        if _progress_enabled():
            elapsed = time.perf_counter() - _START_TIME
            print(
                f"[spatial_ot deep {elapsed:8.1f}s] checkpoint_saved "
                f"{json.dumps({'epoch': int(epoch), 'path': str(checkpoint_path), 'latest_path': str(latest_path)}, sort_keys=True)}",
                file=sys.stderr,
                flush=True,
            )

    def _load_training_checkpoint(
        self,
        *,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, dict[str, torch.Tensor] | None, float, int] | None:
        resume_path = self._resolve_resume_checkpoint_path()
        if resume_path is None:
            return None
        if not resume_path.exists():
            raise FileNotFoundError(f"Deep resume checkpoint not found: {resume_path}")
        assert self.model is not None
        checkpoint = load_training_checkpoint(resume_path, map_location=self.device)
        if checkpoint.get("format") != "spatial_ot_deep_training_checkpoint_v1":
            raise ValueError(f"Unrecognized deep training checkpoint: {resume_path}")
        expected = {
            "input_dim": int(self.input_dim or 0),
            "context_dim": int(self.context_dim or 0),
            "count_dim": int(self.count_dim) if self.count_dim is not None else None,
        }
        observed = {
            "input_dim": int(checkpoint.get("input_dim", -1)),
            "context_dim": int(checkpoint.get("context_dim", -1)),
            "count_dim": checkpoint.get("count_dim"),
        }
        if observed != expected:
            raise ValueError(
                "Deep resume checkpoint feature shape does not match this run: "
                f"checkpoint={observed}, current={expected}."
            )
        saved_config = dict(checkpoint.get("config", {}))
        for key in ("method", "latent_dim", "hidden_dim", "layers", "output_embedding"):
            if saved_config.get(key) != getattr(self.config, key):
                raise ValueError(
                    "Deep resume checkpoint config mismatch for "
                    f"{key}: checkpoint={saved_config.get(key)!r}, "
                    f"current={getattr(self.config, key)!r}."
                )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if bool(self.config.resume_reset_optimizer_lr):
            for group in optimizer.param_groups:
                group["lr"] = float(self.config.learning_rate)
        self.history = list(checkpoint.get("history", []))
        self.feature_mean = np.asarray(
            checkpoint["feature_mean"].detach().cpu().numpy(), dtype=np.float32
        )
        self.feature_std = np.asarray(
            checkpoint["feature_std"].detach().cpu().numpy(), dtype=np.float32
        )
        self.feature_schema.update(dict(checkpoint.get("feature_schema", {})))
        self.validation_report.update(dict(checkpoint.get("validation_report", {})))
        best_state = checkpoint.get("best_state_dict")
        best_val_raw = checkpoint.get("best_val")
        best_val = float("inf") if best_val_raw is None else float(best_val_raw)
        patience_left = int(
            checkpoint.get("patience_left", self.config.early_stopping_patience)
        )
        start_epoch = int(checkpoint.get("epoch", 0))
        if _progress_enabled():
            elapsed = time.perf_counter() - _START_TIME
            print(
                f"[spatial_ot deep {elapsed:8.1f}s] checkpoint_resumed "
                f"{json.dumps({'path': str(resume_path), 'resume_after_epoch': start_epoch, 'history_rows': len(self.history)}, sort_keys=True)}",
                file=sys.stderr,
                flush=True,
            )
        return start_epoch, best_state, best_val, patience_left

    def _clip_gradients(self) -> float | None:
        if self.model is None or self.config.gradient_clip_norm is None:
            return None
        max_norm = float(self.config.gradient_clip_norm)
        if max_norm <= 0.0:
            return None
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        return float(norm.detach().cpu())

    def _count_loss_from_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        row_index: np.ndarray,
        gene_index: np.ndarray,
        count_matrix,
        library_log: np.ndarray,
    ) -> torch.Tensor:
        assert self.model is not None
        gene_index_np = np.asarray(gene_index, dtype=np.int64)
        row_index_np = np.asarray(row_index, dtype=np.int64)
        target_chunk = _slice_count_chunk(count_matrix, row_index_np, gene_index_np)
        target_t = torch.as_tensor(
            target_chunk, dtype=torch.float32, device=self.device
        )
        gene_index_t = torch.as_tensor(
            gene_index_np, dtype=torch.long, device=self.device
        )
        library_t = torch.as_tensor(
            np.asarray(library_log[row_index_np], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        mu, theta = self.model.decode_counts(
            outputs["intrinsic"],
            gene_index=gene_index_t,
            library_log=library_t,
        )
        count_loss = negative_binomial_loss(target_t, mu, theta)
        if float(self.config.count_loss_weight) <= 0:
            return count_loss * 0.0
        return count_loss

    def _enforce_graph_full_batch_limit(self, n_cells: int, *, stage: str) -> None:
        limit = int(self.config.full_batch_max_cells)
        if self.config.method != "graph_autoencoder" or limit <= 0:
            return
        if n_cells > limit:
            raise ValueError(
                f"graph_autoencoder {stage} requires full-batch execution, but n_cells={int(n_cells)} exceeds "
                f"deep.full_batch_max_cells={limit}. Increase the limit explicitly or switch to a non-graph encoder."
            )

    def _collect_output_arrays_std(
        self, x_std: np.ndarray, coords_um: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        self._check_fitted()
        assert self.model is not None
        self.model.eval()
        if self.config.method == "graph_autoencoder":
            if coords_um is None:
                raise ValueError("coords_um is required for graph_autoencoder outputs.")
            self._enforce_graph_full_batch_limit(x_std.shape[0], stage="transform")
            short_graph, mid_graph = _tensor_graphs(
                np.asarray(coords_um, dtype=np.float32),
                config=self.config,
                device=self.device,
            )
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_std).to(self.device)
                outputs = self.model(
                    x_tensor, edge_index_short=short_graph, edge_index_mid=mid_graph
                )
            return {
                name: value.detach().cpu().numpy().astype(np.float32)
                for name, value in outputs.items()
            }

        batch_size = int(self.config.batch_size)
        outputs: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for x_batch_np in _iter_batches(x_std, batch_size):
                x_tensor = torch.from_numpy(x_batch_np).to(self.device)
                batch_outputs = self.model(x_tensor)
                for name, value in batch_outputs.items():
                    outputs.setdefault(name, []).append(
                        value.detach().cpu().numpy().astype(np.float32)
                    )
        return {name: np.vstack(chunks) for name, chunks in outputs.items()}

    def _collect_embedding_array_std(
        self, x_std: np.ndarray, coords_um: np.ndarray | None = None
    ) -> np.ndarray:
        self._check_fitted()
        assert self.model is not None
        if self.config.output_embedding is None:
            raise ValueError("config.output_embedding must be set before transform.")
        self.model.eval()
        if self.config.method == "graph_autoencoder":
            if coords_um is None:
                raise ValueError("coords_um is required for graph_autoencoder outputs.")
            self._enforce_graph_full_batch_limit(x_std.shape[0], stage="transform")
            short_graph, mid_graph = _tensor_graphs(
                np.asarray(coords_um, dtype=np.float32),
                config=self.config,
                device=self.device,
            )
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_std).to(self.device)
                embedding = self.model.encode(
                    x_tensor,
                    edge_index_short=short_graph,
                    edge_index_mid=mid_graph,
                    output_embedding=self.config.output_embedding,
                )
            return embedding.detach().cpu().numpy().astype(np.float32)

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch_np in _iter_batches(x_std, int(self.config.batch_size)):
                x_tensor = torch.from_numpy(x_batch_np).to(self.device)
                embedding = self.model.encode(
                    x_tensor,
                    output_embedding=self.config.output_embedding,
                )
                chunks.append(embedding.detach().cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    def fit(
        self,
        features: np.ndarray,
        coords_um: np.ndarray,
        *,
        batch: np.ndarray | None = None,
        count_matrix=None,
        seed: int = 1337,
        feature_schema_extra: dict | None = None,
    ) -> "SpatialOTFeatureEncoder":
        if self.config.method == "none":
            raise ValueError(
                "SpatialOTFeatureEncoder.fit requires an active deep feature method, not 'none'."
            )
        if self.config.output_embedding is None:
            raise ValueError(
                "SpatialOTFeatureEncoder.fit requires config.output_embedding to be set explicitly."
            )
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        x = np.asarray(features, dtype=np.float32)
        coords_um = np.asarray(coords_um, dtype=np.float32)
        self._enforce_graph_full_batch_limit(int(x.shape[0]), stage="fit")
        counts_target, library_log = self._prepare_count_targets(count_matrix)
        batch_array = np.asarray(batch) if batch is not None else None
        val_mask = _split_validation(
            coords_um=coords_um, batch=batch_array, config=self.config, seed=seed
        )
        train_mask = ~val_mask
        if not np.any(train_mask):
            train_mask[:] = True
            val_mask[:] = False

        train_features = x if bool(np.all(train_mask)) else x[train_mask]
        _, mean, std = _standardize_features(train_features)
        x_std = _apply_standardization(x, mean, std)
        needs_context_targets = (
            self.config.method == "graph_autoencoder"
            or float(self.config.context_weight) > 0.0
            or float(self.config.contrastive_weight) > 0.0
        )
        if needs_context_targets:
            context_std = _build_split_context_targets(
                coords_um=coords_um,
                features_std=x_std,
                train_mask=train_mask,
                val_mask=val_mask,
                config=self.config,
                device=self.device,
            )
        else:
            context_std = np.empty((x_std.shape[0], 0), dtype=np.float32)

        self.feature_mean = mean
        self.feature_std = std
        self.input_dim = int(x_std.shape[1])
        self.context_dim = int(context_std.shape[1])
        self.feature_schema = {
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
            "count_dim": int(self.count_dim) if self.count_dim is not None else None,
            "method": self.config.method,
            "output_embedding": self.config.output_embedding,
            "full_batch_max_cells": int(self.config.full_batch_max_cells),
            "count_layer": str(self.config.count_layer)
            if self.config.count_layer is not None
            else None,
            "count_decoder_rank": int(self.config.count_decoder_rank),
            "count_chunk_size": int(self.config.count_chunk_size),
            "count_loss_weight": float(self.config.count_loss_weight),
            "uses_absolute_coordinate_features": False,
            "uses_spatial_graph": bool(self.config.method == "graph_autoencoder"),
            "spatial_graph_construction": {
                "neighbor_k": int(self.config.neighbor_k),
                "radius_um": float(self.config.radius_um)
                if self.config.radius_um is not None
                else None,
                "short_radius_um": float(self.config.short_radius_um)
                if self.config.short_radius_um is not None
                else None,
                "mid_radius_um": float(self.config.mid_radius_um)
                if self.config.mid_radius_um is not None
                else None,
                "graph_max_neighbors": int(self.config.graph_max_neighbors),
            },
        }
        if feature_schema_extra:
            self.feature_schema.update(dict(feature_schema_extra))
        self.validation_report = {
            "mode": self.config.validation,
            "context_mode": self.config.validation_context_mode,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "held_out_batches": sorted(np.unique(batch_array[val_mask]).tolist())
            if (batch_array is not None and np.any(val_mask))
            else [],
        }

        _seed_everything(seed)
        self.model = _make_model(
            input_dim=self.input_dim,
            context_dim=self.context_dim,
            config=self.config,
            count_dim=self.count_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.history = []
        best_state = None
        best_val = float("inf")
        patience_left = int(self.config.early_stopping_patience)
        count_rng = np.random.default_rng(seed + 5000)
        train_rows = np.flatnonzero(train_mask).astype(np.int64)
        val_rows = np.flatnonzero(val_mask).astype(np.int64)
        epoch_log_context = {
            "n_obs": int(x_std.shape[0]),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "input_dim": int(self.input_dim),
            "context_dim": int(self.context_dim),
            "count_dim": int(self.count_dim) if self.count_dim is not None else None,
            "latent_dim": int(self.config.latent_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "layers": int(self.config.layers),
            "graph_layers": int(self.config.graph_layers),
            "batch_size": int(self.config.batch_size),
            "learning_rate": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "optimizer": "Adam",
            "model_parameters": int(
                sum(param.numel() for param in self.model.parameters())
            ),
            "model_trainable_parameters": int(
                sum(
                    param.numel()
                    for param in self.model.parameters()
                    if param.requires_grad
                )
            ),
            "output_embedding": str(self.config.output_embedding),
            "validation": str(self.config.validation),
            "validation_context_mode": str(self.config.validation_context_mode),
            "neighbor_k": int(self.config.neighbor_k),
            "count_chunk_size": int(self.config.count_chunk_size),
            "reconstruction_weight": float(self.config.reconstruction_weight),
            "context_weight": float(self.config.context_weight),
            "count_loss_weight": float(self.config.count_loss_weight),
            "contrastive_weight": float(self.config.contrastive_weight),
            "variance_weight": float(self.config.variance_weight),
            "decorrelation_weight": float(self.config.decorrelation_weight),
            "independence_weight": float(self.config.independence_weight),
            "gradient_clip_norm": float(self.config.gradient_clip_norm)
            if self.config.gradient_clip_norm is not None
            else None,
            "needs_context_targets": bool(needs_context_targets),
            "has_count_target": bool(counts_target is not None),
            "seed": int(seed),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "device_type": str(self.device.type),
            "cuda_device_name": (
                torch.cuda.get_device_name(
                    self.device.index or torch.cuda.current_device()
                )
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "cuda_device_total_memory_gb": (
                torch.cuda.get_device_properties(
                    self.device.index or torch.cuda.current_device()
                ).total_memory
                / (1024.0**3)
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
        }
        start_epoch = 0
        resumed = self._load_training_checkpoint(optimizer=optimizer)
        if resumed is not None:
            start_epoch, best_state, best_val, patience_left = resumed

        if self.config.method == "graph_autoencoder":
            self._enforce_graph_full_batch_limit(int(x_std.shape[0]), stage="fit")
            x_train = torch.from_numpy(x_std[train_mask]).to(self.device)
            ctx_train = torch.from_numpy(context_std[train_mask]).to(self.device)
            short_train, mid_train = _tensor_graphs(
                coords_um[train_mask], config=self.config, device=self.device
            )
            x_val = (
                torch.from_numpy(x_std[val_mask]).to(self.device)
                if np.any(val_mask)
                else None
            )
            ctx_val = (
                torch.from_numpy(context_std[val_mask]).to(self.device)
                if np.any(val_mask)
                else None
            )
            if x_val is not None:
                short_val, mid_val = _tensor_graphs(
                    coords_um[val_mask], config=self.config, device=self.device
                )
            else:
                short_val = mid_val = None
            self.feature_schema["graph_training_mode"] = "full_batch"
            self.feature_schema["short_graph"] = _graph_summary(
                short_train, int(x_train.shape[0])
            )
            self.feature_schema["mid_graph"] = _graph_summary(
                mid_train, int(x_train.shape[0])
            )

            for epoch in range(int(start_epoch), int(self.config.epochs)):
                assert self.model is not None
                epoch_start = time.perf_counter()
                self.model.train()
                outputs = self.model(
                    x_train, edge_index_short=short_train, edge_index_mid=mid_train
                )
                z = outputs[self.config.output_embedding]
                loss_recon = torch.mean((outputs["recon"] - x_train) ** 2)
                loss_ctx = torch.mean((outputs["context_pred"] - ctx_train) ** 2)
                loss_count = x_train.new_tensor(0.0)
                if (
                    counts_target is not None
                    and library_log is not None
                    and train_rows.size > 0
                ):
                    gene_index = _sample_gene_chunk(
                        int(counts_target.shape[1]),
                        int(self.config.count_chunk_size),
                        count_rng,
                    )
                    loss_count = self._count_loss_from_outputs(
                        outputs,
                        row_index=train_rows,
                        gene_index=gene_index,
                        count_matrix=counts_target,
                        library_log=library_log,
                    )
                loss_contrast = edge_contrastive_loss(outputs["context"], short_train)
                loss_var = variance_loss(z)
                loss_decorr = decorrelation_loss(z)
                loss_indep = cross_correlation_loss(
                    outputs["intrinsic"], outputs["context"]
                )
                loss = (
                    float(self.config.reconstruction_weight) * loss_recon
                    + float(self.config.context_weight) * loss_ctx
                    + float(self.config.count_loss_weight) * loss_count
                    + float(self.config.contrastive_weight) * loss_contrast
                    + float(self.config.variance_weight) * loss_var
                    + float(self.config.decorrelation_weight) * loss_decorr
                    + float(self.config.independence_weight) * loss_indep
                )
                train_loss_value = float(loss.detach().cpu())
                train_recon_value = float(loss_recon.detach().cpu())
                train_context_value = float(loss_ctx.detach().cpu())
                train_count_value = float(loss_count.detach().cpu())
                train_contrast_value = float(loss_contrast.detach().cpu())
                train_variance_value = float(loss_var.detach().cpu())
                train_decorrelation_value = float(loss_decorr.detach().cpu())
                train_independence_value = float(loss_indep.detach().cpu())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = self._clip_gradients()
                optimizer.step()

                epoch_row = {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss_value,
                    "train_recon_loss": train_recon_value,
                    "train_context_loss": train_context_value,
                    "train_count_loss": train_count_value,
                    "train_contrastive_loss": train_contrast_value,
                    "train_variance_loss": train_variance_value,
                    "train_decorrelation_loss": train_decorrelation_value,
                    "train_independence_loss": train_independence_value,
                    "train_batches": 1,
                    "train_samples": int(x_train.shape[0]),
                    "train_batch_size_min": int(x_train.shape[0]),
                    "train_batch_size_max": int(x_train.shape[0]),
                    "train_batch_size_last": int(x_train.shape[0]),
                }
                if grad_norm is not None:
                    epoch_row["train_grad_norm"] = float(grad_norm)
                current_val = None
                if (
                    x_val is not None
                    and ctx_val is not None
                    and short_val is not None
                    and mid_val is not None
                ):
                    self.model.eval()
                    with torch.no_grad():
                        outputs_val = self.model(
                            x_val, edge_index_short=short_val, edge_index_mid=mid_val
                        )
                        z_val = outputs_val[self.config.output_embedding]
                        val_recon = torch.mean((outputs_val["recon"] - x_val) ** 2)
                        val_ctx = torch.mean(
                            (outputs_val["context_pred"] - ctx_val) ** 2
                        )
                        val_count = x_val.new_tensor(0.0)
                        if (
                            counts_target is not None
                            and library_log is not None
                            and val_rows.size > 0
                        ):
                            gene_index_val = _sample_gene_chunk(
                                int(counts_target.shape[1]),
                                int(self.config.count_chunk_size),
                                count_rng,
                            )
                            val_count = self._count_loss_from_outputs(
                                outputs_val,
                                row_index=val_rows,
                                gene_index=gene_index_val,
                                count_matrix=counts_target,
                                library_log=library_log,
                            )
                        val_contrast = edge_contrastive_loss(
                            outputs_val["context"], short_val
                        )
                        val_var = variance_loss(z_val)
                        val_decorr = decorrelation_loss(z_val)
                        val_indep = cross_correlation_loss(
                            outputs_val["intrinsic"], outputs_val["context"]
                        )
                        val_recon_value = float(val_recon.detach().cpu())
                        val_context_value = float(val_ctx.detach().cpu())
                        val_count_value = float(val_count.detach().cpu())
                        val_contrast_value = float(val_contrast.detach().cpu())
                        val_variance_value = float(val_var.detach().cpu())
                        val_decorrelation_value = float(val_decorr.detach().cpu())
                        val_independence_value = float(val_indep.detach().cpu())
                        val_loss = (
                            float(self.config.reconstruction_weight) * val_recon
                            + float(self.config.context_weight) * val_ctx
                            + float(self.config.count_loss_weight) * val_count
                            + float(self.config.contrastive_weight) * val_contrast
                            + float(self.config.variance_weight) * val_var
                            + float(self.config.decorrelation_weight) * val_decorr
                            + float(self.config.independence_weight) * val_indep
                        )
                        current_val = float(val_loss.detach().cpu())
                        epoch_row["val_loss"] = current_val
                        epoch_row["val_recon_loss"] = val_recon_value
                        epoch_row["val_context_loss"] = val_context_value
                        epoch_row["val_count_loss"] = val_count_value
                        epoch_row["val_contrastive_loss"] = val_contrast_value
                        epoch_row["val_variance_loss"] = val_variance_value
                        epoch_row["val_decorrelation_loss"] = val_decorrelation_value
                        epoch_row["val_independence_loss"] = val_independence_value
                        epoch_row["val_batches"] = 1
                        epoch_row["val_samples"] = int(x_val.shape[0])
                        epoch_row["val_batch_size_min"] = int(x_val.shape[0])
                        epoch_row["val_batch_size_max"] = int(x_val.shape[0])
                        epoch_row["val_batch_size_last"] = int(x_val.shape[0])
                epoch_duration = time.perf_counter() - epoch_start
                epoch_row["epoch_duration_sec"] = float(epoch_duration)
                epoch_row["train_samples_per_sec"] = float(
                    int(x_train.shape[0]) / max(epoch_duration, 1e-9)
                )
                if int(epoch_row.get("val_samples", 0)) > 0:
                    epoch_row["val_samples_per_sec"] = float(
                        int(epoch_row["val_samples"]) / max(epoch_duration, 1e-9)
                    )
                epoch_row["current_learning_rate"] = float(
                    optimizer.param_groups[0].get("lr", self.config.learning_rate)
                )
                epoch_row["best_val_loss_before_epoch"] = (
                    None if best_val == float("inf") else float(best_val)
                )
                epoch_row["early_stopping_patience_left_before_epoch"] = int(
                    patience_left
                )
                self.history.append(epoch_row)
                _log_epoch_record(
                    epoch_row,
                    total_epochs=int(self.config.epochs),
                    method=self.config.method,
                    device=self.device,
                    context=epoch_log_context,
                )
                stop_training = False
                if current_val is None:
                    best_state = copy.deepcopy(self.model.state_dict())
                elif current_val < best_val - float(self.config.min_delta):
                    best_val = current_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = int(self.config.early_stopping_patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        stop_training = True
                self._save_training_checkpoint(
                    epoch=int(epoch + 1),
                    optimizer=optimizer,
                    best_state=best_state,
                    best_val=best_val,
                    patience_left=patience_left,
                )
                if stop_training:
                    break
        else:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            x_train_np = x_std if bool(np.all(train_mask)) else x_std[train_mask]
            context_train_np = (
                context_std if bool(np.all(train_mask)) else context_std[train_mask]
            )
            train_dataset = TensorDataset(
                torch.from_numpy(x_train_np),
                torch.from_numpy(context_train_np),
                torch.from_numpy(train_rows.astype(np.int64)),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=min(int(self.config.batch_size), max(len(train_dataset), 1)),
                shuffle=True,
                drop_last=False,
                generator=generator,
            )
            x_val = x_std[val_mask] if np.any(val_mask) else None
            c_val = context_std[val_mask] if np.any(val_mask) else None

            for epoch in range(int(start_epoch), int(self.config.epochs)):
                assert self.model is not None
                epoch_start = time.perf_counter()
                self.model.train()
                loss_accum = 0.0
                n_train = 0
                train_batches = 0
                train_batch_size_min = None
                train_batch_size_max = 0
                train_batch_size_last = 0
                recon_loss_accum = 0.0
                context_loss_accum = 0.0
                count_loss_accum = 0.0
                variance_loss_accum = 0.0
                decorrelation_loss_accum = 0.0
                independence_loss_accum = 0.0
                grad_norm_accum = 0.0
                grad_norm_max = 0.0
                needs_context_loss = float(self.config.context_weight) > 0.0
                needs_independence_loss = float(self.config.independence_weight) > 0.0
                for batch_x, batch_ctx, batch_rows in train_loader:
                    batch_x = batch_x.to(self.device)
                    if needs_context_loss:
                        batch_ctx = batch_ctx.to(self.device)
                    batch_rows_np = (
                        batch_rows.detach().cpu().numpy().astype(np.int64, copy=False)
                    )
                    outputs = self.model(batch_x, predict_context=needs_context_loss)
                    z = outputs[self.config.output_embedding]
                    loss_recon = torch.mean((outputs["recon"] - batch_x) ** 2)
                    loss_ctx = (
                        torch.mean((outputs["context_pred"] - batch_ctx) ** 2)
                        if needs_context_loss
                        else batch_x.new_tensor(0.0)
                    )
                    loss_count = batch_x.new_tensor(0.0)
                    if (
                        counts_target is not None
                        and library_log is not None
                        and batch_rows_np.size > 0
                    ):
                        gene_index = _sample_gene_chunk(
                            int(counts_target.shape[1]),
                            int(self.config.count_chunk_size),
                            count_rng,
                        )
                        loss_count = self._count_loss_from_outputs(
                            outputs,
                            row_index=batch_rows_np,
                            gene_index=gene_index,
                            count_matrix=counts_target,
                            library_log=library_log,
                        )
                    loss_var = variance_loss(z)
                    loss_decorr = decorrelation_loss(z)
                    loss_indep = (
                        cross_correlation_loss(outputs["intrinsic"], outputs["context"])
                        if needs_independence_loss
                        else batch_x.new_tensor(0.0)
                    )
                    loss = (
                        float(self.config.reconstruction_weight) * loss_recon
                        + float(self.config.context_weight) * loss_ctx
                        + float(self.config.count_loss_weight) * loss_count
                        + float(self.config.variance_weight) * loss_var
                        + float(self.config.decorrelation_weight) * loss_decorr
                        + float(self.config.independence_weight) * loss_indep
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = self._clip_gradients()
                    optimizer.step()
                    batch_n = int(batch_x.shape[0])
                    train_batches += 1
                    train_batch_size_last = batch_n
                    train_batch_size_max = max(train_batch_size_max, batch_n)
                    train_batch_size_min = (
                        batch_n
                        if train_batch_size_min is None
                        else min(train_batch_size_min, batch_n)
                    )
                    loss_accum += float(loss.detach().cpu()) * batch_n
                    recon_loss_accum += float(loss_recon.detach().cpu()) * batch_n
                    context_loss_accum += float(loss_ctx.detach().cpu()) * batch_n
                    count_loss_accum += float(loss_count.detach().cpu()) * batch_n
                    variance_loss_accum += float(loss_var.detach().cpu()) * batch_n
                    decorrelation_loss_accum += (
                        float(loss_decorr.detach().cpu()) * batch_n
                    )
                    independence_loss_accum += float(loss_indep.detach().cpu()) * batch_n
                    if grad_norm is not None:
                        grad_norm_accum += float(grad_norm) * batch_n
                        grad_norm_max = max(grad_norm_max, float(grad_norm))
                    n_train += batch_n

                epoch_row = {
                    "epoch": float(epoch + 1),
                    "train_loss": float(loss_accum / max(n_train, 1)),
                    "train_recon_loss": float(recon_loss_accum / max(n_train, 1)),
                    "train_context_loss": float(context_loss_accum / max(n_train, 1)),
                    "train_count_loss": float(count_loss_accum / max(n_train, 1)),
                    "train_variance_loss": float(variance_loss_accum / max(n_train, 1)),
                    "train_decorrelation_loss": float(
                        decorrelation_loss_accum / max(n_train, 1)
                    ),
                    "train_independence_loss": float(
                        independence_loss_accum / max(n_train, 1)
                    ),
                    "train_batches": int(train_batches),
                    "train_samples": int(n_train),
                    "train_batch_size_min": int(train_batch_size_min or 0),
                    "train_batch_size_max": int(train_batch_size_max),
                    "train_batch_size_last": int(train_batch_size_last),
                }
                if self.config.gradient_clip_norm is not None and float(
                    self.config.gradient_clip_norm
                ) > 0.0:
                    epoch_row["train_grad_norm_mean"] = float(
                        grad_norm_accum / max(n_train, 1)
                    )
                    epoch_row["train_grad_norm_max"] = float(grad_norm_max)
                current_val = None
                if x_val is not None and c_val is not None and x_val.shape[0] > 0:
                    self.model.eval()
                    val_loss_accum = 0.0
                    val_recon_accum = 0.0
                    val_context_accum = 0.0
                    val_count_accum = 0.0
                    val_variance_accum = 0.0
                    val_decorrelation_accum = 0.0
                    val_independence_accum = 0.0
                    val_batches = 0
                    val_batch_size_min = None
                    val_batch_size_max = 0
                    val_batch_size_last = 0
                    n_val = 0
                    with torch.no_grad():
                        for start, (x_batch_np, c_batch_np) in enumerate(
                            zip(
                                _iter_batches(x_val, self.config.batch_size),
                                _iter_batches(c_val, self.config.batch_size),
                                strict=False,
                            )
                        ):
                            x_batch = torch.from_numpy(x_batch_np).to(self.device)
                            if needs_context_loss:
                                c_batch = torch.from_numpy(c_batch_np).to(self.device)
                            row_batch_np = val_rows[
                                start * int(self.config.batch_size) : start
                                * int(self.config.batch_size)
                                + x_batch_np.shape[0]
                            ]
                            outputs = self.model(
                                x_batch, predict_context=needs_context_loss
                            )
                            z_val = outputs[self.config.output_embedding]
                            val_recon = torch.mean((outputs["recon"] - x_batch) ** 2)
                            val_ctx = (
                                torch.mean((outputs["context_pred"] - c_batch) ** 2)
                                if needs_context_loss
                                else x_batch.new_tensor(0.0)
                            )
                            val_count = x_batch.new_tensor(0.0)
                            if (
                                counts_target is not None
                                and library_log is not None
                                and row_batch_np.size > 0
                            ):
                                gene_index_val = _sample_gene_chunk(
                                    int(counts_target.shape[1]),
                                    int(self.config.count_chunk_size),
                                    count_rng,
                                )
                                val_count = self._count_loss_from_outputs(
                                    outputs,
                                    row_index=row_batch_np,
                                    gene_index=gene_index_val,
                                    count_matrix=counts_target,
                                    library_log=library_log,
                                )
                            val_var = variance_loss(z_val)
                            val_decorr = decorrelation_loss(z_val)
                            val_indep = (
                                cross_correlation_loss(
                                    outputs["intrinsic"], outputs["context"]
                                )
                                if needs_independence_loss
                                else x_batch.new_tensor(0.0)
                            )
                            val_loss = (
                                float(self.config.reconstruction_weight) * val_recon
                                + float(self.config.context_weight) * val_ctx
                                + float(self.config.count_loss_weight) * val_count
                                + float(self.config.variance_weight) * val_var
                                + float(self.config.decorrelation_weight) * val_decorr
                                + float(self.config.independence_weight) * val_indep
                            )
                            val_loss_accum += float(val_loss.detach().cpu()) * int(
                                x_batch.shape[0]
                            )
                            val_n = int(x_batch.shape[0])
                            val_batches += 1
                            val_batch_size_last = val_n
                            val_batch_size_max = max(val_batch_size_max, val_n)
                            val_batch_size_min = (
                                val_n
                                if val_batch_size_min is None
                                else min(val_batch_size_min, val_n)
                            )
                            val_recon_accum += float(val_recon.detach().cpu()) * val_n
                            val_context_accum += float(val_ctx.detach().cpu()) * val_n
                            val_count_accum += float(val_count.detach().cpu()) * val_n
                            val_variance_accum += float(val_var.detach().cpu()) * val_n
                            val_decorrelation_accum += (
                                float(val_decorr.detach().cpu()) * val_n
                            )
                            val_independence_accum += (
                                float(val_indep.detach().cpu()) * val_n
                            )
                            n_val += val_n
                    current_val = float(val_loss_accum / max(n_val, 1))
                    epoch_row["val_loss"] = current_val
                    epoch_row["val_recon_loss"] = float(
                        val_recon_accum / max(n_val, 1)
                    )
                    epoch_row["val_context_loss"] = float(
                        val_context_accum / max(n_val, 1)
                    )
                    epoch_row["val_count_loss"] = float(val_count_accum / max(n_val, 1))
                    epoch_row["val_variance_loss"] = float(
                        val_variance_accum / max(n_val, 1)
                    )
                    epoch_row["val_decorrelation_loss"] = float(
                        val_decorrelation_accum / max(n_val, 1)
                    )
                    epoch_row["val_independence_loss"] = float(
                        val_independence_accum / max(n_val, 1)
                    )
                    epoch_row["val_batches"] = int(val_batches)
                    epoch_row["val_samples"] = int(n_val)
                    epoch_row["val_batch_size_min"] = int(val_batch_size_min or 0)
                    epoch_row["val_batch_size_max"] = int(val_batch_size_max)
                    epoch_row["val_batch_size_last"] = int(val_batch_size_last)
                epoch_duration = time.perf_counter() - epoch_start
                epoch_row["epoch_duration_sec"] = float(epoch_duration)
                epoch_row["train_samples_per_sec"] = float(
                    n_train / max(epoch_duration, 1e-9)
                )
                if int(epoch_row.get("val_samples", 0)) > 0:
                    epoch_row["val_samples_per_sec"] = float(
                        int(epoch_row["val_samples"]) / max(epoch_duration, 1e-9)
                    )
                epoch_row["current_learning_rate"] = float(
                    optimizer.param_groups[0].get("lr", self.config.learning_rate)
                )
                epoch_row["best_val_loss_before_epoch"] = (
                    None if best_val == float("inf") else float(best_val)
                )
                epoch_row["early_stopping_patience_left_before_epoch"] = int(
                    patience_left
                )
                self.history.append(epoch_row)
                _log_epoch_record(
                    epoch_row,
                    total_epochs=int(self.config.epochs),
                    method=self.config.method,
                    device=self.device,
                    context=epoch_log_context,
                )
                stop_training = False
                if current_val is None:
                    best_state = copy.deepcopy(self.model.state_dict())
                elif current_val < best_val - float(self.config.min_delta):
                    best_val = current_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = int(self.config.early_stopping_patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        stop_training = True
                self._save_training_checkpoint(
                    epoch=int(epoch + 1),
                    optimizer=optimizer,
                    best_state=best_state,
                    best_val=best_val,
                    patience_left=patience_left,
                )
                if stop_training:
                    break

        if bool(self.config.restore_best) and best_state is not None:
            assert self.model is not None
            self.model.load_state_dict(best_state)
        max_diagnostic_elements = _env_int(
            "SPATIAL_OT_DEEP_MAX_DIAGNOSTIC_ELEMENTS", 1_000_000_000
        )
        skip_full_output_diagnostics = int(x_std.size) > int(max_diagnostic_elements)
        outputs_full: dict[str, np.ndarray] | None = None
        if skip_full_output_diagnostics:
            self.latent_diagnostics = {
                "selected_embedding": str(self.config.output_embedding),
                "full_output_diagnostics_skipped": True,
                "full_output_diagnostics_skip_reason": (
                    "feature matrix has more elements than "
                    "SPATIAL_OT_DEEP_MAX_DIAGNOSTIC_ELEMENTS"
                ),
                "feature_elements": int(x_std.size),
                "max_diagnostic_elements": int(max_diagnostic_elements),
            }
        else:
            outputs_full = self._collect_output_arrays_std(x_std, coords_um=coords_um)
            self.latent_diagnostics = _latent_diagnostics(
                outputs_full,
                x_std=x_std,
                context_std=context_std,
                coords_um=coords_um,
                selected_embedding=self.config.output_embedding,
            )
        if (
            counts_target is not None
            and library_log is not None
            and self.model is not None
            and outputs_full is not None
        ):
            diag_rng = np.random.default_rng(seed + 9000)
            gene_index_diag = _sample_gene_chunk(
                int(counts_target.shape[1]), int(self.config.count_chunk_size), diag_rng
            )
            with torch.no_grad():
                intrinsic_t = torch.as_tensor(
                    outputs_full["intrinsic"], dtype=torch.float32, device=self.device
                )
                gene_index_t = torch.as_tensor(
                    gene_index_diag, dtype=torch.long, device=self.device
                )
                library_t = torch.as_tensor(
                    library_log, dtype=torch.float32, device=self.device
                )
                mu_diag, theta_diag = self.model.decode_counts(
                    intrinsic_t,
                    gene_index=gene_index_t,
                    library_log=library_t,
                )
                count_diag = negative_binomial_loss(
                    torch.as_tensor(
                        _slice_count_chunk(
                            counts_target,
                            np.arange(x.shape[0], dtype=np.int64),
                            gene_index_diag,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    mu_diag,
                    theta_diag,
                )
            self.latent_diagnostics["count_reconstruction_nb_loss"] = float(
                count_diag.detach().cpu()
            )
            self.latent_diagnostics["count_reconstruction_gene_chunk_size"] = int(
                gene_index_diag.shape[0]
            )
            self.latent_diagnostics["count_target_dim"] = int(counts_target.shape[1])
        self.latent_diagnostics["runtime_memory"] = _runtime_memory_snapshot(
            self.device
        )
        return self

    def _validate_transform_schema(
        self,
        *,
        input_obsm_key: str | None = None,
        coordinate_keys: tuple[str, str] | None = None,
        spatial_scale: float | None = None,
    ) -> None:
        expected_dim = self.feature_schema.get("input_dim")
        if (
            expected_dim is not None
            and self.input_dim is not None
            and int(expected_dim) != int(self.input_dim)
        ):
            raise ValueError(
                "Saved feature schema is inconsistent with the loaded encoder input dimension."
            )
        expected_key = self.feature_schema.get("input_obsm_key")
        if (
            expected_key is not None
            and input_obsm_key is not None
            and str(expected_key) != str(input_obsm_key)
        ):
            raise ValueError(
                f"Input obsm key mismatch: encoder expects '{expected_key}', got '{input_obsm_key}'."
            )
        expected_coord_keys = self.feature_schema.get("coordinate_keys")
        if (
            expected_coord_keys is not None
            and coordinate_keys is not None
            and list(coordinate_keys) != list(expected_coord_keys)
        ):
            raise ValueError(
                f"Coordinate key mismatch: encoder expects {expected_coord_keys}, got {list(coordinate_keys)}."
            )
        expected_scale = self.feature_schema.get("spatial_scale")
        if (
            expected_scale is not None
            and spatial_scale is not None
            and not np.isclose(float(expected_scale), float(spatial_scale), atol=1e-8)
        ):
            raise ValueError(
                f"Spatial scale mismatch: encoder expects {float(expected_scale)}, got {float(spatial_scale)}."
            )

    def transform(
        self,
        features: np.ndarray,
        coords_um: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        self._validate_transform_schema()
        if self.config.output_embedding is None:
            raise ValueError(
                "SpatialOTFeatureEncoder.transform requires config.output_embedding to be set explicitly."
            )
        assert (
            self.feature_mean is not None
            and self.feature_std is not None
            and self.model is not None
            and self.input_dim is not None
        )
        x_std = _apply_standardization(features, self.feature_mean, self.feature_std)
        if x_std.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {x_std.shape[1]}."
            )
        if batch_size is not None and self.config.method != "graph_autoencoder":
            original_batch_size = self.config.batch_size
            self.config.batch_size = int(batch_size)
            try:
                embedding = self._collect_embedding_array_std(x_std, coords_um=coords_um)
            finally:
                self.config.batch_size = original_batch_size
        else:
            embedding = self._collect_embedding_array_std(x_std, coords_um=coords_um)
        return np.asarray(embedding, dtype=np.float32)

    def fit_transform(
        self,
        features: np.ndarray,
        coords_um: np.ndarray,
        *,
        batch: np.ndarray | None = None,
        count_matrix=None,
        seed: int = 1337,
        feature_schema_extra: dict | None = None,
    ) -> DeepFeatureResult:
        self.fit(
            features=features,
            coords_um=coords_um,
            batch=batch,
            count_matrix=count_matrix,
            seed=seed,
            feature_schema_extra=feature_schema_extra,
        )
        embedding = self.transform(features, coords_um=coords_um)
        return DeepFeatureResult(
            embedding=embedding,
            history=list(self.history),
            config=asdict(self.config),
            feature_mean=np.asarray(self.feature_mean, dtype=np.float32),
            feature_std=np.asarray(self.feature_std, dtype=np.float32),
            feature_schema=dict(self.feature_schema),
            validation_report=dict(self.validation_report),
            latent_diagnostics=dict(self.latent_diagnostics),
            model_path=None,
        )

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        assert (
            self.model is not None
            and self.feature_mean is not None
            and self.feature_std is not None
            and self.input_dim is not None
        )
        save_encoder_bundle(
            path,
            state_dict=self.model.state_dict(),
            metadata={
                "config": asdict(self.config),
                "input_dim": int(self.input_dim),
                "context_dim": int(self.context_dim or 0),
                "count_dim": int(self.count_dim or 0),
                "history": self.history,
                "feature_schema": self.feature_schema,
                "validation_report": self.validation_report,
                "latent_diagnostics": self.latent_diagnostics,
            },
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | None = None,
        device: str | None = None,
    ) -> "SpatialOTFeatureEncoder":
        resolved_map_location = map_location or "cpu"
        state_dict, metadata, feature_mean, feature_std = load_encoder_bundle(
            path, map_location=resolved_map_location
        )
        config = DeepFeatureConfig(**metadata["config"])
        if device is not None:
            config.device = device
        elif str(resolved_map_location) == "cpu":
            config.device = "cpu"
        encoder = cls(config=config)
        encoder.feature_mean = feature_mean
        encoder.feature_std = feature_std
        encoder.input_dim = int(metadata["input_dim"])
        encoder.context_dim = int(metadata.get("context_dim", encoder.input_dim * 2))
        encoder.count_dim = int(metadata.get("count_dim", 0)) or None
        encoder.model = _make_model(
            input_dim=encoder.input_dim,
            context_dim=encoder.context_dim,
            config=config,
            count_dim=encoder.count_dim,
        ).to(encoder.device)
        encoder.model.load_state_dict(state_dict)
        encoder.history = list(metadata.get("history", []))
        encoder.feature_schema = dict(metadata.get("feature_schema", {}))
        encoder.validation_report = dict(metadata.get("validation_report", {}))
        encoder.latent_diagnostics = dict(metadata.get("latent_diagnostics", {}))
        return encoder


def fit_deep_features(
    features: np.ndarray,
    coords_um: np.ndarray,
    *,
    config: DeepFeatureConfig,
    batch: np.ndarray | None = None,
    count_matrix=None,
    seed: int = 1337,
    save_path: str | Path | None = None,
    feature_schema_extra: dict | None = None,
) -> DeepFeatureResult:
    encoder = SpatialOTFeatureEncoder(config=config)
    result = encoder.fit_transform(
        features=features,
        coords_um=coords_um,
        batch=batch,
        count_matrix=count_matrix,
        seed=seed,
        feature_schema_extra=feature_schema_extra,
    )
    if save_path is not None:
        encoder.save(save_path)
        result.model_path = str(Path(save_path))
    return result


def save_deep_feature_history(
    history: list[dict[str, float]], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        import json

        destination.write_text(json.dumps(history, indent=2))
        return
    import pandas as pd

    pd.DataFrame(history).to_csv(destination, index=False)


__all__ = [
    "DeepFeatureResult",
    "SpatialOTFeatureEncoder",
    "_split_validation",
    "fit_deep_features",
    "save_deep_feature_history",
]
