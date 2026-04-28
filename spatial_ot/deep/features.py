from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from pathlib import Path

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
from .checkpoint import load_encoder_bundle, save_encoder_bundle
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
        counts = count_matrix.tocsr(copy=True) if sparse.issparse(count_matrix) else np.asarray(count_matrix)
        library = np.log(np.maximum(_row_sums(counts), 1.0)).astype(np.float32, copy=False)
        self.count_dim = int(counts.shape[1])
        return counts, library

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
        target_t = torch.as_tensor(target_chunk, dtype=torch.float32, device=self.device)
        gene_index_t = torch.as_tensor(gene_index_np, dtype=torch.long, device=self.device)
        library_t = torch.as_tensor(np.asarray(library_log[row_index_np], dtype=np.float32), dtype=torch.float32, device=self.device)
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

    def _collect_output_arrays_std(self, x_std: np.ndarray, coords_um: np.ndarray | None = None) -> dict[str, np.ndarray]:
        self._check_fitted()
        assert self.model is not None
        self.model.eval()
        if self.config.method == "graph_autoencoder":
            if coords_um is None:
                raise ValueError("coords_um is required for graph_autoencoder outputs.")
            self._enforce_graph_full_batch_limit(x_std.shape[0], stage="transform")
            short_graph, mid_graph = _tensor_graphs(np.asarray(coords_um, dtype=np.float32), config=self.config, device=self.device)
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_std).to(self.device)
                outputs = self.model(x_tensor, edge_index_short=short_graph, edge_index_mid=mid_graph)
            return {name: value.detach().cpu().numpy().astype(np.float32) for name, value in outputs.items()}

        batch_size = int(self.config.batch_size)
        outputs: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for x_batch_np in _iter_batches(x_std, batch_size):
                x_tensor = torch.from_numpy(x_batch_np).to(self.device)
                batch_outputs = self.model(x_tensor)
                for name, value in batch_outputs.items():
                    outputs.setdefault(name, []).append(value.detach().cpu().numpy().astype(np.float32))
        return {name: np.vstack(chunks) for name, chunks in outputs.items()}

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
            raise ValueError("SpatialOTFeatureEncoder.fit requires an active deep feature method, not 'none'.")
        if self.config.output_embedding is None:
            raise ValueError("SpatialOTFeatureEncoder.fit requires config.output_embedding to be set explicitly.")
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
        val_mask = _split_validation(coords_um=coords_um, batch=batch_array, config=self.config, seed=seed)
        train_mask = ~val_mask
        if not np.any(train_mask):
            train_mask[:] = True
            val_mask[:] = False

        _, mean, std = _standardize_features(x[train_mask])
        x_std = _apply_standardization(x, mean, std)
        context_std = _build_split_context_targets(
            coords_um=coords_um,
            features_std=x_std,
            train_mask=train_mask,
            val_mask=val_mask,
            config=self.config,
            device=self.device,
        )

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
            "count_layer": str(self.config.count_layer) if self.config.count_layer is not None else None,
            "count_decoder_rank": int(self.config.count_decoder_rank),
            "count_chunk_size": int(self.config.count_chunk_size),
            "count_loss_weight": float(self.config.count_loss_weight),
            "uses_absolute_coordinate_features": False,
            "uses_spatial_graph": bool(self.config.method == "graph_autoencoder"),
            "spatial_graph_construction": {
                "neighbor_k": int(self.config.neighbor_k),
                "radius_um": float(self.config.radius_um) if self.config.radius_um is not None else None,
                "short_radius_um": float(self.config.short_radius_um) if self.config.short_radius_um is not None else None,
                "mid_radius_um": float(self.config.mid_radius_um) if self.config.mid_radius_um is not None else None,
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
            "held_out_batches": sorted(np.unique(batch_array[val_mask]).tolist()) if (batch_array is not None and np.any(val_mask)) else [],
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

        if self.config.method == "graph_autoencoder":
            self._enforce_graph_full_batch_limit(int(x_std.shape[0]), stage="fit")
            x_train = torch.from_numpy(x_std[train_mask]).to(self.device)
            ctx_train = torch.from_numpy(context_std[train_mask]).to(self.device)
            short_train, mid_train = _tensor_graphs(coords_um[train_mask], config=self.config, device=self.device)
            x_val = torch.from_numpy(x_std[val_mask]).to(self.device) if np.any(val_mask) else None
            ctx_val = torch.from_numpy(context_std[val_mask]).to(self.device) if np.any(val_mask) else None
            if x_val is not None:
                short_val, mid_val = _tensor_graphs(coords_um[val_mask], config=self.config, device=self.device)
            else:
                short_val = mid_val = None
            self.feature_schema["graph_training_mode"] = "full_batch"
            self.feature_schema["short_graph"] = _graph_summary(short_train, int(x_train.shape[0]))
            self.feature_schema["mid_graph"] = _graph_summary(mid_train, int(x_train.shape[0]))

            for epoch in range(int(self.config.epochs)):
                assert self.model is not None
                self.model.train()
                outputs = self.model(x_train, edge_index_short=short_train, edge_index_mid=mid_train)
                z = outputs[self.config.output_embedding]
                loss_recon = torch.mean((outputs["recon"] - x_train) ** 2)
                loss_ctx = torch.mean((outputs["context_pred"] - ctx_train) ** 2)
                loss_count = x_train.new_tensor(0.0)
                if counts_target is not None and library_log is not None and train_rows.size > 0:
                    gene_index = _sample_gene_chunk(int(counts_target.shape[1]), int(self.config.count_chunk_size), count_rng)
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
                loss_indep = cross_correlation_loss(outputs["intrinsic"], outputs["context"])
                loss = (
                    float(self.config.reconstruction_weight) * loss_recon
                    + float(self.config.context_weight) * loss_ctx
                    + float(self.config.count_loss_weight) * loss_count
                    + float(self.config.contrastive_weight) * loss_contrast
                    + float(self.config.variance_weight) * loss_var
                    + float(self.config.decorrelation_weight) * loss_decorr
                    + float(self.config.independence_weight) * loss_indep
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_row = {
                    "epoch": float(epoch + 1),
                    "train_loss": float(loss.detach().cpu()),
                    "train_recon_loss": float(loss_recon.detach().cpu()),
                    "train_context_loss": float(loss_ctx.detach().cpu()),
                    "train_count_loss": float(loss_count.detach().cpu()),
                    "train_independence_loss": float(loss_indep.detach().cpu()),
                }
                current_val = None
                if x_val is not None and ctx_val is not None and short_val is not None and mid_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        outputs_val = self.model(x_val, edge_index_short=short_val, edge_index_mid=mid_val)
                        z_val = outputs_val[self.config.output_embedding]
                        val_recon = torch.mean((outputs_val["recon"] - x_val) ** 2)
                        val_ctx = torch.mean((outputs_val["context_pred"] - ctx_val) ** 2)
                        val_count = x_val.new_tensor(0.0)
                        if counts_target is not None and library_log is not None and val_rows.size > 0:
                            gene_index_val = _sample_gene_chunk(int(counts_target.shape[1]), int(self.config.count_chunk_size), count_rng)
                            val_count = self._count_loss_from_outputs(
                                outputs_val,
                                row_index=val_rows,
                                gene_index=gene_index_val,
                                count_matrix=counts_target,
                                library_log=library_log,
                            )
                        val_contrast = edge_contrastive_loss(outputs_val["context"], short_val)
                        val_var = variance_loss(z_val)
                        val_decorr = decorrelation_loss(z_val)
                        val_indep = cross_correlation_loss(outputs_val["intrinsic"], outputs_val["context"])
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
                        epoch_row["val_recon_loss"] = float(val_recon.detach().cpu())
                        epoch_row["val_context_loss"] = float(val_ctx.detach().cpu())
                        epoch_row["val_count_loss"] = float(val_count.detach().cpu())
                        epoch_row["val_independence_loss"] = float(val_indep.detach().cpu())
                self.history.append(epoch_row)
                if current_val is None:
                    best_state = copy.deepcopy(self.model.state_dict())
                elif current_val < best_val - float(self.config.min_delta):
                    best_val = current_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = int(self.config.early_stopping_patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
        else:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            train_dataset = TensorDataset(
                torch.from_numpy(x_std[train_mask]),
                torch.from_numpy(context_std[train_mask]),
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

            for epoch in range(int(self.config.epochs)):
                assert self.model is not None
                self.model.train()
                loss_accum = 0.0
                n_train = 0
                count_loss_accum = 0.0
                for batch_x, batch_ctx, batch_rows in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_ctx = batch_ctx.to(self.device)
                    batch_rows_np = batch_rows.detach().cpu().numpy().astype(np.int64, copy=False)
                    outputs = self.model(batch_x)
                    z = outputs[self.config.output_embedding]
                    loss_recon = torch.mean((outputs["recon"] - batch_x) ** 2)
                    loss_ctx = torch.mean((outputs["context_pred"] - batch_ctx) ** 2)
                    loss_count = batch_x.new_tensor(0.0)
                    if counts_target is not None and library_log is not None and batch_rows_np.size > 0:
                        gene_index = _sample_gene_chunk(int(counts_target.shape[1]), int(self.config.count_chunk_size), count_rng)
                        loss_count = self._count_loss_from_outputs(
                            outputs,
                            row_index=batch_rows_np,
                            gene_index=gene_index,
                            count_matrix=counts_target,
                            library_log=library_log,
                        )
                    loss_var = variance_loss(z)
                    loss_decorr = decorrelation_loss(z)
                    loss_indep = cross_correlation_loss(outputs["intrinsic"], outputs["context"])
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
                    optimizer.step()
                    loss_accum += float(loss.detach().cpu()) * int(batch_x.shape[0])
                    count_loss_accum += float(loss_count.detach().cpu()) * int(batch_x.shape[0])
                    n_train += int(batch_x.shape[0])

                epoch_row = {
                    "epoch": float(epoch + 1),
                    "train_loss": float(loss_accum / max(n_train, 1)),
                    "train_count_loss": float(count_loss_accum / max(n_train, 1)),
                }
                current_val = None
                if x_val is not None and c_val is not None and x_val.shape[0] > 0:
                    self.model.eval()
                    val_loss_accum = 0.0
                    val_count_accum = 0.0
                    n_val = 0
                    with torch.no_grad():
                        for start, (x_batch_np, c_batch_np) in enumerate(zip(_iter_batches(x_val, self.config.batch_size), _iter_batches(c_val, self.config.batch_size), strict=False)):
                            x_batch = torch.from_numpy(x_batch_np).to(self.device)
                            c_batch = torch.from_numpy(c_batch_np).to(self.device)
                            row_batch_np = val_rows[start * int(self.config.batch_size) : start * int(self.config.batch_size) + x_batch_np.shape[0]]
                            outputs = self.model(x_batch)
                            z_val = outputs[self.config.output_embedding]
                            val_recon = torch.mean((outputs["recon"] - x_batch) ** 2)
                            val_ctx = torch.mean((outputs["context_pred"] - c_batch) ** 2)
                            val_count = x_batch.new_tensor(0.0)
                            if counts_target is not None and library_log is not None and row_batch_np.size > 0:
                                gene_index_val = _sample_gene_chunk(int(counts_target.shape[1]), int(self.config.count_chunk_size), count_rng)
                                val_count = self._count_loss_from_outputs(
                                    outputs,
                                    row_index=row_batch_np,
                                    gene_index=gene_index_val,
                                    count_matrix=counts_target,
                                    library_log=library_log,
                                )
                            val_var = variance_loss(z_val)
                            val_decorr = decorrelation_loss(z_val)
                            val_indep = cross_correlation_loss(outputs["intrinsic"], outputs["context"])
                            val_loss = (
                                float(self.config.reconstruction_weight) * val_recon
                                + float(self.config.context_weight) * val_ctx
                                + float(self.config.count_loss_weight) * val_count
                                + float(self.config.variance_weight) * val_var
                                + float(self.config.decorrelation_weight) * val_decorr
                                + float(self.config.independence_weight) * val_indep
                            )
                            val_loss_accum += float(val_loss.detach().cpu()) * int(x_batch.shape[0])
                            val_count_accum += float(val_count.detach().cpu()) * int(x_batch.shape[0])
                            n_val += int(x_batch.shape[0])
                    current_val = float(val_loss_accum / max(n_val, 1))
                    epoch_row["val_loss"] = current_val
                    epoch_row["val_count_loss"] = float(val_count_accum / max(n_val, 1))
                self.history.append(epoch_row)
                if current_val is None:
                    best_state = copy.deepcopy(self.model.state_dict())
                elif current_val < best_val - float(self.config.min_delta):
                    best_val = current_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = int(self.config.early_stopping_patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if bool(self.config.restore_best) and best_state is not None:
            assert self.model is not None
            self.model.load_state_dict(best_state)
        outputs_full = self._collect_output_arrays_std(x_std, coords_um=coords_um)
        self.latent_diagnostics = _latent_diagnostics(
            outputs_full,
            x_std=x_std,
            context_std=context_std,
            coords_um=coords_um,
            selected_embedding=self.config.output_embedding,
        )
        if counts_target is not None and library_log is not None and self.model is not None:
            diag_rng = np.random.default_rng(seed + 9000)
            gene_index_diag = _sample_gene_chunk(int(counts_target.shape[1]), int(self.config.count_chunk_size), diag_rng)
            with torch.no_grad():
                intrinsic_t = torch.as_tensor(outputs_full["intrinsic"], dtype=torch.float32, device=self.device)
                gene_index_t = torch.as_tensor(gene_index_diag, dtype=torch.long, device=self.device)
                library_t = torch.as_tensor(library_log, dtype=torch.float32, device=self.device)
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
            self.latent_diagnostics["count_reconstruction_nb_loss"] = float(count_diag.detach().cpu())
            self.latent_diagnostics["count_reconstruction_gene_chunk_size"] = int(gene_index_diag.shape[0])
            self.latent_diagnostics["count_target_dim"] = int(counts_target.shape[1])
        self.latent_diagnostics["runtime_memory"] = _runtime_memory_snapshot(self.device)
        return self

    def _validate_transform_schema(
        self,
        *,
        input_obsm_key: str | None = None,
        coordinate_keys: tuple[str, str] | None = None,
        spatial_scale: float | None = None,
    ) -> None:
        expected_dim = self.feature_schema.get("input_dim")
        if expected_dim is not None and self.input_dim is not None and int(expected_dim) != int(self.input_dim):
            raise ValueError("Saved feature schema is inconsistent with the loaded encoder input dimension.")
        expected_key = self.feature_schema.get("input_obsm_key")
        if expected_key is not None and input_obsm_key is not None and str(expected_key) != str(input_obsm_key):
            raise ValueError(
                f"Input obsm key mismatch: encoder expects '{expected_key}', got '{input_obsm_key}'."
            )
        expected_coord_keys = self.feature_schema.get("coordinate_keys")
        if expected_coord_keys is not None and coordinate_keys is not None and list(coordinate_keys) != list(expected_coord_keys):
            raise ValueError(
                f"Coordinate key mismatch: encoder expects {expected_coord_keys}, got {list(coordinate_keys)}."
            )
        expected_scale = self.feature_schema.get("spatial_scale")
        if expected_scale is not None and spatial_scale is not None and not np.isclose(float(expected_scale), float(spatial_scale), atol=1e-8):
            raise ValueError(
                f"Spatial scale mismatch: encoder expects {float(expected_scale)}, got {float(spatial_scale)}."
            )

    def transform(self, features: np.ndarray, coords_um: np.ndarray | None = None, batch_size: int | None = None) -> np.ndarray:
        self._check_fitted()
        self._validate_transform_schema()
        if self.config.output_embedding is None:
            raise ValueError("SpatialOTFeatureEncoder.transform requires config.output_embedding to be set explicitly.")
        assert self.feature_mean is not None and self.feature_std is not None and self.model is not None and self.input_dim is not None
        x_std = _apply_standardization(features, self.feature_mean, self.feature_std)
        if x_std.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {x_std.shape[1]}.")
        if batch_size is not None and self.config.method != "graph_autoencoder":
            original_batch_size = self.config.batch_size
            self.config.batch_size = int(batch_size)
            try:
                outputs = self._collect_output_arrays_std(x_std, coords_um=coords_um)
            finally:
                self.config.batch_size = original_batch_size
        else:
            outputs = self._collect_output_arrays_std(x_std, coords_um=coords_um)
        return np.asarray(outputs[self.config.output_embedding], dtype=np.float32)

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
        assert self.model is not None and self.feature_mean is not None and self.feature_std is not None and self.input_dim is not None
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
        state_dict, metadata, feature_mean, feature_std = load_encoder_bundle(path, map_location=resolved_map_location)
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


def save_deep_feature_history(history: list[dict[str, float]], path: str | Path) -> None:
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
