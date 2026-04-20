from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import DeepFeatureConfig
from .checkpoint import load_encoder_bundle, save_encoder_bundle
from .graph import MeanGraphLayer, build_context_distribution_targets, build_multiscale_graphs
from .losses import decorrelation_loss, edge_contrastive_loss, variance_loss


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _apply_standardization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def _iter_batches(array: np.ndarray, batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, array.shape[0], batch_size):
        yield array[start : start + batch_size]


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _split_validation(
    coords_um: np.ndarray,
    batch: np.ndarray | None,
    config: DeepFeatureConfig,
    seed: int,
) -> np.ndarray:
    n = coords_um.shape[0]
    mask = np.zeros(n, dtype=bool)
    if config.validation == "none" or n < 8:
        return mask
    rng = np.random.default_rng(seed)
    if config.validation == "sample_holdout" and batch is not None:
        batch = np.asarray(batch)
        unique = np.unique(batch)
        if unique.size >= 2:
            held_out = unique[int(rng.integers(unique.size))]
            return batch == held_out
    block_count = min(5, max(2, n // 16))
    km = KMeans(n_clusters=block_count, n_init=10, random_state=seed)
    blocks = km.fit_predict(np.asarray(coords_um, dtype=np.float32))
    held_out = int(rng.integers(block_count))
    mask = blocks == held_out
    if np.all(mask) or not np.any(mask):
        mask[np.arange(n) % max(block_count, 2) == 0] = True
    return mask


def _context_radii(config: DeepFeatureConfig) -> tuple[float | None, float | None]:
    short_radius = config.short_radius_um if config.short_radius_um is not None else config.radius_um
    mid_radius = config.mid_radius_um
    if mid_radius is None and short_radius is not None:
        mid_radius = float(short_radius) * 2.0
    return short_radius, mid_radius


def _build_context_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    *,
    config: DeepFeatureConfig,
    device: torch.device,
) -> np.ndarray:
    short_radius, mid_radius = _context_radii(config)
    return build_context_distribution_targets(
        coords_um=coords_um,
        features_std=features_std,
        neighbor_k=config.neighbor_k,
        base_radius_um=config.radius_um,
        short_radius_um=short_radius,
        mid_radius_um=mid_radius,
        max_neighbors=config.graph_max_neighbors,
        device=device,
    )


def _build_split_context_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    config: DeepFeatureConfig,
    device: torch.device,
) -> np.ndarray:
    if config.validation_context_mode == "transductive" or not np.any(val_mask):
        return _build_context_targets(coords_um, features_std, config=config, device=device)

    context = np.zeros((features_std.shape[0], features_std.shape[1] * 2), dtype=np.float32)
    context[train_mask] = _build_context_targets(
        coords_um[train_mask],
        features_std[train_mask],
        config=config,
        device=device,
    )
    context[val_mask] = _build_context_targets(
        coords_um[val_mask],
        features_std[val_mask],
        config=config,
        device=device,
    )
    return context


class _MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, config: DeepFeatureConfig) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(int(config.layers), 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.encoder_body = nn.Sequential(*layers)
        self.encoder_head = nn.Linear(in_dim, int(config.latent_dim))

        decoder_layers = []
        dec_in = int(config.latent_dim)
        for _ in range(max(int(config.layers), 1)):
            decoder_layers.append(nn.Linear(dec_in, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dec_in = hidden_dim
        decoder_layers.append(nn.Linear(dec_in, int(input_dim)))
        self.decoder = nn.Sequential(*decoder_layers)

        context_layers = []
        ctx_in = int(config.latent_dim)
        for _ in range(max(int(config.layers) - 1, 0)):
            context_layers.append(nn.Linear(ctx_in, hidden_dim))
            context_layers.append(nn.ReLU())
            ctx_in = hidden_dim
        context_layers.append(nn.Linear(ctx_in, int(context_dim)))
        self.context_head = nn.Sequential(*context_layers)

    def encode(self, x: torch.Tensor, *, output_embedding: str = "joint") -> torch.Tensor:
        del output_embedding
        h = self.encoder_body(x)
        return self.encoder_head(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        return {
            "intrinsic": z,
            "context": z,
            "joint": z,
            "recon": self.decoder(z),
            "context_pred": self.context_head(z),
        }


class _GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, config: DeepFeatureConfig) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        layers = []
        in_dim = int(input_dim)
        for _ in range(max(int(config.layers), 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.encoder_body = nn.Sequential(*layers)
        self.encoder_head = nn.Linear(in_dim, int(config.latent_dim))
        self.short_layers = nn.ModuleList(
            [MeanGraphLayer(int(config.latent_dim), hidden_dim) for _ in range(max(int(config.graph_layers), 1))]
        )
        self.mid_layers = nn.ModuleList(
            [MeanGraphLayer(int(config.latent_dim), hidden_dim) for _ in range(max(int(config.graph_layers), 1))]
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(config.latent_dim) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(config.latent_dim)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(config.latent_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(input_dim)),
        )
        self.context_head = nn.Sequential(
            nn.Linear(int(config.latent_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(context_dim)),
        )

    def _encode_all(
        self,
        x: torch.Tensor,
        edge_index_short: torch.Tensor,
        edge_index_mid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsic = self.encoder_head(self.encoder_body(x))
        short = intrinsic
        for layer in self.short_layers:
            short = short + layer(short, edge_index_short)
        context = short
        for layer in self.mid_layers:
            context = context + layer(context, edge_index_mid)
        joint = self.fusion(torch.cat([intrinsic, short, context], dim=-1))
        return intrinsic, context, joint

    def encode(
        self,
        x: torch.Tensor,
        *,
        edge_index_short: torch.Tensor,
        edge_index_mid: torch.Tensor,
        output_embedding: str = "joint",
    ) -> torch.Tensor:
        intrinsic, context, joint = self._encode_all(x, edge_index_short=edge_index_short, edge_index_mid=edge_index_mid)
        if output_embedding == "intrinsic":
            return intrinsic
        if output_embedding == "context":
            return context
        return joint

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_index_short: torch.Tensor,
        edge_index_mid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intrinsic, context, joint = self._encode_all(x, edge_index_short=edge_index_short, edge_index_mid=edge_index_mid)
        return {
            "intrinsic": intrinsic,
            "context": context,
            "joint": joint,
            "recon": self.decoder(joint),
            "context_pred": self.context_head(joint),
        }


def _make_model(input_dim: int, context_dim: int, config: DeepFeatureConfig) -> nn.Module:
    if config.method == "graph_autoencoder":
        return _GraphAutoencoder(input_dim=input_dim, context_dim=context_dim, config=config)
    return _MLPAutoencoder(input_dim=input_dim, context_dim=context_dim, config=config)


def _tensor_graphs(
    coords_um: np.ndarray,
    *,
    config: DeepFeatureConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    short_radius, mid_radius = _context_radii(config)
    short_graph, mid_graph = build_multiscale_graphs(
        coords_um,
        neighbor_k=config.neighbor_k,
        base_radius_um=config.radius_um,
        short_radius_um=short_radius,
        mid_radius_um=mid_radius,
        max_neighbors=config.graph_max_neighbors,
    )
    return (
        torch.as_tensor(short_graph, dtype=torch.long, device=device),
        torch.as_tensor(mid_graph, dtype=torch.long, device=device),
    )


def _graph_summary(edge_index: torch.Tensor, n_nodes: int) -> dict[str, float | int]:
    if n_nodes <= 0:
        return {
            "edges": 0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "isolated_fraction": 0.0,
        }
    if edge_index.numel() == 0:
        return {
            "edges": 0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "isolated_fraction": 1.0,
        }
    dst = edge_index[1]
    deg = torch.bincount(dst, minlength=n_nodes)
    return {
        "edges": int(dst.numel()),
        "mean_degree": float(deg.float().mean().detach().cpu()),
        "max_degree": int(deg.max().detach().cpu()),
        "isolated_fraction": float((deg == 0).float().mean().detach().cpu()),
    }


@dataclass
class DeepFeatureResult:
    embedding: np.ndarray
    history: list[dict[str, float]]
    config: dict
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_schema: dict
    validation_report: dict
    model_path: str | None = None


class SpatialOTFeatureEncoder:
    def __init__(self, config: DeepFeatureConfig) -> None:
        self.config = config
        self.model: nn.Module | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.input_dim: int | None = None
        self.context_dim: int | None = None
        self.history: list[dict[str, float]] = []
        self.feature_schema: dict = {}
        self.validation_report: dict = {}
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

    def fit(
        self,
        features: np.ndarray,
        coords_um: np.ndarray,
        *,
        batch: np.ndarray | None = None,
        seed: int = 1337,
        feature_schema_extra: dict | None = None,
    ) -> "SpatialOTFeatureEncoder":
        if self.config.method == "none":
            raise ValueError("SpatialOTFeatureEncoder.fit requires an active deep feature method, not 'none'.")
        if self.config.count_layer is not None:
            raise NotImplementedError("deep.count_layer is configured but count reconstruction is not implemented yet.")

        x = np.asarray(features, dtype=np.float32)
        coords_um = np.asarray(coords_um, dtype=np.float32)
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
            "method": self.config.method,
            "output_embedding": self.config.output_embedding,
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
        self.model = _make_model(input_dim=self.input_dim, context_dim=self.context_dim, config=self.config).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.history = []
        best_state = None
        best_val = float("inf")
        patience_left = int(self.config.early_stopping_patience)

        if self.config.method == "graph_autoencoder":
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
                loss_contrast = edge_contrastive_loss(outputs["joint"], short_train)
                loss_var = variance_loss(z)
                loss_decorr = decorrelation_loss(z)
                loss = (
                    float(self.config.reconstruction_weight) * loss_recon
                    + float(self.config.context_weight) * loss_ctx
                    + float(self.config.contrastive_weight) * loss_contrast
                    + float(self.config.variance_weight) * loss_var
                    + float(self.config.decorrelation_weight) * loss_decorr
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_row = {
                    "epoch": float(epoch + 1),
                    "train_loss": float(loss.detach().cpu()),
                }
                current_val = None
                if x_val is not None and ctx_val is not None and short_val is not None and mid_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        outputs_val = self.model(x_val, edge_index_short=short_val, edge_index_mid=mid_val)
                        z_val = outputs_val[self.config.output_embedding]
                        val_loss = (
                            float(self.config.reconstruction_weight) * torch.mean((outputs_val["recon"] - x_val) ** 2)
                            + float(self.config.context_weight) * torch.mean((outputs_val["context_pred"] - ctx_val) ** 2)
                            + float(self.config.contrastive_weight) * edge_contrastive_loss(outputs_val["joint"], short_val)
                            + float(self.config.variance_weight) * variance_loss(z_val)
                            + float(self.config.decorrelation_weight) * decorrelation_loss(z_val)
                        )
                        current_val = float(val_loss.detach().cpu())
                        epoch_row["val_loss"] = current_val
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
                for batch_x, batch_ctx in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_ctx = batch_ctx.to(self.device)
                    outputs = self.model(batch_x)
                    z = outputs[self.config.output_embedding]
                    loss_recon = torch.mean((outputs["recon"] - batch_x) ** 2)
                    loss_ctx = torch.mean((outputs["context_pred"] - batch_ctx) ** 2)
                    loss_var = variance_loss(z)
                    loss_decorr = decorrelation_loss(z)
                    loss = (
                        float(self.config.reconstruction_weight) * loss_recon
                        + float(self.config.context_weight) * loss_ctx
                        + float(self.config.variance_weight) * loss_var
                        + float(self.config.decorrelation_weight) * loss_decorr
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_accum += float(loss.detach().cpu()) * int(batch_x.shape[0])
                    n_train += int(batch_x.shape[0])

                epoch_row = {
                    "epoch": float(epoch + 1),
                    "train_loss": float(loss_accum / max(n_train, 1)),
                }
                current_val = None
                if x_val is not None and c_val is not None and x_val.shape[0] > 0:
                    self.model.eval()
                    val_loss_accum = 0.0
                    n_val = 0
                    with torch.no_grad():
                        for x_batch_np, c_batch_np in zip(_iter_batches(x_val, self.config.batch_size), _iter_batches(c_val, self.config.batch_size), strict=False):
                            x_batch = torch.from_numpy(x_batch_np).to(self.device)
                            c_batch = torch.from_numpy(c_batch_np).to(self.device)
                            outputs = self.model(x_batch)
                            z_val = outputs[self.config.output_embedding]
                            val_loss = (
                                float(self.config.reconstruction_weight) * torch.mean((outputs["recon"] - x_batch) ** 2)
                                + float(self.config.context_weight) * torch.mean((outputs["context_pred"] - c_batch) ** 2)
                                + float(self.config.variance_weight) * variance_loss(z_val)
                                + float(self.config.decorrelation_weight) * decorrelation_loss(z_val)
                            )
                            val_loss_accum += float(val_loss.detach().cpu()) * int(x_batch.shape[0])
                            n_val += int(x_batch.shape[0])
                    current_val = float(val_loss_accum / max(n_val, 1))
                    epoch_row["val_loss"] = current_val
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
        assert self.feature_mean is not None and self.feature_std is not None and self.model is not None and self.input_dim is not None
        x_std = _apply_standardization(features, self.feature_mean, self.feature_std)
        if x_std.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {x_std.shape[1]}.")
        self.model.eval()
        if self.config.method == "graph_autoencoder":
            if coords_um is None:
                raise ValueError("coords_um is required for graph_autoencoder transforms.")
            short_graph, mid_graph = _tensor_graphs(np.asarray(coords_um, dtype=np.float32), config=self.config, device=self.device)
            with torch.no_grad():
                x_tensor = torch.from_numpy(x_std).to(self.device)
                z = self.model.encode(
                    x_tensor,
                    edge_index_short=short_graph,
                    edge_index_mid=mid_graph,
                    output_embedding=self.config.output_embedding,
                )
            return z.detach().cpu().numpy().astype(np.float32)

        batch_size = int(batch_size or self.config.batch_size)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch_np in _iter_batches(x_std, batch_size):
                x_tensor = torch.from_numpy(x_batch_np).to(self.device)
                z = self.model.encode(x_tensor, output_embedding=self.config.output_embedding).detach().cpu().numpy().astype(np.float32)
                outputs.append(z)
        return np.vstack(outputs) if outputs else np.zeros((0, int(self.config.latent_dim)), dtype=np.float32)

    def fit_transform(
        self,
        features: np.ndarray,
        coords_um: np.ndarray,
        *,
        batch: np.ndarray | None = None,
        seed: int = 1337,
        feature_schema_extra: dict | None = None,
    ) -> DeepFeatureResult:
        self.fit(
            features=features,
            coords_um=coords_um,
            batch=batch,
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
                "history": self.history,
                "feature_schema": self.feature_schema,
                "validation_report": self.validation_report,
            },
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | None = None) -> "SpatialOTFeatureEncoder":
        state_dict, metadata, feature_mean, feature_std = load_encoder_bundle(path, map_location=map_location or "cpu")
        config = DeepFeatureConfig(**metadata["config"])
        encoder = cls(config=config)
        encoder.feature_mean = feature_mean
        encoder.feature_std = feature_std
        encoder.input_dim = int(metadata["input_dim"])
        encoder.context_dim = int(metadata.get("context_dim", encoder.input_dim * 2))
        encoder.model = _make_model(input_dim=encoder.input_dim, context_dim=encoder.context_dim, config=config).to(encoder.device)
        encoder.model.load_state_dict(state_dict)
        encoder.history = list(metadata.get("history", []))
        encoder.feature_schema = dict(metadata.get("feature_schema", {}))
        encoder.validation_report = dict(metadata.get("validation_report", {}))
        return encoder


def fit_deep_features(
    features: np.ndarray,
    coords_um: np.ndarray,
    *,
    config: DeepFeatureConfig,
    batch: np.ndarray | None = None,
    seed: int = 1337,
    save_path: str | Path | None = None,
    feature_schema_extra: dict | None = None,
) -> DeepFeatureResult:
    encoder = SpatialOTFeatureEncoder(config=config)
    result = encoder.fit_transform(
        features=features,
        coords_um=coords_um,
        batch=batch,
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
