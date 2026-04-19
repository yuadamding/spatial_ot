from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import copy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DeepFeatureConfig


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


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off-diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _variance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(1.0 - std).mean()


def _decorrelation_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0], 1)
    return _off_diagonal(cov).pow(2).mean()


def _build_context_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    *,
    neighbor_k: int,
    radius_um: float | None,
    device: torch.device | None = None,
) -> np.ndarray:
    coords = np.asarray(coords_um, dtype=np.float32)
    feats = np.asarray(features_std, dtype=np.float32)
    if coords.shape[0] == 0:
        return feats.copy()

    if radius_um is not None:
        nn_model = NearestNeighbors(radius=float(radius_um), metric="euclidean")
        nn_model.fit(coords)
        neighborhoods = nn_model.radius_neighbors(coords, return_distance=False)
    else:
        n_neighbors = min(max(int(neighbor_k) + 1, 2), coords.shape[0])
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn_model.fit(coords)
        neighborhoods = nn_model.kneighbors(coords, return_distance=False)

    src_index: list[int] = []
    dst_index: list[int] = []
    has_neighbors = np.zeros(coords.shape[0], dtype=bool)
    for idx, neighbors in enumerate(neighborhoods):
        neigh = np.asarray(neighbors, dtype=np.int64)
        neigh = neigh[neigh != idx]
        if neigh.size == 0:
            continue
        has_neighbors[idx] = True
        src_index.extend([idx] * int(neigh.size))
        dst_index.extend(neigh.tolist())
    if not src_index:
        return feats.copy()

    target_device = device or torch.device("cpu")
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=target_device)
    src_t = torch.as_tensor(src_index, dtype=torch.long, device=target_device)
    dst_t = torch.as_tensor(dst_index, dtype=torch.long, device=target_device)
    context_t = torch.zeros_like(feats_t)
    deg_t = torch.zeros((feats_t.shape[0], 1), dtype=feats_t.dtype, device=target_device)
    context_t.index_add_(0, src_t, feats_t[dst_t])
    deg_t.index_add_(0, src_t, torch.ones((dst_t.numel(), 1), dtype=feats_t.dtype, device=target_device))
    context_t = context_t / deg_t.clamp_min(1.0)
    no_neighbor_mask = torch.as_tensor(~has_neighbors, dtype=torch.bool, device=target_device)
    context_t[no_neighbor_mask] = feats_t[no_neighbor_mask]
    return context_t.detach().cpu().numpy().astype(np.float32, copy=False)


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


class _DeepFeatureNet(nn.Module):
    def __init__(self, input_dim: int, config: DeepFeatureConfig) -> None:
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
        context_layers.append(nn.Linear(ctx_in, int(input_dim)))
        self.context_head = nn.Sequential(*context_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_body(x)
        return self.encoder_head(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decoder(z)
        ctx = self.context_head(z)
        return z, recon, ctx


@dataclass
class DeepFeatureResult:
    embedding: np.ndarray
    history: list[dict[str, float]]
    config: dict
    feature_mean: np.ndarray
    feature_std: np.ndarray
    model_path: str | None = None


class SpatialOTFeatureEncoder:
    def __init__(self, config: DeepFeatureConfig) -> None:
        self.config = config
        self.model: _DeepFeatureNet | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.input_dim: int | None = None
        self.history: list[dict[str, float]] = []
        self.device = _resolve_device(config.device)

    @property
    def uses_coordinate_input(self) -> bool:
        return False

    def _check_fitted(self) -> None:
        if self.model is None or self.feature_mean is None or self.feature_std is None or self.input_dim is None:
            raise RuntimeError("SpatialOTFeatureEncoder is not fitted.")

    def fit(
        self,
        features: np.ndarray,
        coords_um: np.ndarray,
        *,
        batch: np.ndarray | None = None,
        seed: int = 1337,
    ) -> "SpatialOTFeatureEncoder":
        if self.config.method == "none":
            raise ValueError("SpatialOTFeatureEncoder.fit requires an active deep feature method, not 'none'.")
        if self.config.count_layer is not None:
            raise NotImplementedError("deep.count_layer is configured but count reconstruction is not implemented yet.")
        x = np.asarray(features, dtype=np.float32)
        context_targets = _build_context_targets(
            coords_um=coords_um,
            features_std=x,
            neighbor_k=self.config.neighbor_k,
            radius_um=self.config.radius_um,
            device=self.device,
        )
        val_mask = _split_validation(coords_um=coords_um, batch=batch, config=self.config, seed=seed)
        train_mask = ~val_mask
        if not np.any(train_mask):
            train_mask[:] = True
            val_mask[:] = False

        _, mean, std = _standardize_features(x[train_mask])
        x_std = _apply_standardization(x, mean, std)
        context_std = _apply_standardization(context_targets, mean, std)

        self.feature_mean = mean
        self.feature_std = std
        self.input_dim = int(x_std.shape[1])
        self.model = _DeepFeatureNet(input_dim=self.input_dim, config=self.config).to(self.device)

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

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.history = []
        torch.manual_seed(int(seed))
        best_state = None
        best_val = float("inf")
        patience_left = int(self.config.early_stopping_patience)
        for epoch in range(int(self.config.epochs)):
            self.model.train()
            loss_accum = 0.0
            n_train = 0
            for batch_x, batch_ctx in train_loader:
                batch_x = batch_x.to(self.device)
                batch_ctx = batch_ctx.to(self.device)
                z, recon, pred_ctx = self.model(batch_x)
                loss_recon = torch.mean((recon - batch_x) ** 2)
                loss_ctx = torch.mean((pred_ctx - batch_ctx) ** 2)
                loss_var = _variance_loss(z)
                loss_decorr = _decorrelation_loss(z)
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
                        z_val, recon_val, pred_ctx_val = self.model(x_batch)
                        loss_recon = torch.mean((recon_val - x_batch) ** 2)
                        loss_ctx = torch.mean((pred_ctx_val - c_batch) ** 2)
                        loss_var = _variance_loss(z_val)
                        loss_decorr = _decorrelation_loss(z_val)
                        val_loss = (
                            float(self.config.reconstruction_weight) * loss_recon
                            + float(self.config.context_weight) * loss_ctx
                            + float(self.config.variance_weight) * loss_var
                            + float(self.config.decorrelation_weight) * loss_decorr
                        )
                        val_loss_accum += float(val_loss.detach().cpu()) * int(x_batch.shape[0])
                        n_val += int(x_batch.shape[0])
                current_val = float(val_loss_accum / max(n_val, 1))
                epoch_row["val_loss"] = current_val
                if current_val < best_val - float(self.config.min_delta):
                    best_val = current_val
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = int(self.config.early_stopping_patience)
                else:
                    patience_left -= 1
            self.history.append(epoch_row)
            if x_val is None or c_val is None or x_val.shape[0] == 0:
                if best_state is None:
                    best_state = copy.deepcopy(self.model.state_dict())
                best_val = min(best_val, epoch_row["train_loss"])
            elif patience_left <= 0:
                break
        if bool(self.config.restore_best) and best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def transform(self, features: np.ndarray, coords_um: np.ndarray | None = None, batch_size: int | None = None) -> np.ndarray:
        del coords_um
        self._check_fitted()
        assert self.feature_mean is not None and self.feature_std is not None and self.model is not None and self.input_dim is not None
        x_std = _apply_standardization(features, self.feature_mean, self.feature_std)
        if x_std.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {x_std.shape[1]}.")
        batch_size = int(batch_size or self.config.batch_size)
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch_np in _iter_batches(x_std, batch_size):
                x_tensor = torch.from_numpy(x_batch_np).to(self.device)
                z = self.model.encode(x_tensor).detach().cpu().numpy().astype(np.float32)
                outputs.append(z)
        return np.vstack(outputs) if outputs else np.zeros((0, int(self.config.latent_dim)), dtype=np.float32)

    def fit_transform(
        self,
        features: np.ndarray,
        coords_um: np.ndarray,
        *,
        batch: np.ndarray | None = None,
        seed: int = 1337,
    ) -> DeepFeatureResult:
        self.fit(features=features, coords_um=coords_um, batch=batch, seed=seed)
        embedding = self.transform(features, coords_um=coords_um)
        return DeepFeatureResult(
            embedding=embedding,
            history=list(self.history),
            config=asdict(self.config),
            feature_mean=np.asarray(self.feature_mean, dtype=np.float32),
            feature_std=np.asarray(self.feature_std, dtype=np.float32),
            model_path=None,
        )

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        assert self.model is not None and self.feature_mean is not None and self.feature_std is not None and self.input_dim is not None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), destination)
        destination.with_suffix(destination.suffix + ".meta.json").write_text(
            json.dumps(
                {
                    "config": asdict(self.config),
                    "input_dim": int(self.input_dim),
                    "history": self.history,
                },
                indent=2,
            )
        )
        np.savez_compressed(
            destination.with_suffix(destination.suffix + ".scaler.npz"),
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | None = None) -> "SpatialOTFeatureEncoder":
        path = Path(path)
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        scaler = np.load(path.with_suffix(path.suffix + ".scaler.npz"))
        config = DeepFeatureConfig(**meta["config"])
        encoder = cls(config=config)
        encoder.feature_mean = np.asarray(scaler["feature_mean"], dtype=np.float32)
        encoder.feature_std = np.asarray(scaler["feature_std"], dtype=np.float32)
        encoder.input_dim = int(meta["input_dim"])
        encoder.model = _DeepFeatureNet(input_dim=encoder.input_dim, config=config).to(encoder.device)
        state_dict = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        encoder.model.load_state_dict(state_dict)
        encoder.history = list(meta.get("history", []))
        return encoder


def fit_deep_features(
    features: np.ndarray,
    coords_um: np.ndarray,
    *,
    config: DeepFeatureConfig,
    batch: np.ndarray | None = None,
    seed: int = 1337,
    save_path: str | Path | None = None,
) -> DeepFeatureResult:
    encoder = SpatialOTFeatureEncoder(config=config)
    result = encoder.fit_transform(features=features, coords_um=coords_um, batch=batch, seed=seed)
    if save_path is not None:
        encoder.save(save_path)
        result.model_path = str(Path(save_path))
    return result


def save_deep_feature_history(history: list[dict[str, float]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(history, indent=2))
        return
    import pandas as pd

    pd.DataFrame(history).to_csv(destination, index=False)
