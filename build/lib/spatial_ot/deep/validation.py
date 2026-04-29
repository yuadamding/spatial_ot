from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
import torch

from ..config import DeepFeatureConfig
from .graph import build_context_distribution_targets


def split_validation(
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


def context_radii(config: DeepFeatureConfig) -> tuple[float | None, float | None]:
    short_radius = config.short_radius_um if config.short_radius_um is not None else config.radius_um
    mid_radius = config.mid_radius_um
    if mid_radius is None and short_radius is not None:
        mid_radius = float(short_radius) * 2.0
    return short_radius, mid_radius


def build_context_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    *,
    config: DeepFeatureConfig,
    device: torch.device,
) -> np.ndarray:
    short_radius, mid_radius = context_radii(config)
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


def build_split_context_targets(
    coords_um: np.ndarray,
    features_std: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    config: DeepFeatureConfig,
    device: torch.device,
) -> np.ndarray:
    if config.validation_context_mode == "transductive" or not np.any(val_mask):
        return build_context_targets(coords_um, features_std, config=config, device=device)

    target_dim = int(features_std.shape[1]) * 4 + 2
    context = np.zeros((features_std.shape[0], target_dim), dtype=np.float32)
    context[train_mask] = build_context_targets(
        coords_um[train_mask],
        features_std[train_mask],
        config=config,
        device=device,
    )
    context[val_mask] = build_context_targets(
        coords_um[val_mask],
        features_std[val_mask],
        config=config,
        device=device,
    )
    return context
