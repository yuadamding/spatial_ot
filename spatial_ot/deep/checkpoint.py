from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def save_encoder_bundle(
    path: str | Path,
    *,
    state_dict: dict,
    metadata: dict,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, destination)
    destination.with_suffix(destination.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2)
    )
    np.savez_compressed(
        destination.with_suffix(destination.suffix + ".scaler.npz"),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
    )


def load_encoder_bundle(
    path: str | Path, *, map_location: str = "cpu"
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    source = Path(path)
    state_dict = torch.load(source, map_location=map_location, weights_only=True)
    metadata = json.loads(source.with_suffix(source.suffix + ".meta.json").read_text())
    scaler = np.load(source.with_suffix(source.suffix + ".scaler.npz"))
    feature_mean = np.asarray(scaler["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(scaler["feature_std"], dtype=np.float32)
    return state_dict, metadata, feature_mean, feature_std
