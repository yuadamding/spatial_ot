from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class FeatureSpace:
    values: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StateCodebook:
    posteriors: np.ndarray
    centers: np.ndarray
    labels: np.ndarray
    entropy: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    temperature: float
    metadata: dict[str, object] = field(default_factory=dict)


def standardize_features(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("feature matrix must be 2D.")
    if x.shape[0] == 0:
        raise ValueError("feature matrix must contain at least one row.")
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    finite = scale[np.isfinite(scale) & (scale > 1e-8)]
    floor = float(np.median(finite) * 1e-3) if finite.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > max(floor, 1e-8)), scale, 1.0)
    z = ((x.astype(np.float64, copy=False) - mean[None, :]) / scale[None, :]).astype(
        np.float32
    )
    return z, mean.astype(np.float32), scale.astype(np.float32)


def prepare_feature_space(
    values: np.ndarray,
    *,
    n_components: int = 128,
    random_state: int = 1337,
) -> FeatureSpace:
    """Whiten and optionally reduce the molecular feature view used by SHE-lite."""

    z, mean, scale = standardize_features(values)
    n_cells, n_features = int(z.shape[0]), int(z.shape[1])
    requested = max(int(n_components), 0)
    metadata: dict[str, object] = {
        "preprocessing": "mean_std_whiten",
        "input_dim": n_features,
        "output_dim": n_features,
        "requested_components": int(requested),
        "reduction": "none",
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }
    if requested > 0 and n_features > requested and n_cells > requested + 1:
        svd = TruncatedSVD(
            n_components=int(requested),
            n_iter=7,
            random_state=int(random_state),
        )
        reduced = svd.fit_transform(z).astype(np.float32, copy=False)
        reduced, reduced_mean, reduced_scale = standardize_features(reduced)
        metadata.update(
            {
                "reduction": "truncated_svd_after_whitening",
                "output_dim": int(reduced.shape[1]),
                "explained_variance_ratio_sum": float(
                    np.sum(np.asarray(svd.explained_variance_ratio_, dtype=np.float64))
                ),
                "reduced_feature_mean": reduced_mean.tolist(),
                "reduced_feature_scale": reduced_scale.tolist(),
            }
        )
        z = reduced
    return FeatureSpace(values=z.astype(np.float32), metadata=metadata)


def _pairwise_sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    return np.maximum(
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1, keepdims=True).T
        - 2.0 * (a @ b.T),
        0.0,
    ).astype(np.float32)


def _soft_assign(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    temperature: float,
    batch_size: int = 50000,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    c = np.asarray(centers, dtype=np.float32)
    out = np.zeros((int(x.shape[0]), int(c.shape[0])), dtype=np.float32)
    temp = max(float(temperature), 1e-8)
    for start in range(0, int(x.shape[0]), max(int(batch_size), 1)):
        stop = min(start + max(int(batch_size), 1), int(x.shape[0]))
        d2 = _pairwise_sqdist(x[start:stop], c)
        logits = -d2 / temp
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits).astype(np.float32)
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        out[start:stop] = probs
    return out


def fit_state_codebook(
    values: np.ndarray,
    *,
    n_codewords: int = 64,
    sample_size: int = 50000,
    temperature: float | None = None,
    random_state: int = 1337,
) -> StateCodebook:
    """Fit a global soft cell-state codebook over calibrated cell features."""

    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("values must be a 2D matrix.")
    if x.shape[0] == 0:
        raise ValueError("values must contain at least one cell.")
    z, mean, scale = standardize_features(x)
    n_cells = int(z.shape[0])
    n_codes = min(max(int(n_codewords), 2), n_cells)
    fit_n = min(max(int(sample_size), n_codes), n_cells)
    rng = np.random.default_rng(int(random_state))
    fit_idx = np.sort(rng.choice(n_cells, size=fit_n, replace=False))
    model = MiniBatchKMeans(
        n_clusters=n_codes,
        n_init=3,
        batch_size=min(max(n_codes * 256, 2048), max(fit_n, 2048)),
        max_iter=60,
        max_no_improvement=8,
        reassignment_ratio=0.0,
        random_state=int(random_state),
    )
    model.fit(z[fit_idx])
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    nearest_d2 = np.min(_pairwise_sqdist(z[fit_idx], centers), axis=1)
    positive = nearest_d2[np.isfinite(nearest_d2) & (nearest_d2 > 1e-8)]
    resolved_temperature = (
        float(temperature)
        if temperature is not None and float(temperature) > 0.0
        else float(np.median(positive))
        if positive.size
        else 1.0
    )
    posteriors = _soft_assign(z, centers, temperature=resolved_temperature)
    labels = np.asarray(np.argmax(posteriors, axis=1), dtype=np.int32)
    entropy = -np.sum(
        posteriors.astype(np.float64) * np.log(np.maximum(posteriors, 1e-12)),
        axis=1,
    ) / max(float(np.log(max(n_codes, 2))), 1e-8)
    metadata = {
        "method": "minibatch_kmeans_soft_assignment",
        "n_cells": n_cells,
        "n_codewords_requested": int(n_codewords),
        "n_codewords": int(n_codes),
        "sample_size_requested": int(sample_size),
        "sample_size_used": int(fit_n),
        "temperature": float(resolved_temperature),
        "temperature_source": "user"
        if temperature is not None and float(temperature) > 0.0
        else "median_nearest_center_sqdist",
        "mean_state_entropy": float(np.mean(entropy)) if entropy.size else 0.0,
    }
    return StateCodebook(
        posteriors=posteriors.astype(np.float32),
        centers=centers.astype(np.float32),
        labels=labels,
        entropy=entropy.astype(np.float32),
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        temperature=float(resolved_temperature),
        metadata=metadata,
    )


__all__ = [
    "FeatureSpace",
    "StateCodebook",
    "fit_state_codebook",
    "prepare_feature_space",
    "standardize_features",
]
