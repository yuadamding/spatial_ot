from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD


@dataclass(frozen=True)
class ExpressionEmbedding:
    values: np.ndarray
    metadata: dict[str, object]


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z = ((x.astype(np.float64, copy=False) - mean[None, :]) / scale[None, :]).astype(
        np.float32
    )
    return z, mean.astype(np.float32), scale.astype(np.float32)


def fit_expression_embedding(
    features: np.ndarray,
    *,
    method: str = "pca",
    embedding_dim: int = 32,
    random_state: int = 1337,
) -> ExpressionEmbedding:
    """Fit a cohort-wide expression embedding without using spatial coordinates."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    requested = str(method or "pca").strip().lower()
    if requested in {"scvi", "totalvi"}:
        raise NotImplementedError(
            "scVI/totalVI embeddings are not bundled in this compact runtime. "
            "Fit them externally and pass the result with --embedding-method precomputed."
        )
    if requested in {"precomputed", "none", "identity"}:
        z, center, scale = _standardize(x)
        return ExpressionEmbedding(
            values=z,
            metadata={
                "method": "precomputed",
                "input_dim": int(x.shape[1]),
                "embedding_dim": int(z.shape[1]),
                "uses_spatial_coordinates": False,
                "standardization": "mean_std",
                "center": center,
                "scale": scale,
            },
        )

    x_std, feature_mean, feature_scale = _standardize(x)
    target = min(max(int(embedding_dim), 1), int(x_std.shape[1]), max(int(x_std.shape[0]) - 1, 1))
    if requested in {"pca"}:
        reducer = PCA(n_components=target, random_state=int(random_state))
        z_raw = reducer.fit_transform(x_std).astype(np.float32, copy=False)
        explained = float(np.sum(np.asarray(reducer.explained_variance_ratio_, dtype=np.float64)))
        reducer_name = "pca_after_feature_standardization"
    elif requested in {"svd", "truncated_svd"}:
        reducer = TruncatedSVD(n_components=target, n_iter=7, random_state=int(random_state))
        z_raw = reducer.fit_transform(x_std).astype(np.float32, copy=False)
        explained = float(np.sum(np.asarray(reducer.explained_variance_ratio_, dtype=np.float64)))
        reducer_name = "truncated_svd_after_feature_standardization"
    else:
        raise ValueError("embedding_method must be pca, svd, precomputed, or externally fitted scvi.")

    z, embedding_mean, embedding_scale = _standardize(z_raw)
    return ExpressionEmbedding(
        values=z,
        metadata={
            "method": str(requested),
            "reduction": reducer_name,
            "input_dim": int(x.shape[1]),
            "embedding_dim": int(z.shape[1]),
            "requested_dim": int(embedding_dim),
            "explained_variance_ratio_sum": explained,
            "uses_spatial_coordinates": False,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "embedding_mean": embedding_mean,
            "embedding_scale": embedding_scale,
        },
    )
