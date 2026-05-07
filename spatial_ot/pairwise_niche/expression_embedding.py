from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD


@dataclass(frozen=True)
class ExpressionEmbedding:
    values: np.ndarray
    metadata: dict[str, object]
    state: "ExpressionEmbeddingState"


@dataclass(frozen=True)
class ExpressionEmbeddingState:
    method: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    reducer_components: np.ndarray | None
    reducer_mean: np.ndarray | None
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("features must be a 2D matrix.")
        if x.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("features do not match fitted expression embedding state.")
        x_std = (
            (x.astype(np.float64, copy=False) - self.feature_mean[None, :])
            / self.feature_scale[None, :]
        ).astype(np.float32)
        if self.reducer_components is None:
            z_raw = x_std
        else:
            reducer_mean = (
                np.zeros(x_std.shape[1], dtype=np.float32)
                if self.reducer_mean is None
                else self.reducer_mean
            )
            z_raw = (x_std - reducer_mean[None, :]) @ self.reducer_components.T
        z = (
            (z_raw.astype(np.float64, copy=False) - self.embedding_mean[None, :])
            / self.embedding_scale[None, :]
        )
        return z.astype(np.float32)


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
    standardize_precomputed: bool = True,
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
        if bool(standardize_precomputed):
            z, center, scale = _standardize(x)
            standardization = "mean_std"
        else:
            z = x.astype(np.float32, copy=True)
            center = np.zeros(x.shape[1], dtype=np.float32)
            scale = np.ones(x.shape[1], dtype=np.float32)
            standardization = "none"
        embedding_mean = np.zeros(z.shape[1], dtype=np.float32)
        embedding_scale = np.ones(z.shape[1], dtype=np.float32)
        state = ExpressionEmbeddingState(
            method="precomputed",
            feature_mean=center,
            feature_scale=scale,
            reducer_components=None,
            reducer_mean=None,
            embedding_mean=embedding_mean,
            embedding_scale=embedding_scale,
        )
        return ExpressionEmbedding(
            values=z,
            metadata={
                "method": "precomputed",
                "input_dim": int(x.shape[1]),
                "embedding_dim": int(z.shape[1]),
                "uses_spatial_coordinates": False,
                "standardization": standardization,
                "center": center,
                "scale": scale,
            },
            state=state,
        )

    x_std, feature_mean, feature_scale = _standardize(x)
    target = min(max(int(embedding_dim), 1), int(x_std.shape[1]), max(int(x_std.shape[0]) - 1, 1))
    if requested in {"pca"}:
        reducer = PCA(n_components=target, random_state=int(random_state))
        z_raw = reducer.fit_transform(x_std).astype(np.float32, copy=False)
        explained = float(np.sum(np.asarray(reducer.explained_variance_ratio_, dtype=np.float64)))
        reducer_name = "pca_after_feature_standardization"
        reducer_components = np.asarray(reducer.components_, dtype=np.float32)
        reducer_mean = np.asarray(reducer.mean_, dtype=np.float32)
    elif requested in {"svd", "truncated_svd"}:
        reducer = TruncatedSVD(n_components=target, n_iter=7, random_state=int(random_state))
        z_raw = reducer.fit_transform(x_std).astype(np.float32, copy=False)
        explained = float(np.sum(np.asarray(reducer.explained_variance_ratio_, dtype=np.float64)))
        reducer_name = "truncated_svd_after_feature_standardization"
        reducer_components = np.asarray(reducer.components_, dtype=np.float32)
        reducer_mean = np.zeros(x_std.shape[1], dtype=np.float32)
    else:
        raise ValueError("embedding_method must be pca, svd, precomputed, or externally fitted scvi.")

    z, embedding_mean, embedding_scale = _standardize(z_raw)
    state = ExpressionEmbeddingState(
        method=str(requested),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        reducer_components=reducer_components,
        reducer_mean=reducer_mean,
        embedding_mean=embedding_mean,
        embedding_scale=embedding_scale,
    )
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
        state=state,
    )


def save_expression_embedding_state(
    state: ExpressionEmbeddingState,
    path,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "feature_mean": np.asarray(state.feature_mean, dtype=np.float32),
        "feature_scale": np.asarray(state.feature_scale, dtype=np.float32),
        "embedding_mean": np.asarray(state.embedding_mean, dtype=np.float32),
        "embedding_scale": np.asarray(state.embedding_scale, dtype=np.float32),
        "method": np.asarray(str(state.method)),
        "has_reducer": np.asarray(state.reducer_components is not None),
    }
    if state.reducer_components is not None:
        arrays["reducer_components"] = np.asarray(
            state.reducer_components,
            dtype=np.float32,
        )
    if state.reducer_mean is not None:
        arrays["reducer_mean"] = np.asarray(state.reducer_mean, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_expression_embedding_state(path) -> ExpressionEmbeddingState:
    arrays = np.load(path, allow_pickle=False)
    method = str(np.asarray(arrays["method"]).item())
    has_reducer = bool(np.asarray(arrays["has_reducer"]).item())
    reducer_components = (
        np.asarray(arrays["reducer_components"], dtype=np.float32)
        if has_reducer and "reducer_components" in arrays
        else None
    )
    reducer_mean = (
        np.asarray(arrays["reducer_mean"], dtype=np.float32)
        if "reducer_mean" in arrays
        else None
    )
    return ExpressionEmbeddingState(
        method=method,
        feature_mean=np.asarray(arrays["feature_mean"], dtype=np.float32),
        feature_scale=np.asarray(arrays["feature_scale"], dtype=np.float32),
        reducer_components=reducer_components,
        reducer_mean=reducer_mean,
        embedding_mean=np.asarray(arrays["embedding_mean"], dtype=np.float32),
        embedding_scale=np.asarray(arrays["embedding_scale"], dtype=np.float32),
    )


__all__ = [
    "ExpressionEmbedding",
    "ExpressionEmbeddingState",
    "fit_expression_embedding",
    "load_expression_embedding_state",
    "save_expression_embedding_state",
]
