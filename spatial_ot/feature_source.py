from __future__ import annotations

import os
import warnings

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

FULL_GENE_FEATURE_KEY = "X"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _normalize_counts_log1p(matrix, *, target_sum: float):
    if sparse.issparse(matrix):
        x = matrix.tocsr(copy=True).astype(np.float32)
        if x.nnz > 0:
            row_sums = np.asarray(x.sum(axis=1)).ravel().astype(np.float32, copy=False)
            scale = np.ones_like(row_sums, dtype=np.float32)
            nonzero = row_sums > 0
            scale[nonzero] = float(target_sum) / row_sums[nonzero]
            x.data *= np.repeat(scale, np.diff(x.indptr))
            x.data = np.log1p(x.data).astype(np.float32, copy=False)
        return x

    x = np.asarray(matrix, dtype=np.float32)
    row_sums = x.sum(axis=1, keepdims=True).astype(np.float32, copy=False)
    scale = np.ones_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    scale[nonzero] = float(target_sum) / row_sums[nonzero]
    return np.log1p(x * scale).astype(np.float32, copy=False)


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _resolve_x_features(adata: ad.AnnData) -> tuple[np.ndarray, dict]:
    if adata.X is None or int(adata.n_vars) <= 0:
        raise ValueError("Feature key 'X' was requested, but the input H5AD does not contain a usable gene matrix.")

    target_sum = _env_float("SPATIAL_OT_X_TARGET_SUM", 10000.0)
    requested_components = max(_env_int("SPATIAL_OT_X_SVD_COMPONENTS", 256), 2)
    randomized_svd_iters = max(_env_int("SPATIAL_OT_X_SVD_N_ITER", 7), 1)
    randomized_svd_seed = _env_int("SPATIAL_OT_X_SVD_RANDOM_STATE", 1337)

    normalized = _normalize_counts_log1p(adata.X, target_sum=target_sum)
    max_components = min(int(adata.n_obs) - 1, int(adata.n_vars) - 1)
    explained = None
    preprocessing = "library_size_normalize_log1p"
    svd_components_used = None

    if max_components >= 2:
        svd_components_used = int(min(requested_components, max_components))
        svd = TruncatedSVD(
            n_components=svd_components_used,
            algorithm="randomized",
            n_iter=randomized_svd_iters,
            random_state=randomized_svd_seed,
        )
        features = svd.fit_transform(normalized).astype(np.float32, copy=False)
        explained = float(np.sum(np.asarray(svd.explained_variance_ratio_, dtype=np.float64)))
        preprocessing = "library_size_normalize_log1p_truncated_svd"
    else:
        features = _dense_float32(normalized)

    return features, {
        "feature_key": FULL_GENE_FEATURE_KEY,
        "requested_feature_key": FULL_GENE_FEATURE_KEY,
        "input_mode": "X",
        "preprocessing": preprocessing,
        "source_feature_dim": int(adata.n_vars),
        "feature_dim": int(features.shape[1]),
        "target_sum": float(target_sum),
        "svd_components_requested": int(requested_components),
        "svd_components_used": int(svd_components_used) if svd_components_used is not None else None,
        "svd_random_state": int(randomized_svd_seed),
        "svd_n_iter": int(randomized_svd_iters),
        "svd_explained_variance_ratio_sum": explained,
        "feature_embedding_warning": None,
    }


def resolve_h5ad_features(
    adata: ad.AnnData,
    *,
    feature_obsm_key: str,
    allow_umap_as_feature: bool = False,
) -> tuple[np.ndarray, dict]:
    if feature_obsm_key == FULL_GENE_FEATURE_KEY:
        return _resolve_x_features(adata)

    if feature_obsm_key not in adata.obsm:
        raise KeyError(
            f"Feature key '{feature_obsm_key}' was not found. "
            f"Use an existing obsm key or '{FULL_GENE_FEATURE_KEY}' for the full gene matrix."
        )

    feature_embedding_warning = None
    if "umap" in feature_obsm_key.lower():
        if not allow_umap_as_feature:
            raise ValueError(
                "Using UMAP coordinates as the OT feature space requires explicit opt-in. "
                "Pass allow_umap_as_feature=True or --allow-umap-as-feature for exploratory runs."
            )
        warnings.warn(
            "Using UMAP coordinates as the OT feature space. UMAP is not generally metric-preserving; "
            "prefer full-gene, PCA, or standardized marker features for validated runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        feature_embedding_warning = "umap_exploratory"

    features = np.asarray(adata.obsm[feature_obsm_key], dtype=np.float32)
    return features, {
        "feature_key": str(feature_obsm_key),
        "requested_feature_key": str(feature_obsm_key),
        "input_mode": "obsm",
        "preprocessing": "none",
        "source_feature_dim": int(features.shape[1]),
        "feature_dim": int(features.shape[1]),
        "target_sum": None,
        "svd_components_requested": None,
        "svd_components_used": None,
        "svd_random_state": None,
        "svd_n_iter": None,
        "svd_explained_variance_ratio_sum": None,
        "feature_embedding_warning": feature_embedding_warning,
    }


__all__ = [
    "FULL_GENE_FEATURE_KEY",
    "resolve_h5ad_features",
]
