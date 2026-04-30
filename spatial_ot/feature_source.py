from __future__ import annotations

import os
from pathlib import Path
import tempfile
import warnings

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

FULL_GENE_FEATURE_KEY = "X"
PREPARED_FEATURES_UNS_KEY = "spatial_ot_prepared_features"
PREPARED_X_FEATURE_KEY_PREFIX = "X_spatial_ot_x_svd"
_VISUALIZATION_LIKE_FEATURE_TOKENS = (
    "tsne",
    "t_sne",
    "t-sne",
    "phate",
    "draw_graph",
    "forceatlas",
    "force_atlas",
)


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


def _x_feature_request() -> dict[str, int | float]:
    target_sum = _env_float("SPATIAL_OT_X_TARGET_SUM", 10000.0)
    requested_components = max(_env_int("SPATIAL_OT_X_SVD_COMPONENTS", 256), 2)
    randomized_svd_iters = max(_env_int("SPATIAL_OT_X_SVD_N_ITER", 7), 1)
    randomized_svd_seed = _env_int("SPATIAL_OT_X_SVD_RANDOM_STATE", 1337)
    return {
        "target_sum": float(target_sum),
        "svd_components_requested": int(requested_components),
        "svd_n_iter": int(randomized_svd_iters),
        "svd_random_state": int(randomized_svd_seed),
    }


def default_precomputed_x_feature_key(
    *, requested_components: int | None = None
) -> str:
    components = (
        int(requested_components)
        if requested_components is not None
        else int(_x_feature_request()["svd_components_requested"])
    )
    return f"{PREPARED_X_FEATURE_KEY_PREFIX}_{components}"


def _feature_space_kind(
    *, feature_key: str, input_mode: str, preprocessing: str, warning: str | None
) -> str:
    if warning == "umap_exploratory":
        return "umap_embedding"
    if warning == "visualization_embedding_like":
        return "visualization_like_embedding"
    if input_mode == "X":
        return (
            "full_gene_runtime_svd"
            if "truncated_svd" in preprocessing
            else "full_gene_dense"
        )
    return "obsm"


def _resolve_x_features(adata: ad.AnnData) -> tuple[np.ndarray, dict]:
    if adata.X is None or int(adata.n_vars) <= 0:
        raise ValueError(
            "Feature key 'X' was requested, but the input H5AD does not contain a usable gene matrix."
        )

    request = _x_feature_request()
    target_sum = float(request["target_sum"])
    requested_components = int(request["svd_components_requested"])
    randomized_svd_iters = int(request["svd_n_iter"])
    randomized_svd_seed = int(request["svd_random_state"])

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
        explained = float(
            np.sum(np.asarray(svd.explained_variance_ratio_, dtype=np.float64))
        )
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
        "svd_components_used": int(svd_components_used)
        if svd_components_used is not None
        else None,
        "svd_random_state": int(randomized_svd_seed),
        "svd_n_iter": int(randomized_svd_iters),
        "svd_explained_variance_ratio_sum": explained,
        "feature_embedding_warning": None,
        "feature_space_kind": _feature_space_kind(
            feature_key=FULL_GENE_FEATURE_KEY,
            input_mode="X",
            preprocessing=preprocessing,
            warning=None,
        ),
    }


def prepare_h5ad_feature_cache(
    input_h5ad: str | Path,
    *,
    output_h5ad: str | Path | None = None,
    feature_obsm_key: str = FULL_GENE_FEATURE_KEY,
    output_obsm_key: str | None = None,
    overwrite: bool = False,
) -> dict:
    if feature_obsm_key != FULL_GENE_FEATURE_KEY:
        raise NotImplementedError(
            "prepare_h5ad_feature_cache currently supports only feature_obsm_key='X'."
        )

    input_path = Path(input_h5ad)
    output_path = Path(output_h5ad) if output_h5ad is not None else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = _x_feature_request()
    prepared_key = str(
        output_obsm_key
        or default_precomputed_x_feature_key(
            requested_components=int(request["svd_components_requested"])
        )
    )

    adata = ad.read_h5ad(input_path)
    prepared_uns = dict(adata.uns.get(PREPARED_FEATURES_UNS_KEY, {}))
    existing_metadata = (
        dict(prepared_uns.get(prepared_key, {})) if prepared_key in prepared_uns else {}
    )
    reusable = (
        not bool(overwrite)
        and prepared_key in adata.obsm
        and int(existing_metadata.get("svd_components_requested", -1))
        == int(request["svd_components_requested"])
        and int(existing_metadata.get("svd_random_state", -1))
        == int(request["svd_random_state"])
        and int(existing_metadata.get("svd_n_iter", -1)) == int(request["svd_n_iter"])
        and float(existing_metadata.get("target_sum", float("nan")))
        == float(request["target_sum"])
        and str(existing_metadata.get("input_feature_key", "")) == FULL_GENE_FEATURE_KEY
    )

    if reusable:
        features = np.asarray(adata.obsm[prepared_key], dtype=np.float32)
        feature_source = {
            "feature_key": prepared_key,
            "requested_feature_key": FULL_GENE_FEATURE_KEY,
            "input_mode": "obsm",
            "preprocessing": str(
                existing_metadata.get(
                    "preprocessing", "library_size_normalize_log1p_truncated_svd"
                )
            ),
            "source_feature_dim": int(adata.n_vars),
            "feature_dim": int(features.shape[1]),
            "target_sum": float(
                existing_metadata.get("target_sum", request["target_sum"])
            ),
            "svd_components_requested": int(
                existing_metadata.get(
                    "svd_components_requested", request["svd_components_requested"]
                )
            ),
            "svd_components_used": int(
                existing_metadata.get("svd_components_used", features.shape[1])
            ),
            "svd_random_state": int(
                existing_metadata.get("svd_random_state", request["svd_random_state"])
            ),
            "svd_n_iter": int(
                existing_metadata.get("svd_n_iter", request["svd_n_iter"])
            ),
            "svd_explained_variance_ratio_sum": existing_metadata.get(
                "svd_explained_variance_ratio_sum"
            ),
            "feature_embedding_warning": None,
            "feature_space_kind": "prepared_full_gene_svd",
        }
        wrote_output = output_path != input_path
        if wrote_output:
            adata.write_h5ad(output_path, compression="gzip")
        return {
            "input_h5ad": str(input_path),
            "output_h5ad": str(output_path),
            "feature_obsm_key_requested": FULL_GENE_FEATURE_KEY,
            "prepared_feature_obsm_key": prepared_key,
            "reused_existing": True,
            "feature_source": feature_source,
        }

    features, feature_source = _resolve_x_features(adata)
    adata.obsm[prepared_key] = np.asarray(features, dtype=np.float32)
    prepared_uns[prepared_key] = {
        "input_feature_key": FULL_GENE_FEATURE_KEY,
        "output_feature_key": prepared_key,
        "preprocessing": str(feature_source["preprocessing"]),
        "source_feature_dim": int(feature_source["source_feature_dim"]),
        "feature_dim": int(feature_source["feature_dim"]),
        "target_sum": float(feature_source["target_sum"]),
        "svd_components_requested": int(feature_source["svd_components_requested"]),
        "svd_components_used": int(feature_source["svd_components_used"])
        if feature_source["svd_components_used"] is not None
        else None,
        "svd_random_state": int(feature_source["svd_random_state"]),
        "svd_n_iter": int(feature_source["svd_n_iter"]),
        "svd_explained_variance_ratio_sum": feature_source[
            "svd_explained_variance_ratio_sum"
        ],
    }
    adata.uns[PREPARED_FEATURES_UNS_KEY] = prepared_uns
    if output_path == input_path:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{output_path.name}.",
            suffix=".tmp.h5ad",
            dir=str(output_path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            adata.write_h5ad(tmp_path, compression="gzip")
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        adata.write_h5ad(output_path, compression="gzip")

    return {
        "input_h5ad": str(input_path),
        "output_h5ad": str(output_path),
        "feature_obsm_key_requested": FULL_GENE_FEATURE_KEY,
        "prepared_feature_obsm_key": prepared_key,
        "reused_existing": False,
        "feature_source": {
            **feature_source,
            "feature_key": prepared_key,
        },
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
    if feature_embedding_warning is None and any(
        token in feature_obsm_key.lower()
        for token in _VISUALIZATION_LIKE_FEATURE_TOKENS
    ):
        warnings.warn(
            "Using a visualization-like embedding as the OT feature space. "
            "These coordinates are often optimized for plotting rather than metric fidelity; "
            "prefer full-gene, PCA, or standardized marker/program features for validated runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        feature_embedding_warning = "visualization_embedding_like"
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
        "feature_space_kind": _feature_space_kind(
            feature_key=str(feature_obsm_key),
            input_mode="obsm",
            preprocessing="none",
            warning=feature_embedding_warning,
        ),
    }


__all__ = [
    "FULL_GENE_FEATURE_KEY",
    "PREPARED_FEATURES_UNS_KEY",
    "PREPARED_X_FEATURE_KEY_PREFIX",
    "default_precomputed_x_feature_key",
    "prepare_h5ad_feature_cache",
    "resolve_h5ad_features",
]
