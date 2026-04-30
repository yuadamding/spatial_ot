#!/usr/bin/env python3
"""Compute per-subregion internal heterogeneity diagnostics for a spatial_ot run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_EMBEDDING_KEY = "X_spatial_ot_deep_expression_autoencoder"
DEFAULT_SUBREGION_OBS = "mlot_subregion_int"


def _as_float_array(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _as_jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_as_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_jsonable(v) for v in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool | int | float | str):
        return value
    try:
        return str(value)
    except Exception:
        return None


def _entropy_from_counts(counts: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    totals = counts.groupby(level=0).sum()
    group_index = counts.index.get_level_values(0)
    probs = counts.to_numpy(dtype=np.float64) / totals.reindex(group_index).to_numpy(
        dtype=np.float64
    )
    entropy_values = -(probs * np.log(np.clip(probs, 1e-12, 1.0)))
    entropy = pd.Series(entropy_values, index=counts.index).groupby(level=0).sum()
    richness = counts.groupby(level=0).size().astype(float)
    denom = np.log(np.maximum(richness.to_numpy(dtype=np.float64), 2.0))
    normalized = pd.Series(
        np.where(richness.to_numpy(dtype=np.float64) <= 1.0, 0.0, entropy / denom),
        index=richness.index,
    )
    top_fraction = counts.groupby(level=0).max() / totals
    return entropy, normalized, top_fraction


def _categorical_diversity(
    obs: pd.DataFrame,
    group_codes: np.ndarray,
    subregion_ids: np.ndarray,
    column: str,
    prefix: str,
) -> pd.DataFrame:
    out = pd.DataFrame({"subregion_id": subregion_ids})
    if column not in obs.columns:
        for suffix in (
            "entropy",
            "norm_entropy",
            "top_fraction",
            "n_categories",
        ):
            out[f"{prefix}_{suffix}"] = np.nan
        return out
    values = obs[column].astype("string").fillna("__missing__").to_numpy()
    df = pd.DataFrame({"group_code": group_codes, "value": values})
    counts = df.groupby(["group_code", "value"], observed=True).size()
    entropy, normalized, top_fraction = _entropy_from_counts(counts)
    n_categories = counts.groupby(level=0).size()
    out[f"{prefix}_entropy"] = np.nan
    out[f"{prefix}_norm_entropy"] = np.nan
    out[f"{prefix}_top_fraction"] = np.nan
    out[f"{prefix}_n_categories"] = 0
    out.loc[entropy.index, f"{prefix}_entropy"] = entropy.to_numpy()
    out.loc[normalized.index, f"{prefix}_norm_entropy"] = normalized.to_numpy()
    out.loc[top_fraction.index, f"{prefix}_top_fraction"] = top_fraction.to_numpy()
    out.loc[n_categories.index, f"{prefix}_n_categories"] = n_categories.to_numpy()
    return out


def _numeric_group_stats(
    values: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    prefix: str,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(group_codes, kind="mergesort")
    codes = group_codes[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    present = codes[starts]
    counts = np.diff(np.r_[starts, len(codes)])
    sums = np.add.reduceat(sorted_values, starts)
    out = pd.DataFrame({"group_code": np.arange(n_groups)})
    out[f"{prefix}_mean"] = np.nan
    out[f"{prefix}_std"] = np.nan
    out[f"{prefix}_p50"] = np.nan
    out[f"{prefix}_p90"] = np.nan
    out[f"{prefix}_p95"] = np.nan
    out.loc[present, f"{prefix}_mean"] = sums / counts
    for code, start, count in zip(present, starts, counts, strict=True):
        block = sorted_values[start : start + count]
        out.loc[code, f"{prefix}_std"] = float(np.std(block))
        out.loc[code, f"{prefix}_p50"] = float(np.quantile(block, 0.50))
        out.loc[code, f"{prefix}_p90"] = float(np.quantile(block, 0.90))
        out.loc[code, f"{prefix}_p95"] = float(np.quantile(block, 0.95))
    return out.drop(columns=["group_code"])


def _embedding_group_stats(
    values: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    prefix: str,
) -> pd.DataFrame:
    values = _as_float_array(values)
    order = np.argsort(group_codes, kind="mergesort")
    codes = group_codes[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    present = codes[starts]
    counts = np.diff(np.r_[starts, len(codes)])
    sorted_values64 = sorted_values.astype(np.float64, copy=False)
    row_sq = np.einsum("ij,ij->i", sorted_values64, sorted_values64)
    sums = np.add.reduceat(sorted_values64, starts, axis=0)
    sum_sq = np.add.reduceat(row_sq, starts)
    means = sums / counts[:, None]
    centroid_sq = np.einsum("ij,ij->i", means, means)
    variance_trace = np.maximum(sum_sq / counts - centroid_sq, 0.0)

    out = pd.DataFrame({"group_code": np.arange(n_groups)})
    for suffix in (
        "variance_trace",
        "rms_radius",
        "median_radius",
        "p90_radius",
        "p95_radius",
    ):
        out[f"{prefix}_{suffix}"] = np.nan
    out.loc[present, f"{prefix}_variance_trace"] = variance_trace
    out.loc[present, f"{prefix}_rms_radius"] = np.sqrt(variance_trace)

    for idx, (code, start, count) in enumerate(
        zip(present, starts, counts, strict=True)
    ):
        block = sorted_values64[start : start + count]
        dist = np.sqrt(
            np.maximum(
                np.einsum("ij,ij->i", block - means[idx], block - means[idx]),
                0.0,
            )
        )
        out.loc[code, f"{prefix}_median_radius"] = float(np.quantile(dist, 0.50))
        out.loc[code, f"{prefix}_p90_radius"] = float(np.quantile(dist, 0.90))
        out.loc[code, f"{prefix}_p95_radius"] = float(np.quantile(dist, 0.95))
    return out.drop(columns=["group_code"])


def _normalize_log1p_chunk(matrix, *, target_sum: float) -> np.ndarray:
    if sparse.issparse(matrix):
        x = matrix.tocsr(copy=True).astype(np.float32)
        if x.nnz > 0:
            row_sums = np.asarray(x.sum(axis=1)).ravel().astype(np.float32, copy=False)
            scale = np.ones_like(row_sums, dtype=np.float32)
            nonzero = row_sums > 0
            scale[nonzero] = float(target_sum) / row_sums[nonzero]
            x.data *= np.repeat(scale, np.diff(x.indptr))
            x.data = np.log1p(x.data).astype(np.float32, copy=False)
        return x.toarray().astype(np.float32, copy=False)
    x = np.asarray(matrix, dtype=np.float32)
    row_sums = x.sum(axis=1, keepdims=True).astype(np.float32, copy=False)
    scale = np.ones_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    scale[nonzero] = float(target_sum) / row_sums[nonzero]
    return np.log1p(x * scale).astype(np.float32, copy=False)


def _accumulate_dense_group_sums(
    values: np.ndarray,
    group_codes: np.ndarray,
    sums: np.ndarray,
    sum_sq: np.ndarray,
) -> None:
    order = np.argsort(group_codes, kind="mergesort")
    codes = group_codes[order]
    sorted_values = values[order].astype(np.float64, copy=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    present = codes[starts]
    grouped_sums = np.add.reduceat(sorted_values, starts, axis=0)
    row_sq = np.einsum("ij,ij->i", sorted_values, sorted_values)
    grouped_sq = np.add.reduceat(row_sq, starts)
    sums[present] += grouped_sums
    sum_sq[present] += grouped_sq


def _expression_h5ad_group_stats(
    expression_h5ad: Path,
    *,
    obs_names: pd.Index,
    group_codes: np.ndarray,
    n_groups: int,
    target_sum: float,
    chunk_size: int,
    prefix: str = "expression_all_genes",
) -> pd.DataFrame:
    expr = sc.read_h5ad(expression_h5ad, backed="r")
    if int(expr.n_obs) != int(obs_names.shape[0]):
        raise ValueError(
            f"{expression_h5ad} has {int(expr.n_obs)} cells, expected {int(obs_names.shape[0])}."
        )
    if not expr.obs_names.equals(obs_names):
        raise ValueError(
            f"{expression_h5ad} obs_names do not match the spatial_ot cell output order."
        )
    n_vars = int(expr.n_vars)
    sums = np.zeros((n_groups, n_vars), dtype=np.float64)
    sum_sq = np.zeros(n_groups, dtype=np.float64)
    counts = np.bincount(group_codes, minlength=n_groups).astype(np.float64)
    for start in range(0, int(expr.n_obs), int(chunk_size)):
        end = min(start + int(chunk_size), int(expr.n_obs))
        chunk = _normalize_log1p_chunk(expr.X[start:end], target_sum=target_sum)
        _accumulate_dense_group_sums(
            chunk,
            group_codes[start:end],
            sums,
            sum_sq,
        )
    means = sums / np.maximum(counts[:, None], 1.0)
    variance_trace = np.maximum(
        sum_sq / np.maximum(counts, 1.0) - np.einsum("ij,ij->i", means, means),
        0.0,
    )
    out = pd.DataFrame(
        {
            f"{prefix}_source_dim": np.full(n_groups, n_vars, dtype=int),
            f"{prefix}_target_sum": np.full(n_groups, float(target_sum)),
            f"{prefix}_variance_trace": variance_trace,
            f"{prefix}_rms_radius": np.sqrt(variance_trace),
        }
    )
    expr.file.close()
    return out


def _probability_entropy_stats(
    probs: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    prefix: str,
) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    denom = np.log(max(probs.shape[1], 2))
    entropy = -np.sum(probs * np.log(probs), axis=1) / denom
    confidence = np.max(probs, axis=1)
    entropy_stats = _numeric_group_stats(
        entropy, group_codes, n_groups, f"{prefix}_entropy"
    )
    confidence_stats = _numeric_group_stats(
        confidence, group_codes, n_groups, f"{prefix}_confidence"
    )
    return pd.concat([entropy_stats, confidence_stats], axis=1)


def _robust_z_by_group(
    values: pd.Series,
    groups: pd.Series,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for _group, idx in groups.groupby(groups, observed=True).groups.items():
        group_values = values.loc[idx].astype(float)
        median = float(np.nanmedian(group_values))
        mad = float(np.nanmedian(np.abs(group_values - median)))
        if not np.isfinite(mad) or mad <= 1e-12:
            std = float(np.nanstd(group_values))
            scale = std if std > 1e-12 else np.nan
            result.loc[idx] = (
                (group_values - float(np.nanmean(group_values))) / scale
                if np.isfinite(scale)
                else 0.0
            )
        else:
            result.loc[idx] = 0.67448975 * (group_values - median) / mad
    return result


def _size_adjusted_z(values: pd.Series, n_cells: pd.Series) -> pd.Series:
    mask = np.isfinite(values) & np.isfinite(n_cells) & (n_cells > 0)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if int(mask.sum()) < 5:
        return out
    x = np.log1p(n_cells.loc[mask].astype(float).to_numpy())
    y = np.log1p(values.loc[mask].astype(float).to_numpy())
    degree = 2 if int(mask.sum()) >= 10 else 1
    coeffs = np.polyfit(x, y, degree)
    resid = y - np.polyval(coeffs, x)
    scale = float(np.nanstd(resid))
    if not np.isfinite(scale) or scale <= 1e-12:
        out.loc[mask] = 0.0
    else:
        out.loc[mask] = (resid - float(np.nanmean(resid))) / scale
    return out


def _percentile_rank(values: pd.Series) -> pd.Series:
    return values.astype(float).rank(pct=True, method="average").fillna(0.0)


def _merge_left(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([base.reset_index(drop=True), other.reset_index(drop=True)], axis=1)


def _plot_outputs(metrics: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(metrics["latent_rms_radius"].dropna(), bins=60, color="#3b82f6")
    axes[0, 0].set_title("Latent RMS Radius")
    axes[0, 0].set_xlabel("within-subregion RMS")
    axes[0, 0].set_ylabel("subregions")
    axes[0, 1].scatter(
        metrics["n_cells"],
        metrics["latent_rms_radius"],
        s=5,
        alpha=0.35,
        c=metrics["internal_heterogeneity_score"],
        cmap="viridis",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Latent Heterogeneity vs Size")
    axes[0, 1].set_xlabel("cells")
    axes[0, 1].set_ylabel("latent RMS")
    axes[1, 0].hist(
        metrics["shared_cell_type_norm_entropy"].dropna(),
        bins=40,
        color="#f97316",
    )
    axes[1, 0].set_title("Shared Cell-Type Diversity")
    axes[1, 0].set_xlabel("normalized entropy")
    axes[1, 0].set_ylabel("subregions")
    axes[1, 1].hist(
        metrics["internal_heterogeneity_score"].dropna(),
        bins=50,
        color="#14b8a6",
    )
    axes[1, 1].set_title("Composite Internal Heterogeneity")
    axes[1, 1].set_xlabel("score")
    axes[1, 1].set_ylabel("subregions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _read_spot_latent_npz(
    path: Path,
    group_codes: np.ndarray,
    n_groups: int,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "cell_indices" not in data or "atom_argmax" not in data:
            return None
        cell_indices = np.asarray(data["cell_indices"], dtype=np.int64)
        if cell_indices.shape[0] != group_codes.shape[0]:
            return None
        aligned_codes = group_codes[cell_indices]
        atom_argmax = np.asarray(data["atom_argmax"])
        out = pd.DataFrame({"subregion_id": np.arange(n_groups)})
        atom_df = pd.DataFrame({"group_code": aligned_codes, "atom": atom_argmax})
        counts = atom_df.groupby(["group_code", "atom"], observed=True).size()
        entropy, normalized, top_fraction = _entropy_from_counts(counts)
        out["spot_atom_entropy"] = np.nan
        out["spot_atom_norm_entropy"] = np.nan
        out["spot_atom_top_fraction"] = np.nan
        out.loc[entropy.index, "spot_atom_entropy"] = entropy.to_numpy()
        out.loc[normalized.index, "spot_atom_norm_entropy"] = normalized.to_numpy()
        out.loc[top_fraction.index, "spot_atom_top_fraction"] = top_fraction.to_numpy()
        if "normalized_posterior_entropy" in data:
            posterior_entropy = np.asarray(data["normalized_posterior_entropy"])
            posterior_stats = _numeric_group_stats(
                posterior_entropy,
                aligned_codes,
                n_groups,
                "spot_atom_posterior_entropy",
            )
            out = pd.concat(
                [out.drop(columns=["subregion_id"]), posterior_stats], axis=1
            )
            out.insert(0, "subregion_id", np.arange(n_groups))
        return out


def _ordered_top_columns(columns: Iterable[str]) -> list[str]:
    preferred = [
        "subregion_id",
        "sample_id",
        "cluster_id",
        "n_cells",
        "internal_heterogeneity_score",
        "expression_all_genes_rms_radius",
        "expression_all_genes_rms_size_adjusted_z",
        "latent_rms_radius",
        "latent_p90_radius",
        "latent_rms_size_adjusted_z",
        "latent_rms_cluster_robust_z",
        "shared_cell_type_norm_entropy",
        "shared_cell_type_top_fraction",
        "leiden_norm_entropy",
        "leiden_top_fraction",
        "cluster_prob_entropy_mean",
        "cluster_prob_confidence_mean",
        "spot_atom_norm_entropy",
        "spot_atom_posterior_entropy_mean",
        "spatial_rms_radius_um",
        "heterogeneity_flag_count",
        "flag_high_latent",
        "flag_extreme_latent",
        "flag_high_expression_all_genes",
        "flag_extreme_expression_all_genes",
        "flag_size_adjusted_expression_all_genes_outlier",
        "flag_size_adjusted_latent_outlier",
        "flag_cluster_adjusted_latent_outlier",
        "flag_high_cell_type_diversity",
        "flag_high_leiden_diversity",
        "flag_high_assignment_entropy",
    ]
    existing = set(columns)
    return [c for c in preferred if c in existing] + [
        c for c in columns if c not in preferred
    ]


def compute_internal_heterogeneity(
    run_dir: Path,
    *,
    h5ad_path: Path | None = None,
    subregions_path: Path | None = None,
    expression_h5ad_path: Path | None = None,
    embedding_key: str = DEFAULT_EMBEDDING_KEY,
    subregion_obs: str = DEFAULT_SUBREGION_OBS,
    output_prefix: str = "subregion_internal_heterogeneity",
    top_n: int = 100,
    expression_target_sum: float = 10000.0,
    expression_chunk_size: int = 50000,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    h5ad_path = h5ad_path or run_dir / "cells_multilevel_ot.h5ad"
    subregions_path = subregions_path or run_dir / "subregions_multilevel_ot.parquet"
    adata = sc.read_h5ad(h5ad_path, backed="r")
    if subregion_obs not in adata.obs.columns:
        raise KeyError(f"{h5ad_path} obs does not contain {subregion_obs!r}.")
    if embedding_key not in adata.obsm.keys():
        raise KeyError(f"{h5ad_path} obsm does not contain {embedding_key!r}.")

    obs = adata.obs.copy()
    subregion_values = pd.to_numeric(obs[subregion_obs], errors="coerce")
    valid = subregion_values.notna().to_numpy()
    if not np.all(valid):
        obs = obs.loc[valid].copy()
        subregion_values = subregion_values.loc[valid]
    subregion_labels = subregion_values.to_numpy(dtype=np.int64)
    subregion_ids = np.sort(np.unique(subregion_labels))
    group_codes = np.searchsorted(subregion_ids, subregion_labels)
    n_groups = int(subregion_ids.shape[0])

    metrics = pd.DataFrame({"subregion_id": subregion_ids})
    counts = np.bincount(group_codes, minlength=n_groups)
    metrics["n_cells"] = counts.astype(int)

    if subregions_path.exists():
        subregion_table = pd.read_parquet(subregions_path)
        keep_cols = [
            c
            for c in (
                "subregion_id",
                "sample_id",
                "cluster_id",
                "cluster_int",
                "assignment_margin",
                "shape_area",
                "shape_perimeter",
                "shape_compactness",
                "shape_aspect_ratio",
                "shape_eccentricity",
            )
            if c in subregion_table.columns
        ]
        metrics = metrics.merge(
            subregion_table[keep_cols].drop_duplicates("subregion_id"),
            on="subregion_id",
            how="left",
            suffixes=("", "_subregion_table"),
        )

    embedding = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    if not np.all(valid):
        embedding = embedding[valid]
    metrics = _merge_left(
        metrics,
        _embedding_group_stats(embedding, group_codes, n_groups, "latent"),
    )

    if expression_h5ad_path is not None:
        metrics = _merge_left(
            metrics,
            _expression_h5ad_group_stats(
                expression_h5ad_path,
                obs_names=obs.index,
                group_codes=group_codes,
                n_groups=n_groups,
                target_sum=expression_target_sum,
                chunk_size=expression_chunk_size,
            ),
        )

    coords = obs[[c for c in ("x_centroid", "y_centroid") if c in obs.columns]]
    if coords.shape[1] == 2:
        metrics = _merge_left(
            metrics,
            _embedding_group_stats(
                coords.to_numpy(dtype=np.float32),
                group_codes,
                n_groups,
                "spatial",
            ).rename(
                columns={
                    "spatial_variance_trace": "spatial_variance_trace_um2",
                    "spatial_rms_radius": "spatial_rms_radius_um",
                    "spatial_median_radius": "spatial_median_radius_um",
                    "spatial_p90_radius": "spatial_p90_radius_um",
                    "spatial_p95_radius": "spatial_p95_radius_um",
                }
            ),
        )

    for column, prefix in (
        ("shared_cell_type", "shared_cell_type"),
        ("leiden", "leiden"),
    ):
        metrics = metrics.merge(
            _categorical_diversity(obs, group_codes, subregion_ids, column, prefix),
            on="subregion_id",
            how="left",
        )

    for key, prefix in (
        ("mlot_cluster_probs", "cluster_prob"),
        ("mlot_feature_cluster_probs", "feature_cluster_prob"),
        ("mlot_context_cluster_probs", "context_cluster_prob"),
    ):
        if key in adata.obsm.keys():
            probs = np.asarray(adata.obsm[key], dtype=np.float32)
            if not np.all(valid):
                probs = probs[valid]
            metrics = _merge_left(
                metrics,
                _probability_entropy_stats(probs, group_codes, n_groups, prefix),
            )

    if "mlot_spot_latent_posterior_entropy" in obs.columns:
        entropy = pd.to_numeric(
            obs["mlot_spot_latent_posterior_entropy"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        metrics = _merge_left(
            metrics,
            _numeric_group_stats(
                entropy,
                group_codes,
                n_groups,
                "spot_latent_posterior_entropy",
            ),
        )

    spot_npz = _read_spot_latent_npz(
        run_dir / "spot_level_latent_multilevel_ot.npz", group_codes, n_groups
    )
    if spot_npz is not None:
        metrics = metrics.merge(spot_npz, on="subregion_id", how="left")

    if "cluster_id" not in metrics.columns and "cluster_int" in metrics.columns:
        metrics["cluster_id"] = metrics["cluster_int"].map(lambda v: f"C{int(v)}")

    metrics["latent_rms_size_adjusted_z"] = _size_adjusted_z(
        metrics["latent_rms_radius"], metrics["n_cells"]
    )
    if "cluster_id" in metrics.columns:
        metrics["latent_rms_cluster_robust_z"] = _robust_z_by_group(
            metrics["latent_rms_radius"], metrics["cluster_id"].astype("string")
        )
    else:
        metrics["latent_rms_cluster_robust_z"] = np.nan

    score_components = [
        "expression_all_genes_rms_radius",
        "latent_rms_radius",
        "latent_p90_radius",
        "shared_cell_type_norm_entropy",
        "leiden_norm_entropy",
        "cluster_prob_entropy_mean",
        "feature_cluster_prob_entropy_mean",
        "spot_atom_norm_entropy",
        "spot_atom_posterior_entropy_mean",
    ]
    available_components = [c for c in score_components if c in metrics.columns]
    component_ranks = pd.concat(
        [_percentile_rank(metrics[c]).rename(c) for c in available_components],
        axis=1,
    )
    metrics["internal_heterogeneity_score"] = component_ranks.mean(axis=1)

    thresholds = {
        "latent_rms_radius_p95": float(metrics["latent_rms_radius"].quantile(0.95)),
        "latent_rms_radius_p99": float(metrics["latent_rms_radius"].quantile(0.99)),
        "latent_p90_radius_p95": float(metrics["latent_p90_radius"].quantile(0.95)),
        "internal_heterogeneity_score_p95": float(
            metrics["internal_heterogeneity_score"].quantile(0.95)
        ),
    }
    if "expression_all_genes_rms_radius" in metrics.columns:
        thresholds["expression_all_genes_rms_radius_p95"] = float(
            metrics["expression_all_genes_rms_radius"].quantile(0.95)
        )
        thresholds["expression_all_genes_rms_radius_p99"] = float(
            metrics["expression_all_genes_rms_radius"].quantile(0.99)
        )
        metrics["expression_all_genes_rms_size_adjusted_z"] = _size_adjusted_z(
            metrics["expression_all_genes_rms_radius"], metrics["n_cells"]
        )
        metrics["flag_high_expression_all_genes"] = (
            metrics["expression_all_genes_rms_radius"]
            >= thresholds["expression_all_genes_rms_radius_p95"]
        )
        metrics["flag_extreme_expression_all_genes"] = (
            metrics["expression_all_genes_rms_radius"]
            >= thresholds["expression_all_genes_rms_radius_p99"]
        )
        metrics["flag_size_adjusted_expression_all_genes_outlier"] = (
            metrics["expression_all_genes_rms_size_adjusted_z"] >= 2.5
        )
    else:
        metrics["flag_high_expression_all_genes"] = False
        metrics["flag_extreme_expression_all_genes"] = False
        metrics["flag_size_adjusted_expression_all_genes_outlier"] = False
    metrics["flag_high_latent"] = (
        metrics["latent_rms_radius"] >= thresholds["latent_rms_radius_p95"]
    )
    metrics["flag_extreme_latent"] = (
        metrics["latent_rms_radius"] >= thresholds["latent_rms_radius_p99"]
    )
    metrics["flag_size_adjusted_latent_outlier"] = (
        metrics["latent_rms_size_adjusted_z"] >= 2.5
    )
    metrics["flag_cluster_adjusted_latent_outlier"] = (
        metrics["latent_rms_cluster_robust_z"] >= 2.5
    )
    metrics["flag_high_cell_type_diversity"] = (
        metrics.get("shared_cell_type_norm_entropy", pd.Series(0, index=metrics.index))
        >= 0.75
    )
    metrics["flag_high_leiden_diversity"] = (
        metrics.get("leiden_norm_entropy", pd.Series(0, index=metrics.index)) >= 0.75
    )
    if "cluster_prob_entropy_mean" in metrics.columns:
        thresholds["cluster_prob_entropy_mean_p95"] = float(
            metrics["cluster_prob_entropy_mean"].quantile(0.95)
        )
        metrics["flag_high_assignment_entropy"] = (
            metrics["cluster_prob_entropy_mean"]
            >= thresholds["cluster_prob_entropy_mean_p95"]
        )
    else:
        metrics["flag_high_assignment_entropy"] = False
    metrics["flag_high_composite"] = (
        metrics["internal_heterogeneity_score"]
        >= thresholds["internal_heterogeneity_score_p95"]
    )
    flag_cols = [c for c in metrics.columns if c.startswith("flag_")]
    metrics["heterogeneity_flag_count"] = metrics[flag_cols].sum(axis=1).astype(int)
    metrics["needs_manual_review"] = (
        metrics["flag_extreme_expression_all_genes"]
        | metrics["flag_size_adjusted_expression_all_genes_outlier"]
        | metrics["flag_extreme_latent"]
        | metrics["flag_size_adjusted_latent_outlier"]
        | metrics["flag_cluster_adjusted_latent_outlier"]
        | (
            metrics["flag_high_expression_all_genes"]
            & (
                metrics["flag_high_cell_type_diversity"]
                | metrics["flag_high_leiden_diversity"]
                | metrics["flag_high_assignment_entropy"]
                | metrics["flag_high_composite"]
            )
        )
        | (
            metrics["flag_high_latent"]
            & (
                metrics["flag_high_cell_type_diversity"]
                | metrics["flag_high_leiden_diversity"]
                | metrics["flag_high_assignment_entropy"]
                | metrics["flag_high_composite"]
            )
        )
    )

    metrics = metrics[_ordered_top_columns(metrics.columns)]
    all_path = run_dir / f"{output_prefix}.csv"
    top_path = run_dir / f"{output_prefix}_top{top_n}.csv"
    cluster_path = run_dir / f"{output_prefix}_by_cluster.csv"
    summary_path = run_dir / f"{output_prefix}_summary.json"
    plot_path = run_dir / f"{output_prefix}.png"

    ranked = metrics.sort_values(
        ["needs_manual_review", "internal_heterogeneity_score", "latent_rms_radius"],
        ascending=[False, False, False],
    )
    metrics.to_csv(all_path, index=False)
    ranked.head(top_n).to_csv(top_path, index=False)

    cluster_group_cols = ["cluster_id"] if "cluster_id" in metrics.columns else []
    if cluster_group_cols:
        cluster_summary = (
            metrics.groupby("cluster_id", observed=True)
            .agg(
                n_subregions=("subregion_id", "size"),
                n_review=("needs_manual_review", "sum"),
                median_cells=("n_cells", "median"),
                median_internal_score=("internal_heterogeneity_score", "median"),
                p95_internal_score=("internal_heterogeneity_score", lambda s: s.quantile(0.95)),
                median_latent_rms=("latent_rms_radius", "median"),
                p95_latent_rms=("latent_rms_radius", lambda s: s.quantile(0.95)),
                median_cell_type_entropy=(
                    "shared_cell_type_norm_entropy",
                    "median",
                ),
            )
            .reset_index()
        )
        cluster_summary.to_csv(cluster_path, index=False)

    _plot_outputs(metrics, plot_path)

    summary = {
        "run_dir": str(run_dir),
        "h5ad": str(h5ad_path),
        "subregions": str(subregions_path),
        "expression_h5ad": str(expression_h5ad_path)
        if expression_h5ad_path is not None
        else None,
        "embedding_key": embedding_key,
        "subregion_obs": subregion_obs,
        "n_cells": int(obs.shape[0]),
        "n_subregions": int(n_groups),
        "thresholds": thresholds,
        "counts": {
            "needs_manual_review": int(metrics["needs_manual_review"].sum()),
            "flag_high_latent": int(metrics["flag_high_latent"].sum()),
            "flag_extreme_latent": int(metrics["flag_extreme_latent"].sum()),
            "flag_size_adjusted_latent_outlier": int(
                metrics["flag_size_adjusted_latent_outlier"].sum()
            ),
            "flag_cluster_adjusted_latent_outlier": int(
                metrics["flag_cluster_adjusted_latent_outlier"].sum()
            ),
            "flag_high_cell_type_diversity": int(
                metrics["flag_high_cell_type_diversity"].sum()
            ),
            "flag_high_leiden_diversity": int(
                metrics["flag_high_leiden_diversity"].sum()
            ),
            "flag_high_assignment_entropy": int(
                metrics["flag_high_assignment_entropy"].sum()
            ),
            "flag_high_composite": int(metrics["flag_high_composite"].sum()),
            "flag_high_expression_all_genes": int(
                metrics["flag_high_expression_all_genes"].sum()
            ),
            "flag_extreme_expression_all_genes": int(
                metrics["flag_extreme_expression_all_genes"].sum()
            ),
            "flag_size_adjusted_expression_all_genes_outlier": int(
                metrics["flag_size_adjusted_expression_all_genes_outlier"].sum()
            ),
        },
        "outputs": {
            "all_subregions": str(all_path),
            "top_subregions": str(top_path),
            "by_cluster": str(cluster_path) if cluster_path.exists() else None,
            "plot": str(plot_path) if plot_path.exists() else None,
            "summary": str(summary_path),
        },
        "top_subregions": ranked.head(20)[
            [
                c
                for c in (
                    "subregion_id",
                    "sample_id",
                    "cluster_id",
                    "n_cells",
                    "internal_heterogeneity_score",
                    "expression_all_genes_rms_radius",
                    "expression_all_genes_rms_size_adjusted_z",
                    "latent_rms_radius",
                    "latent_rms_size_adjusted_z",
                    "latent_rms_cluster_robust_z",
                    "shared_cell_type_norm_entropy",
                    "leiden_norm_entropy",
                    "cluster_prob_entropy_mean",
                    "needs_manual_review",
                )
                if c in ranked.columns
            ]
        ].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(_as_jsonable(summary), indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute within-subregion heterogeneity from a spatial_ot cell-level "
            "output H5AD."
        )
    )
    parser.add_argument("run_dir", type=Path, help="spatial_ot output directory")
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--subregions", type=Path, default=None)
    parser.add_argument(
        "--expression-h5ad",
        type=Path,
        default=None,
        help=(
            "Optional source expression H5AD. When provided, the checker adds "
            "direct all-gene library-size-normalized log1p dispersion per subregion."
        ),
    )
    parser.add_argument("--embedding-key", default=DEFAULT_EMBEDDING_KEY)
    parser.add_argument("--subregion-obs", default=DEFAULT_SUBREGION_OBS)
    parser.add_argument("--output-prefix", default="subregion_internal_heterogeneity")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--expression-target-sum", type=float, default=10000.0)
    parser.add_argument("--expression-chunk-size", type=int, default=50000)
    args = parser.parse_args()
    summary = compute_internal_heterogeneity(
        args.run_dir,
        h5ad_path=args.h5ad,
        subregions_path=args.subregions,
        expression_h5ad_path=args.expression_h5ad,
        embedding_key=args.embedding_key,
        subregion_obs=args.subregion_obs,
        output_prefix=args.output_prefix,
        top_n=args.top_n,
        expression_target_sum=args.expression_target_sum,
        expression_chunk_size=args.expression_chunk_size,
    )
    print(json.dumps(_as_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
